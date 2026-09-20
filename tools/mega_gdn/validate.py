# SPDX-License-Identifier: Apache-2.0
"""Explicit, synchronous NPU validation of the P1 slot ABI (never a benchmark)."""

import argparse
import hashlib
import importlib
import json
import os
import platform
import subprocess
import sys
import traceback
from dataclasses import asdict
from pathlib import Path

import torch

from tools.mega_gdn.contract import slot_accesses
from tools.mega_gdn.reference import MegaGdnMtpResult, make_inputs, reference

ROOT = Path(__file__).resolve().parents[2]
# P1 reference tolerances from the pinned xllm-ops tests. These are NOT vLLM
# equivalence criteria; differences from the native chain require separate P2/P3 work.
TOLERANCES = {"conv_out": (8e-3, 1e-6), "conv_state": (0, 0), "ssm_state": (5e-3, 2.5e-5), "out": (5e-3, 2e-2)}


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def tensor_info(tensor: torch.Tensor) -> dict:
    return {"shape": list(tensor.shape), "stride": list(tensor.stride()), "dtype": str(tensor.dtype)}


def manifest() -> dict:
    return {
        "parent_commit": git("rev-parse", "HEAD"),
        "tracked_status": git("status", "--porcelain", "--untracked-files=no"),
        "submodules": git("submodule", "status", "--recursive"),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "environment": {
            key: os.environ.get(key)
            for key in (
                "ASCEND_HOME_PATH",
                "ASCEND_TOOLKIT_HOME",
                "ASCEND_CUSTOM_OPP_PATH",
                "LD_LIBRARY_PATH",
                "SOC_VERSION",
            )
        },
    }


def load_operator() -> object:
    torch_npu = importlib.import_module("torch_npu")
    if not torch_npu.npu.is_available():
        raise RuntimeError("NPU unavailable; this validation must not count skipped tests as PASS")
    device_name = torch_npu.npu.get_device_name(0)
    if "910B" not in device_name.upper():
        raise RuntimeError(f"P1 is restricted to Ascend 910B, found {device_name}")
    torch_npu.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    utils = importlib.import_module("vllm_ascend.utils")
    if not utils.enable_custom_op():
        raise RuntimeError("vllm-ascend custom operator loading failed")
    return torch.ops._C_ascend.npu_mega_gdn_mtp_decode


def native_convolution(cpu: dict[str, torch.Tensor]) -> torch.Tensor:
    """Independent existing Ascend Conv; compact read-only snapshots for this test.

    This mirrors the source project's reference conditioning. It is a diagnostic
    bridge, not a proposed production gather/scatter adapter.
    """
    batch, sequence, channels = cpu["qkv"].shape
    inputs = cpu["qkv"].reshape(-1, channels).to("npu:0")
    result = torch.empty_like(inputs)
    state = cpu["conv_state"].index_select(0, cpu["read_state_indices"].long()).to("npu:0")
    torch.ops._C_ascend.npu_causal_conv1d_custom(
        result,
        inputs,
        cpu["conv_weight"].to("npu:0"),
        conv_state=state,
        bias_opt=None,
        query_start_loc_opt=torch.arange(0, (batch + 1) * sequence, sequence, dtype=torch.int32, device="npu:0"),
        cache_indices_opt=torch.arange(batch, dtype=torch.int32, device="npu:0"),
        initial_state_mode_opt=None,
        num_accepted_tokens_opt=cpu["num_accepted_tokens"].to("npu:0"),
        activation_mode=1,
        pad_slot_id=-1,
        run_mode=1,
    )
    torch.npu.synchronize()
    return result.cpu().reshape(batch, sequence, channels)


def run_case(op, output: Path, speculative: int, batch: int, same_slot: bool, fla: bool, accepted: int) -> dict:
    case_id = f"k{speculative}-b{batch}-inplace{int(same_slot)}-fla{int(fla)}-m{accepted}"
    cpu = make_inputs(speculative, batch_size=batch, num_k_heads=2, num_v_heads=4, same_slot=same_slot)
    cpu["num_accepted_tokens"].fill_(accepted)
    slot_accesses(
        cpu["read_state_indices"].tolist(),
        cpu["write_state_indices"].tolist(),
        cpu["num_accepted_tokens"].tolist(),
        speculative + 1,
        cpu["conv_state"].shape[0],
    )
    conv_reference = native_convolution(cpu)
    expected = reference(**cpu, conv_output=conv_reference, fla_ssm_state_layout=fla)
    device = {name: value.to("npu:0") for name, value in cpu.items()}
    state_pointers = [device[name].data_ptr() for name in ("conv_state", "ssm_state")]
    # Save inputs before launch, so even an asynchronous kernel fault leaves a repro.
    torch.save(cpu, output / "last-inputs.pt")
    (output / "last-case.json").write_text(json.dumps({"case_id": case_id, "fla": fla}), encoding="utf-8")
    conv_out, result = op(**device, fla_ssm_state_layout=fla)
    torch.npu.synchronize()
    assert state_pointers == [device[name].data_ptr() for name in ("conv_state", "ssm_state")]
    actual = MegaGdnMtpResult(conv_out.cpu(), device["conv_state"].cpu(), device["ssm_state"].cpu(), result.cpu())
    errors = {}
    try:
        for name, (rtol, atol) in TOLERANCES.items():
            lhs, rhs = getattr(actual, name), getattr(expected, name)
            diff = (lhs.float() - rhs.float()).abs()
            errors[name] = {"max_abs": diff.max().item(), "rms": diff.square().mean().sqrt().item()}
            torch.testing.assert_close(lhs, rhs, rtol=rtol, atol=atol)
        # Untouched slots are bit-exact, even when numerical state tolerance is nonzero.
        written = set(cpu["write_state_indices"].tolist())
        sequence = speculative + 1
        for slot in range(cpu["conv_state"].shape[0]):
            if slot not in written:
                torch.testing.assert_close(actual.conv_state[slot], cpu["conv_state"][slot], rtol=0, atol=0)
                region = slice(slot * sequence, (slot + 1) * sequence)
                torch.testing.assert_close(actual.ssm_state[region], cpu["ssm_state"][region], rtol=0, atol=0)
    except AssertionError:
        torch.save({"expected": asdict(expected), "actual": asdict(actual)}, output / "failure.pt")
        raise
    return {
        "case_id": case_id,
        "status": "passed",
        "errors": errors,
        "inputs": {name: tensor_info(value) for name, value in cpu.items()},
    }


def check_dispatch_contract(op) -> None:
    cpu = make_inputs(1)
    meta = {name: value.to("meta") for name, value in cpu.items()}
    outputs = op(**meta)
    assert tuple(outputs[0].shape) == tuple(cpu["qkv"].shape)
    assert tuple(outputs[1].shape) == tuple(cpu["z"].shape)
    schema = op.default._schema
    for name in ("conv_state", "ssm_state"):
        argument = next(arg for arg in schema.arguments if arg.name == name)
        assert argument.alias_info is not None and argument.alias_info.is_write
    device = {name: value.to("npu:0") for name, value in cpu.items()}
    # Preserve shape while making a mutable state non-contiguous. Must reject prelaunch.
    device["ssm_state"] = device["ssm_state"].transpose(-1, -2)
    try:
        op(**device)
    except RuntimeError as error:
        if "must be contiguous" not in str(error):
            raise
    else:
        raise AssertionError("non-contiguous mutable state was not rejected")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument(
        "--matrix", action="store_true", help="cover every S=2..17, both layouts and accepted endpoints"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"status": "failed", "cases": []}
    torch.set_num_threads(1)
    try:
        report["manifest"] = manifest()
        if report["manifest"]["parent_commit"] != args.expected_commit or report["manifest"]["tracked_status"]:
            raise RuntimeError("expected a clean checkout at --expected-commit")
        op = load_operator()
        extension = importlib.import_module("vllm_ascend.vllm_ascend_C")
        extension_path = Path(extension.__file__).resolve()
        if not extension_path.is_relative_to(ROOT):
            raise RuntimeError(f"loaded extension outside checkout: {extension_path}")
        report["loaded"] = {
            "extension": str(extension_path),
            "sha256": hashlib.sha256(extension_path.read_bytes()).hexdigest(),
            "device": torch.npu.get_device_name(0),
            "torch_npu": importlib.import_module("torch_npu").__version__,
            "opp_path": os.environ.get("ASCEND_CUSTOM_OPP_PATH"),
        }
        check_dispatch_contract(op)
        for speculative in range(1, 17) if args.matrix else (1, 2):
            for fla in (False, True):
                for same_slot in (True, False):
                    for accepted in (1, speculative + 1):
                        row = run_case(op, args.output, speculative, 2, same_slot, fla, accepted)
                        report["cases"].append(row)
                        print(row["case_id"], "PASS", flush=True)
        report["status"] = "passed"
    except Exception:
        report["error"] = traceback.format_exc()
        print(report["error"], file=sys.stderr)
    finally:
        (args.output / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
