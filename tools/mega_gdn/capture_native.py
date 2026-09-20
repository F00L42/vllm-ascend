# SPDX-License-Identifier: Apache-2.0
"""Opt-in native GDN capture for P0. Synchronizes selected calls; not for timing."""

import argparse
import importlib
import json
import os
from pathlib import Path
from unittest.mock import patch

import torch

from tools.mega_gdn.validate import manifest, tensor_info


def cpu_copy(value):
    return value.detach().cpu().clone() if isinstance(value, torch.Tensor) else value


def cache_snapshot(layer, metadata) -> dict:
    """Gather only referenced blocks, retaining IDs, original strides and PAD rows."""
    indices = metadata.spec_state_indices_tensor
    if indices is None:
        raise RuntimeError("selected call is not a speculative GDN call")
    indices_cpu = cpu_copy(indices)
    conv, ssm = layer.kv_cache
    result = {"physical_table": indices_cpu, "conv_layout": tensor_info(conv), "ssm_layout": tensor_info(ssm)}
    for name, pool, ids in (("conv", conv, indices_cpu[:, 0]), ("ssm", ssm, indices_cpu.flatten())):
        # Padding is recorded in physical_table and excluded from diagnostic gather.
        ids = torch.unique(ids[ids >= 0]).to(torch.int64)
        if ids.numel() and ids.max().item() >= pool.shape[0]:
            raise RuntimeError(f"{name} physical index outside state pool")
        result[f"{name}_ids"] = ids
        result[f"{name}_blocks"] = cpu_copy(pool.index_select(0, ids.to(pool.device)))
    return result


def capture_hook(original, output: Path, selected_layer: str, limit: int, records: list, hooks: list):
    def wrapped(layer, mixed_qkv, b, a, core_attn_out, *args, **kwargs):
        context = importlib.import_module("vllm.forward_context").get_forward_context()
        metadata = context.attn_metadata
        metadata = metadata.get(layer.prefix) if isinstance(metadata, dict) else None
        selected = (
            layer.prefix == selected_layer
            and len(records) < limit
            and metadata is not None
            and metadata.num_spec_decodes > 0
            and metadata.num_prefills == 0
            and metadata.num_decodes == 0
        )
        if not selected:
            return original(layer, mixed_qkv, b, a, core_attn_out, *args, **kwargs)
        record = {
            "layer": layer.prefix,
            "call": len(records),
            "before": cache_snapshot(layer, metadata),
            "qkv": cpu_copy(mixed_qkv),
            "b": cpu_copy(b),
            "a": cpu_copy(a),
            "metadata": {
                name: cpu_copy(getattr(metadata, name, None))
                for name in (
                    "num_actual_tokens",
                    "num_prefills",
                    "num_decodes",
                    "num_spec_decodes",
                    "spec_query_start_loc",
                    "spec_token_indx",
                    "spec_sequence_masks",
                    "num_accepted_tokens",
                )
            },
        }
        spec = metadata.spec_decode_metadata
        record["actual_seq_lengths"] = cpu_copy(spec.actual_seq_lengths)
        record["conv_metadata"] = {
            name: cpu_copy(getattr(spec.spec_causal_conv1d, name))
            for name in ("query_start_loc", "cache_indices", "num_accepted_tokens")
        }
        records.append({"layer": layer.prefix, "call": record["call"]})
        torch.save(record, output / f"call-{record['call']:03d}-before.pt")
        value = original(layer, mixed_qkv, b, a, core_attn_out, *args, **kwargs)
        record["after"] = cache_snapshot(layer, metadata)
        record["core_output"] = cpu_copy(core_attn_out)
        gdn = importlib.import_module("vllm_ascend.ops.gdn")
        record["weights"] = {
            "conv_weight": cpu_copy(gdn._get_packed_conv_weights(layer)),
            "a_log": cpu_copy(layer.A_log),
            "dt_bias": cpu_copy(layer.dt_bias),
            "norm_weight": cpu_copy(layer.norm.weight),
            "conv_bias": cpu_copy(layer.conv1d.bias),
            "norm_eps": layer.norm.eps,
            "norm_before_gate": layer.norm.norm_before_gate,
            "norm_group_size": layer.norm.group_size,
            "conv_activation": layer.activation,
        }

        def norm_hook(module, inputs, norm_output):
            record["norm_input"] = cpu_copy(inputs[0])
            record["z"] = cpu_copy(inputs[1])
            record["norm_output"] = cpu_copy(norm_output)
            torch.save(record, output / f"call-{record['call']:03d}.pt")
            handle.remove()

        handle = layer.norm.register_forward_hook(norm_hook)
        hooks.append(handle)
        return value

    return wrapped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-args", type=Path, required=True, help="JSON object passed to vllm.LLM")
    parser.add_argument("--prompts", type=Path, required=True, help="JSON list of prompts")
    parser.add_argument("--layer", required=True, help="exact GDN layer prefix, e.g. model.layers.0.linear_attn")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calls", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=32)
    args = parser.parse_args()
    if os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") != "0":
        raise RuntimeError("set existing vLLM option VLLM_ENABLE_V1_MULTIPROCESSING=0 for this in-process diagnostic")
    engine_args = json.loads(args.engine_args.read_text(encoding="utf-8"))
    if not engine_args.get("enforce_eager") or not engine_args.get("speculative_config"):
        raise ValueError("P0 capture requires enforce_eager=true and the server's known-working speculative_config")
    if engine_args.get("tensor_parallel_size", 1) != 1 or engine_args.get("pipeline_parallel_size", 1) != 1:
        raise ValueError("first capture supports TP=PP=1; worker-process/TP capture is a later round")
    if engine_args.get("async_scheduling", False) or args.calls < 1:
        raise ValueError("first capture requires synchronous scheduling and calls>=1")
    if engine_args.get("kv_transfer_config") or engine_args.get("enable_prefix_caching", False):
        raise ValueError("first capture excludes KV connectors and prefix caching")
    engine_args["async_scheduling"] = False
    engine_args["enable_prefix_caching"] = False
    args.output.mkdir(parents=True, exist_ok=False)
    records, hooks = [], []
    report = {"manifest": manifest(), "engine_args": engine_args, "captures": records, "status": "failed"}
    try:
        importlib.import_module("torch_npu")
        gdn = importlib.import_module("vllm_ascend.ops.gdn")
        vllm = importlib.import_module("vllm")
        original = gdn.AscendGatedDeltaNetAttention._forward_core
        wrapper = capture_hook(original, args.output, args.layer, args.calls, records, hooks)
        with patch.object(gdn.AscendGatedDeltaNetAttention, "_forward_core", wrapper):
            llm = vllm.LLM(**engine_args)
            outputs = llm.generate(
                json.loads(args.prompts.read_text(encoding="utf-8")),
                vllm.SamplingParams(temperature=0, max_tokens=args.max_tokens),
            )
        (args.output / "outputs.json").write_text(
            json.dumps([result.outputs[0].text for result in outputs], ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if len(list(args.output.glob("call-???.pt"))) != args.calls:
            raise RuntimeError("insufficient complete captures: verify layer prefix, spec path and process placement")
        report["status"] = "captured_pending_analysis"
    finally:
        for handle in hooks:
            handle.remove()
        (args.output / "manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
