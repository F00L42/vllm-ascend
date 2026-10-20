# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[3] / "benchmarks/scripts/serve_systemone_apc.py"
_SPEC = importlib.util.spec_from_file_location("systemone_apc_launch", _PATH)
_LAUNCH = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_LAUNCH)


@pytest.mark.parametrize("strategy", ["direct", "producer", "first_question"])
def test_existing_runtime_arguments_and_config_are_preserved(strategy):
    original = [
        "vllm",
        "serve",
        "/models/unchanged",
        "--port",
        "8181",
        "--tensor-parallel-size=4",
        "--max-num-batched-tokens",
        "2048",
        "--max-num-seqs",
        "8",
        "--dtype",
        "bfloat16",
        "--additional-config",
        '{"other":{"value":1},"systemone_config":{"mode":"answer_tokens"}}',
    ]
    command, config = _LAUNCH.build_command(original, strategy, True, 1536)
    assert command[:-4] == original[:-2]
    assert config["other"] == {"value": 1}
    assert config["systemone_config"]["mode"] == "answer_tokens"
    assert config["systemone_config"]["diagnostics"] is True
    shared = config["systemone_config"]["shared_prefix"]
    assert shared["enabled"] == (strategy != "direct")
    assert shared["verified_boundary_tokens"] == 1536
    assert command[-4:-2] == ["--scheduler-cls", _LAUNCH.TRACE_SCHEDULER]
    assert json.loads(command[-1]) == config


def test_measurement_keeps_original_scheduler():
    original = ["vllm", "serve", "model", "--scheduler-cls=custom.Scheduler"]
    command, _ = _LAUNCH.build_command(original, "direct", False, 1536)
    assert command[:4] == original
    with pytest.raises(ValueError, match="native synchronous"):
        _LAUNCH.build_command(original, "direct", True, 1536)


def test_invalid_config_is_not_silently_replaced():
    with pytest.raises(ValueError, match="JSON object"):
        _LAUNCH.build_command(["vllm", "serve", "model", "--additional-config=[]"], "direct", False, 1536)
    with pytest.raises(ValueError, match="repeated"):
        _LAUNCH.build_command(
            ["vllm", "serve", "model", "--additional-config={}", "--additional-config", "{}"], "direct", False, 1536
        )


def test_manifest_reuse_does_not_keep_diagnostic_scheduler(monkeypatch, tmp_path):
    original = [
        "vllm",
        "serve",
        "/models/existing",
        "--port",
        "8181",
        "--additional-config",
        '{"systemone_config":{"mode":"answer_tokens"}}',
    ]
    previous, _ = _LAUNCH.build_command(original, "direct", True, 1536)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"original_command": original, "effective_command": previous}), encoding="utf-8")
    observed = {}

    def capture(command, output_dir, manifest):
        observed.update(command=command, manifest=manifest)
        return 0

    monkeypatch.setattr(_LAUNCH, "run_server", capture)
    monkeypatch.setattr(_LAUNCH, "package_evidence", lambda name: {"sha": "test"})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "serve_systemone_apc.py",
            "--strategy",
            "first_question",
            "--from-manifest",
            str(path),
            "--output-dir",
            str(tmp_path / "formal"),
        ],
    )
    assert _LAUNCH.main() == 0
    assert "--scheduler-cls" not in observed["command"]
    assert observed["manifest"]["original_command"] == original
    config = observed["manifest"]["additional_config"]["systemone_config"]
    assert config["shared_prefix"]["strategy"] == "first_question"
    assert config["mode"] == "answer_tokens"
