# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Run an existing serve command with one SystemOne strategy and save evidence."""

import argparse
import copy
import importlib.metadata
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

TRACE_SCHEDULER = "vllm_ascend.core.systemone_trace_scheduler.SystemOneTraceScheduler"


def take_option(command, option):
    """Remove one option while preserving every unrelated argument verbatim."""
    remaining = []
    values = []
    index = 0
    while index < len(command):
        argument = command[index]
        if argument == option:
            if index + 1 >= len(command):
                raise ValueError(f"Missing value for {option}")
            values.append(command[index + 1])
            index += 2
        elif argument.startswith(option + "="):
            values.append(argument.split("=", 1)[1])
            index += 1
        else:
            remaining.append(argument)
            index += 1
    if len(values) > 1:
        raise ValueError(f"Ambiguous repeated option: {option}")
    return remaining, values[0] if values else None


def build_command(command, strategy, trace_scheduler, boundary_tokens):
    if not command:
        raise ValueError("Pass the unchanged vllm serve command after --")
    command, additional = take_option(command, "--additional-config")
    config = json.loads(additional) if additional else {}
    if not isinstance(config, dict):
        raise ValueError("--additional-config must be a JSON object")
    config = copy.deepcopy(config)
    readout = config.setdefault("systemone_config", {})
    readout["diagnostics"] = True
    shared = readout.setdefault("shared_prefix", {})
    shared.update(
        enabled=strategy != "direct",
        strategy="producer" if strategy == "direct" else strategy,
        verified_boundary_tokens=boundary_tokens,
    )
    shared.setdefault("min_shared_tokens", 4096)
    if trace_scheduler:
        command, scheduler = take_option(command, "--scheduler-cls")
        if scheduler not in (None, "vllm.v1.core.sched.scheduler.Scheduler", TRACE_SCHEDULER):
            raise ValueError("Trace requires the native synchronous scheduler; found " + scheduler)
        command += ["--scheduler-cls", TRACE_SCHEDULER]
    return command + ["--additional-config", json.dumps(config)], config


def git_evidence(directory):
    def run(*arguments):
        result = subprocess.run(
            ["git", "-C", str(directory), *arguments],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        return result.stdout.strip() if result.returncode == 0 else None

    try:
        root = run("rev-parse", "--show-toplevel")
        return {
            "source": str(directory),
            "root": root,
            "sha": run("rev-parse", "HEAD") if root else None,
            "status": run("status", "--porcelain=v1") if root else None,
            "diff": run("diff", "HEAD", "--") if root else None,
        }
    except OSError as error:
        return {"source": str(directory), "sha": None, "error": str(error)}


def package_evidence(name):
    spec = importlib.util.find_spec(name)
    if spec is None or spec.origin is None:
        return {"source": None, "sha": None, "error": "package source unavailable"}
    evidence = git_evidence(Path(spec.origin).resolve().parent)
    try:
        evidence["version"] = importlib.metadata.version(name.replace("_", "-"))
    except importlib.metadata.PackageNotFoundError:
        evidence["version"] = None
    return evidence


def run_server(command, output_dir, manifest):
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    # Exclusive creation prevents a second run from erasing a frozen baseline.
    with manifest_path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
    print(f"Evidence: {manifest_path}; server output: {output_dir / 'server.log'}", flush=True)
    with (output_dir / "server.log").open("xb") as log:
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=os.name == "posix"
        )
        try:
            for line in iter(process.stdout.readline, b""):
                log.write(line)
                log.flush()
                sys.stdout.buffer.write(line)
                sys.stdout.buffer.flush()
            return process.wait()
        except KeyboardInterrupt:
            # Signal only the process group this wrapper created, never other servers.
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGINT)
            else:
                process.terminate()
            return process.wait()
        finally:
            process.stdout.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", choices=("direct", "producer", "first_question"), required=True)
    parser.add_argument("--trace-scheduler", action="store_true", help="Mechanism runs only; omit for performance")
    parser.add_argument("--boundary-tokens", type=int, default=1536)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--from-manifest", type=Path, help="Reuse the original command saved by the first strategy run")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="-- followed by your existing vllm serve command")
    args = parser.parse_args()
    if args.boundary_tokens <= 0:
        parser.error("--boundary-tokens must be positive")
    original = args.command[1:] if args.command[:1] == ["--"] else args.command
    if args.from_manifest:
        if original:
            parser.error("Use either --from-manifest or a command after --")
        original = json.loads(args.from_manifest.read_text(encoding="utf-8"))["original_command"]
    try:
        command, config = build_command(original, args.strategy, args.trace_scheduler, args.boundary_tokens)
    except (ValueError, TypeError, AttributeError) as error:
        parser.error(str(error))
    manifest = {
        "schema_version": 1,
        "created_wall_ns": time.time_ns(),
        "cwd": str(Path.cwd()),
        "python": sys.executable,
        "original_command": original,
        "effective_command": command,
        "additional_config": config,
        "strategy": args.strategy,
        "trace_scheduler": args.trace_scheduler,
        "vllm": package_evidence("vllm"),
        "vllm_ascend": package_evidence("vllm_ascend"),
    }
    return run_server(command, args.output_dir, manifest)


if __name__ == "__main__":
    sys.exit(main())
