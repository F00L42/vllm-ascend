# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Join HTTP observations to native scheduler steps, or compare strategy runs."""

import argparse
import json
import math
import sys
from pathlib import Path

from benchmark_systemone_apc import PERCENTILES, compare_questions, percentile, question_results


def strip_ansi(line):
    pieces = line.split("\x1b[")
    cleaned = [pieces[0]]
    for piece in pieces[1:]:
        end = next((index for index, character in enumerate(piece) if "@" <= character <= "~"), None)
        cleaned.append(piece[end + 1 :] if end is not None else piece)
    return "".join(cleaned)


def records(value):
    """Walk the benchmark report without treating response internals as trials."""
    if isinstance(value, dict):
        if "ok" in value and "request_id" in value:
            yield value
        else:
            for child in value.values():
                yield from records(child)
    elif isinstance(value, list):
        for child in value:
            yield from records(child)


def load_log(path):
    steps, runtime, errors = [], [], []
    with path.open(encoding="utf-8", errors="replace") as stream:
        for number, line in enumerate(stream, 1):
            line = strip_ansi(line)
            for marker, target in (
                ("SystemOne scheduler runtime ", runtime),
                ("SystemOne runtime ", runtime),
                ("SystemOne scheduler ", steps),
            ):
                if marker not in line:
                    continue
                content = line.split(marker, 1)[1].strip()
                try:
                    item = json.loads(content)
                    item["log_line"] = number
                    target.append(item)
                except (ValueError, TypeError):
                    errors.append(f"Malformed trace at {path}:{number}")
                break
    return steps, runtime, errors


def internal_matches(external_id, internal_id):
    # v0.27.1 InputProcessor.assign_request_id appends exactly 8 UUID hex digits.
    # Exact matching also supports VLLM_DISABLE_REQUEST_ID_RANDOMIZATION.
    if internal_id == external_id:
        return True
    prefix = external_id + "-"
    suffix = internal_id[len(prefix) :]
    return internal_id.startswith(prefix) and len(suffix) == 8 and all(c in "0123456789abcdef" for c in suffix)


def trace_errors(external, trace, scheduled, prompt, cached):
    errors = []
    first_step, first = scheduled[0]
    last_step, last = scheduled[-1]
    if first["computed_before"] != cached:
        errors.append(f"First chunk does not start at initial APC hit: {external}")
    if last["computed_before"] + last["num_scheduled_tokens"] != prompt:
        errors.append(f"Last chunk does not reach prompt end: {external}")
    observed_steps = [
        step["step_id"] for step, _ in trace if first_step["step_id"] <= step["step_id"] <= last_step["step_id"]
    ]
    if observed_steps != list(range(first_step["step_id"], last_step["step_id"] + 1)):
        errors.append(f"Missing or duplicate live-request scheduler steps: {external}")
    for (_, previous), (_, current) in zip(scheduled, scheduled[1:]):
        if (current.get("num_preemptions") or 0) > (previous.get("num_preemptions") or 0):
            # A resumed admission may do a new cache lookup. Preserve this in
            # the timeline and require review rather than infer a hit delta.
            errors.append(f"Preempted trace needs recomputation review: {external}")
        elif current["computed_before"] != previous["computed_before"] + previous["num_scheduled_tokens"]:
            errors.append(f"Gap in computed-token progress: {external}")
    return errors


def join_trace(steps, reports):
    rows, errors = [], []
    by_request = {}
    for step in steps:
        for request in step.get("requests", []):
            by_request.setdefault(request["request_id"], []).append((step, request))
    seen = set()
    for report in reports:
        for record in records(report):
            if not record.get("ok"):
                errors.append(f"Failed HTTP capture: {record.get('request_id')}: {record.get('error')}")
            response = record.get("response") or {}
            diagnostic = response.get("diagnostics") or {}
            details = list(diagnostic.get("question_details", []))
            if diagnostic.get("producer"):
                details.insert(0, {**diagnostic["producer"], "question_id": "<producer>"})
            if not details:
                errors.append(f"Missing endpoint diagnostics: {record.get('request_id')}")
            for detail in details:
                external = detail["engine_request_id"]
                if external in seen:
                    continue
                seen.add(external)
                matches = [key for key in by_request if internal_matches(external, key)]
                if len(matches) != 1:
                    errors.append(f"Expected one scheduler identity for {external}, found {len(matches)}")
                    continue
                trace = by_request[matches[0]]
                scheduled = [(step, req) for step, req in trace if req["num_scheduled_tokens"] > 0]
                if not scheduled:
                    errors.append(f"No scheduled tokens for {external}")
                    continue
                first_step, first = scheduled[0]
                total = sum(req["num_scheduled_tokens"] for _, req in scheduled)
                prompt = detail["prompt_tokens"]
                cached = first["num_cached_tokens"]
                if cached is None:
                    errors.append(f"Missing first APC hit count for {external}")
                preemptions = max(req.get("num_preemptions") or 0 for _, req in trace)
                if not preemptions and cached != detail.get("cached_tokens"):
                    errors.append(f"Scheduler and endpoint cache counts differ: {external}")
                errors.extend(trace_errors(external, trace, scheduled, prompt, cached))
                rows.append(
                    {
                        "parent_request_id": diagnostic.get("parent_request_id"),
                        "question_id": detail.get("question_id"),
                        "engine_request_id": external,
                        "scheduler_request_id": matches[0],
                        "strategy": diagnostic.get("strategy"),
                        "http_ok": record.get("ok", False),
                        "prompt_tokens": prompt,
                        "first_step": first_step["step_id"],
                        "first_cached_tokens": cached,
                        "endpoint_cached_tokens": detail.get("cached_tokens"),
                        "scheduled_new_tokens": total,
                        "scheduled_steps": len(scheduled),
                        "preemptions": preemptions,
                        "scheduled_minus_uncached_prompt": None if cached is None else total - (prompt - cached),
                        "timeline": [
                            {
                                "step_id": step["step_id"],
                                "log_line": step.get("log_line"),
                                "monotonic_ns": step["monotonic_ns"],
                                **req,
                            }
                            for step, req in trace
                        ],
                    }
                )
                if not record.get("ok"):
                    errors.append(f"HTTP did not complete successfully: {external}")
    if not rows:
        errors.append("No correlated request rows")
    return {"passed": not errors, "errors": errors, "requests": rows}


def compare_runs(reports):
    errors, comparisons, summaries = [], [], []
    if len(reports) < 2:
        return {
            "passed": False,
            "errors": ["Compare requires at least two result files"],
            "comparisons": [],
            "summaries": [],
        }
    reference = reports[0]
    runtime_reference = None
    phases = reference.get("phases", {})
    phase_name = "correctness" if "correctness" in phases else "performance" if "performance" in phases else None
    if phase_name is None:
        errors.append("Reference has no correctness or performance phase")
    parameters = reference.get("parameters", {})
    for key in ("atol", "rtol"):
        if not _finite_nonnegative(parameters.get(key)):
            errors.append(f"Reference numerical tolerance is missing or invalid: {key}")
    for report in reports:
        if not report.get("input_sha256") or report.get("input_sha256") != reference.get("input_sha256"):
            errors.append("Input file SHA256 is missing or differs")
        for key in ("atol", "rtol"):
            if report.get("parameters", {}).get(key) != parameters.get(key):
                errors.append(f"Numerical tolerance differs: {key}")
        if report.get("question_ids") != reference.get("question_ids"):
            errors.append("Original question IDs or ordering differ")
        phase = report.get("phases", {}).get(phase_name, {})
        if not phase or phase.get("skipped"):
            errors.append(f"A {phase_name or 'required'} phase is missing or skipped")
        if phase_name == "correctness" and not phase.get("passed"):
            errors.append("A correctness phase is missing or failed")
        if phase_name == "performance" and (not phase.get("data_passed") or not report.get("data_passed")):
            errors.append("A performance run has data_passed=false or missing")
        observed_records = list(records(report))
        if not observed_records:
            errors.append("Run contains no HTTP observations")
        for record in observed_records:
            if not record.get("ok"):
                errors.append(f"Failed HTTP capture: {record.get('request_id')}")
            runtime = (record.get("response") or {}).get("diagnostics", {}).get("runtime")
            if not isinstance(runtime, dict):
                errors.append(f"Missing runtime configuration: {record['request_id']}")
                continue
            runtime = normalize_runtime(runtime)
            if runtime_reference is None:
                runtime_reference = runtime
            elif runtime != runtime_reference:
                errors.append(f"Runtime configuration differs: {record['request_id']}")
        summaries.append(
            {
                "strategy": report.get("parameters", {}).get("expected_strategy"),
                "phases": {
                    name: {key: value for key, value in phase.items() if key not in ("trials", "pairs", "baseline")}
                    for name, phase in report.get("phases", {}).items()
                },
            }
        )
    if phase_name == "correctness":
        _compare_correctness(reports, comparisons, errors)
    elif phase_name == "performance":
        _compare_performance(reports, comparisons, summaries, errors)
    return {
        "passed": not errors,
        "phase": phase_name,
        "reference_strategy": parameters.get("expected_strategy"),
        "errors": errors,
        "comparisons": comparisons,
        "summaries": summaries,
    }


def _finite_nonnegative(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def _compare_capture(left, right, parameters):
    if not left.get("ok") or not right.get("ok"):
        return {"passed": False, "error": "One or both HTTP captures failed"}
    try:
        left_questions, right_questions = question_results(left), question_results(right)
        if left_questions.keys() != right_questions.keys():
            return {"passed": False, "error": "Question IDs differ"}
        if any(not _finite_nonnegative(parameters.get(key)) for key in ("atol", "rtol")):
            return {"passed": False, "error": "Numerical tolerance is missing or invalid"}
        return compare_questions(left_questions, right_questions, atol=parameters["atol"], rtol=parameters["rtol"])
    except (KeyError, TypeError, ValueError) as exc:
        return {"passed": False, "error": f"Incomplete question observation: {exc}"}


def _compare_correctness(reports, comparisons, errors):
    reference = reports[0]
    baseline_trials = reference.get("phases", {}).get("correctness", {}).get("trials", [])
    if not baseline_trials:
        errors.append("Reference correctness trials are missing")
    for report in reports[1:]:
        trials = report.get("phases", {}).get("correctness", {}).get("trials", [])
        if len(trials) != len(baseline_trials):
            errors.append("Correctness repeat counts differ")
        for left, right in zip(baseline_trials, trials):
            for variant in ("batch", "reversed", "duplicated"):
                comparison = _compare_capture(
                    left.get(variant, {}), right.get(variant, {}), reference.get("parameters", {})
                )
                comparisons.append(
                    {
                        "strategy": report.get("parameters", {}).get("expected_strategy"),
                        "repeat": left.get("repeat"),
                        "variant": variant,
                        **comparison,
                    }
                )
                if not comparison["passed"]:
                    errors.append(f"Cross-strategy probability/answer mismatch: {variant}")


def _performance_pairs(report, errors):
    parameters = report.get("parameters", {})
    phase = report.get("phases", {}).get("performance", {})
    strategy = parameters.get("expected_strategy")
    pairs = phase.get("pairs", [])
    repeats = parameters.get("repeats")
    if type(repeats) is not int or repeats <= 0:
        errors.append(f"Missing or invalid performance repeat count: {strategy}")
        repeats = 0
    expected = {(repeat, cache) for repeat in range(repeats) for cache in ("logical_cold", "warm_repeat")}
    indexed = {}
    for pair in pairs:
        errors_before = len(errors)
        key = pair.get("repeat"), pair.get("cache")
        if key in indexed or key not in expected:
            errors.append(f"Duplicate or unexpected performance pair: {strategy}, {key}")
        if not pair.get("data_passed"):
            errors.append(f"Invalid performance pair: {strategy}, {key}")
        for label in ("one", "many"):
            record = pair.get(label, {})
            if not record.get("ok") or not _finite_nonnegative(record.get("elapsed_ms")) or record["elapsed_ms"] <= 0:
                errors.append(f"Missing valid HTTP latency: {strategy}, {key}, {label}")
            question_ids = [parameters.get("single_question")] if label == "one" else report.get("question_ids")
            if not question_ids or record.get("question_ids") != question_ids:
                errors.append(f"Incomplete performance question set: {strategy}, {key}, {label}")
            comparison = _compare_capture(record, record, parameters)
            if not comparison["passed"]:
                errors.append(f"Incomplete performance probability evidence: {strategy}, {key}, {label}")
        indexed[key] = {**pair, "comparison_data_passed": len(errors) == errors_before}
    if not expected or set(indexed) != expected:
        errors.append(f"Performance pair coverage differs from declared repeats: {strategy}")
    return indexed, expected


def _performance_summary(indexed, expected, source_pairs):
    summary = {}
    for cache in ("logical_cold", "warm_repeat"):
        selected = [pair for (_, pair_cache), pair in indexed.items() if pair_cache == cache]
        valid = [
            pair
            for pair in selected
            if pair.get("comparison_data_passed")
            and all(
                pair.get(label, {}).get("ok")
                and _finite_nonnegative(pair[label].get("elapsed_ms"))
                and pair[label]["elapsed_ms"] > 0
                for label in ("one", "many")
            )
        ]
        summary[cache] = {
            "expected_pairs": sum(pair_cache == cache for _, pair_cache in expected),
            "observed_pairs": sum(pair.get("cache") == cache for pair in source_pairs),
            "valid_pairs": len(valid),
            "invalid_pairs": sum(pair.get("cache") == cache for pair in source_pairs) - len(valid),
            "many_latency_ms_valid_pairs": {
                f"p{percent}": percentile([pair["many"]["elapsed_ms"] for pair in valid], percent)
                for percent in PERCENTILES
            },
            "one_latency_ms_valid_pairs": {
                f"p{percent}": percentile([pair["one"]["elapsed_ms"] for pair in valid], percent)
                for percent in PERCENTILES
            },
            "target_failed_valid_pairs": sum(not pair.get("target_passed") for pair in valid),
        }
    return summary


def _compare_performance(reports, comparisons, summaries, errors):
    reference = reports[0]
    indexed_runs = []
    for report, summary in zip(reports, summaries):
        parameters = report.get("parameters", {})
        for field in ("seed", "single_question", "repeats", "ratio_limit"):
            if parameters.get(field) != reference.get("parameters", {}).get(field):
                errors.append(f"Performance protocol parameter differs: {field}")
        indexed, expected = _performance_pairs(report, errors)
        indexed_runs.append(indexed)
        phase = report.get("phases", {}).get("performance", {})
        summary["data_passed"] = phase.get("data_passed", False)
        summary["target_passed"] = phase.get("target_passed")
        summary["performance"] = _performance_summary(indexed, expected, phase.get("pairs", []))
    baseline = indexed_runs[0]
    for report, indexed, summary in zip(reports[1:], indexed_runs[1:], summaries[1:]):
        if set(indexed) != set(baseline):
            errors.append("Cross-strategy performance pair keys differ")
        for key in sorted(set(indexed) & set(baseline)):
            for label in ("one", "many"):
                comparison = _compare_capture(
                    baseline[key].get(label, {}), indexed[key].get(label, {}), reference.get("parameters", {})
                )
                comparisons.append(
                    {
                        "strategy": report.get("parameters", {}).get("expected_strategy"),
                        "repeat": key[0],
                        "cache": key[1],
                        "variant": label,
                        **comparison,
                    }
                )
                if not comparison["passed"]:
                    errors.append(f"Cross-strategy probability/answer mismatch: {key}, {label}")
        for cache, group in summary["performance"].items():
            reference_latency = summaries[0]["performance"][cache]["many_latency_ms_valid_pairs"]
            group["many_change_vs_reference"] = {}
            for quantile, latency in group["many_latency_ms_valid_pairs"].items():
                previous = reference_latency[quantile]
                group["many_change_vs_reference"][quantile] = {
                    "absolute_ms": None if latency is None or previous is None else latency - previous,
                    "relative_percent": None
                    if latency is None or previous is None or previous <= 0
                    else (latency / previous - 1) * 100,
                }


def normalize_runtime(runtime):
    shared = runtime.get("shared_prefix", {})
    return {
        **runtime,
        "shared_prefix": {
            key: value
            for key, value in shared.items()
            if key not in ("enabled", "strategy", "capability_reason", "min_shared_tokens_applies")
        },
    }


def manifest_command(manifest):
    command = list(manifest.get("effective_command", []))
    if "--additional-config" in command:
        index = command.index("--additional-config")
        del command[index : index + 2]
    config = dict(manifest.get("additional_config", {}))
    config["systemone_config"] = normalize_runtime(config.get("systemone_config", {}))
    return command, config


def compare_manifests(manifests):
    errors = []
    reference = manifests[0]
    for manifest in manifests:
        for package in ("vllm", "vllm_ascend"):
            evidence, baseline = manifest.get(package, {}), reference.get(package, {})
            if not evidence.get("sha"):
                errors.append(f"Missing {package} Git SHA in manifest")
            elif (evidence["sha"], evidence.get("diff")) != (baseline.get("sha"), baseline.get("diff")):
                errors.append(f"{package} Git SHA or tracked local changes differ")
        if manifest.get("trace_scheduler") != reference.get("trace_scheduler"):
            errors.append("Scheduler tracing differs between strategy runs")
        if manifest_command(manifest) != manifest_command(reference):
            errors.append("Launch arguments differ beyond the sharing strategy")
    return errors


def write_markdown(path, result):
    lines = [
        "# SystemOne APC 调度证据",
        "",
        f"证据完整性通过：`{result['passed']}`。",
        "",
        "`first_cached_tokens` 是首次 APC 命中；`computed_before` 也包含自身先前 chunks，二者不能互换。",
        "`scheduled_new_tokens` 累计调度新增 token（含重算），不包含设备 padding，也不是 profiler 实测 FLOPs。",
        "逐步记录、等待状态和日志行号保存在同名 JSON 的 `timeline`。",
        "",
        "| 父请求 | 问题 | prompt | 首步 | 首次命中 | 调度新增 | 抢占 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in result.get("requests", [])[:100]:
        cells = [
            row[key]
            for key in (
                "parent_request_id",
                "question_id",
                "prompt_tokens",
                "first_step",
                "first_cached_tokens",
                "scheduled_new_tokens",
                "preemptions",
            )
        ]
        lines.append("| " + " | ".join(str(cell).replace("|", "\\|").replace("\n", " ") for cell in cells) + " |")
    lines.extend(["", "表格最多显示前 100 条；JSON 保留全部请求与步骤。", ""])
    if result["errors"]:
        lines += ["证据缺口：", "", *[f"- {error}" for error in result["errors"]], ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--log", type=Path)
    mode.add_argument("--compare", action="store_true")
    parser.add_argument("--results", nargs="+", type=Path, required=True)
    parser.add_argument("--manifests", nargs="+", type=Path, help="One launch manifest per strategy result")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.results]
    if args.compare:
        result = compare_runs(reports)
        if not args.manifests or len(args.manifests) != len(reports):
            result["errors"].append("Supply one --manifests file for each strategy result to verify code baselines")
        else:
            manifests = [json.loads(path.read_text(encoding="utf-8")) for path in args.manifests]
            result["errors"].extend(compare_manifests(manifests))
        result["passed"] = not result["errors"]
    else:
        steps, runtime, errors = load_log(args.log)
        result = join_trace(steps, reports)
        result["errors"].extend(errors)
        result["runtime"] = runtime
        if not any("cache_groups" in item and "hash_block_size" in item for item in runtime):
            result["errors"].append("Missing scheduler runtime cache-group/hash-block evidence")
        result["passed"] = not result["errors"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if not args.compare:
        write_markdown(args.output.with_suffix(".md"), result)
    print(json.dumps({"passed": result["passed"], "errors": result["errors"]}, ensure_ascii=False))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
