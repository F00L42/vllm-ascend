# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Capture salt isolation, concurrent APC reuse, numerical checks, and paired HTTP latency."""

import argparse
import hashlib
import json
import math
import random
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from http.client import HTTPException, IncompleteRead
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

TARGET_QUESTIONS = 20
BRIDGE_QUESTIONS = ("department", "refund", "urgency")
PERCENTILES = (50, 95, 99)
PROBABILITY_SUM_TOLERANCE = 1e-6


def _finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _reject_json_constant(value):
    raise ValueError(f"Invalid JSON numeric constant: {value}")


def question_results(record):
    """Index full-precision observations by the original JSON question ID."""
    if not record["ok"]:
        raise ValueError(record.get("error", "HTTP capture failed"))
    response = record["response"]
    return {
        detail["question_id"]: {
            "prompt_tokens": detail["prompt_tokens"],
            "probabilities": detail["probabilities"],
            "answer": response["answers"][detail["question_id"]],
        }
        for detail in response["diagnostics"]["question_details"]
    }


def validate_response(record, payload, expected_strategy):
    response = record["response"]
    if not isinstance(response, dict):
        raise ValueError("Response must be a JSON object")
    questions = payload["questions"]
    answers = response.get("answers")
    if not isinstance(answers, dict) or set(answers) != set(questions):
        raise ValueError("Response must contain exactly the requested question IDs")
    diagnostics = response.get("diagnostics")
    if not isinstance(diagnostics, dict) or diagnostics.get("schema_version") != 1:
        raise ValueError("Missing diagnostics schema_version=1; enable server diagnostics")
    if diagnostics.get("parent_request_id") != record["request_id"]:
        raise ValueError("diagnostics.parent_request_id does not match X-Request-Id")
    if diagnostics.get("configured_strategy") != expected_strategy:
        raise ValueError("Server configured_strategy differs from --expected-strategy")
    actual_strategy = "direct" if len(questions) == 1 else expected_strategy
    if diagnostics.get("strategy") != actual_strategy:
        raise ValueError("Actual strategy differs from expected strategy (including fallback)")
    salt_hash = hashlib.sha256(payload["cache_salt"].encode()).hexdigest()
    if diagnostics.get("cache_salt_sha256") != salt_hash:
        raise ValueError("Server cache salt hash does not match submitted cache_salt")
    details = diagnostics.get("question_details")
    if not isinstance(details, list) or not all(isinstance(item, dict) for item in details):
        raise ValueError("Missing per-question diagnostics")
    if len(details) != len(questions) or {item.get("question_id") for item in details} != set(questions):
        raise ValueError("Per-question diagnostics must cover every question exactly once")
    for item in details:
        question_id = item["question_id"]
        question = questions[question_id]
        candidates = 2 if question["type"] == "noul" else len(question["criteria"])
        if question["type"] == "choice" and isinstance(question["criteria"], list):
            candidates = len(set(question["criteria"]))
        values = item.get("probabilities")
        if (
            not isinstance(values, list)
            or len(values) != candidates
            or any(not _finite_number(value) or not 0 <= value <= 1 for value in values)
            or abs(sum(values) - 1) > PROBABILITY_SUM_TOLERANCE
        ):
            raise ValueError(f"{question_id}: incomplete or invalid unrounded probability vector")
        prompt_tokens, cached_tokens = item.get("prompt_tokens"), item.get("cached_tokens")
        if type(prompt_tokens) is not int or prompt_tokens <= 0:
            raise ValueError(f"{question_id}: prompt_tokens must be a positive integer")
        if type(cached_tokens) is not int or not 0 <= cached_tokens <= prompt_tokens:
            raise ValueError(f"{question_id}: cached_tokens must be known and within the prompt length")
        if item.get("status") != "completed" or not item.get("engine_request_id"):
            raise ValueError(f"{question_id}: missing completed engine request identity")
        answer = answers[question_id]
        if not isinstance(answer, dict) or answer.get("type") != question["type"]:
            raise ValueError(f"{question_id}: answer type differs from the requested type")
        if question["type"] == "choice" and answer.get("choice") not in question["criteria"]:
            raise ValueError(f"{question_id}: answer choice is not a candidate")
        for field in ("confidence", "score" if question["type"] == "score" else "noul"):
            if (field == "confidence" or question["type"] in ("score", "noul")) and not _finite_number(
                answer.get(field)
            ):
                raise ValueError(f"{question_id}: missing finite answer {field}")


def capture(url, payload, timeout, expected_strategy):
    """Keep the full response, HTTP errors, identities, and both client clock domains."""
    request_id = f"systemone-apc-{uuid.uuid4().hex}"
    request = Request(
        url,
        data=json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "X-Request-Id": request_id},
        method="POST",
    )
    record = {
        "request_id": request_id,
        "question_ids": list(payload["questions"]),
        "cache_salt": payload["cache_salt"],
        "ok": False,
        "submit_wall_ns": time.time_ns(),
        "submit_monotonic_ns": time.perf_counter_ns(),
    }
    body = None
    try:
        with urlopen(request, timeout=timeout) as response:
            record["http_status"] = response.status
            body = response.read()
    except (HTTPError, URLError, TimeoutError, OSError, HTTPException) as exc:
        record["error"] = str(exc)
        if isinstance(exc, IncompleteRead):
            body = exc.partial
        if isinstance(exc, HTTPError):
            record["http_status"] = exc.code
            try:
                body = exc.read()
            except (OSError, ValueError) as read_error:
                record["body_read_error"] = str(read_error)
    finally:
        record["complete_monotonic_ns"] = time.perf_counter_ns()
        record["complete_wall_ns"] = time.time_ns()
        record["elapsed_ms"] = (record["complete_monotonic_ns"] - record["submit_monotonic_ns"]) / 1e6
    if body is not None:
        record["body_text"] = body.decode("utf-8", errors="replace")
        try:
            record["response"] = json.loads(body, parse_constant=_reject_json_constant)
            if "error" not in record:
                validate_response(record, payload, expected_strategy)
                record["ok"] = True
        except (ValueError, KeyError, TypeError) as exc:
            record.setdefault("error", str(exc))
    return record


def _close(reference, observed, atol, rtol):
    return abs(observed - reference) <= atol + rtol * abs(reference)


def compare_questions(reference, observed, atol, rtol, mapping=None):
    """Compare full vectors and answer/score fields; tolerances never grow with observed noise."""
    mapping = mapping or {question_id: question_id for question_id in observed}
    if set(observed) != set(mapping) or any(question_id not in reference for question_id in mapping.values()):
        return {"passed": False, "error": "Question IDs differ from the declared comparison mapping"}
    results = {}
    for observed_id, reference_id in mapping.items():
        left, right = reference[reference_id], observed[observed_id]
        a, b = left["probabilities"], right["probabilities"]
        same_length = len(a) == len(b)
        deltas = [abs(y - x) for x, y in zip(a, b, strict=False)]
        raw_passed = same_length and all(_close(x, y, atol, rtol) for x, y in zip(a, b, strict=False))
        left_answer, right_answer = left["answer"], right["answer"]
        numeric_deltas = {}
        numeric_passed = True
        for field in ("confidence", "score", "noul"):
            if field in left_answer or field in right_answer:
                x, y = left_answer.get(field), right_answer.get(field)
                valid = _finite_number(x) and _finite_number(y)
                numeric_deltas[field] = y - x if valid else None
                numeric_passed = numeric_passed and valid and _close(x, y, atol, rtol)
        choice_changed = left_answer.get("choice") != right_answer.get("choice")
        prompt_equal = left["prompt_tokens"] == right["prompt_tokens"]
        results[observed_id] = {
            "reference_question_id": reference_id,
            "max_abs": max(deltas) if deltas else None,
            "probabilities_passed": raw_passed,
            "prompt_tokens_equal": prompt_equal,
            "choice_changed": choice_changed,
            "answer_numeric_deltas": numeric_deltas,
            "passed": raw_passed
            and prompt_equal
            and numeric_passed
            and not choice_changed
            and left_answer.get("type") == right_answer.get("type"),
        }
    return {"passed": bool(results) and all(result["passed"] for result in results.values()), "questions": results}


def compare_records(reference, observed, atol, rtol):
    if not reference["ok"] or not observed["ok"]:
        return {"passed": False, "error": "One or both HTTP captures failed"}
    left, right = question_results(reference), question_results(observed)
    if set(left) != set(right):
        return {"passed": False, "error": "Question IDs differ"}
    return compare_questions(left, right, atol, rtol)


def _against_baseline(baseline, record, atol, rtol, mapping=None):
    if not record["ok"]:
        return {"passed": False, "error": record.get("error", "HTTP capture failed")}
    return compare_questions(baseline, question_results(record), atol, rtol, mapping)


def _payload(payload, question_ids, salt=None):
    return {
        **payload,
        "questions": {question_id: payload["questions"][question_id] for question_id in question_ids},
        "cache_salt": salt or uuid.uuid4().hex,
    }


def run_salt(payload, args, capture_request):
    trials = []
    for repeat in range(args.repeats):
        salt_a, salt_b = uuid.uuid4().hex, uuid.uuid4().hex
        calls = [capture_request(_payload(payload, [args.single_question], salt)) for salt in (salt_a, salt_a, salt_b)]
        cached = [
            call["response"]["diagnostics"]["question_details"][0]["cached_tokens"] if call["ok"] else None
            for call in calls
        ]
        comparisons = [compare_records(calls[0], call, args.atol, args.rtol) for call in calls[1:]]
        passed = cached[0] == 0 and cached[1] is not None and cached[1] > 0 and cached[2] == 0
        trials.append(
            {
                "repeat": repeat,
                "calls": calls,
                "cached_tokens_A_A_B": cached,
                "comparisons": comparisons,
                "passed": passed and all(comparison["passed"] for comparison in comparisons),
            }
        )
    return {"passed": all(trial["passed"] for trial in trials), "trials": trials}


def _concurrent(payload, concurrency, capture_request):
    barrier = threading.Barrier(concurrency)

    def submit(_):
        barrier.wait()
        return capture_request(payload)

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        return list(executor.map(submit, range(concurrency)))


def run_bridge(payload, args, capture_request):
    trials = []
    for count in args.question_counts:
        question_ids = (
            args.bridge_questions if count == len(args.bridge_questions) else list(payload["questions"])[:count]
        )
        for concurrency in args.concurrency:
            for repeat in range(args.repeats):
                measured = _payload(payload, question_ids)
                cold = _concurrent(measured, concurrency, capture_request)
                warm = _concurrent(measured, concurrency, capture_request)
                comparisons = [compare_records(cold[0], call, args.atol, args.rtol) for call in cold[1:] + warm]
                warm_cache_verified = all(
                    call["ok"]
                    and all(
                        detail["cached_tokens"] > 0 for detail in call["response"]["diagnostics"]["question_details"]
                    )
                    for call in warm
                )
                trials.append(
                    {
                        "repeat": repeat,
                        "question_count": count,
                        "question_ids": question_ids,
                        "concurrency": concurrency,
                        "cache_salt": measured["cache_salt"],
                        "cold": cold,
                        "warm": warm,
                        "comparisons": comparisons,
                        "warm_cache_verified": warm_cache_verified,
                        "passed": all(call["ok"] for call in cold + warm)
                        and all(comparison["passed"] for comparison in comparisons)
                        and warm_cache_verified,
                    }
                )
    return {"passed": all(trial["passed"] for trial in trials), "trials": trials}


def run_correctness(payload, args, capture_request):
    trials = []
    duplicate_id = f"__apc_duplicate_{args.single_question}"
    if duplicate_id in payload["questions"]:
        raise ValueError(f"Reserved duplicate question ID already exists: {duplicate_id}")
    for repeat in range(args.repeats):
        independent = []
        baseline = {}
        for question_id in payload["questions"]:
            first = capture_request(_payload(payload, [question_id]))
            second = capture_request(_payload(payload, [question_id]))
            comparison = compare_records(first, second, args.atol, args.rtol)
            cold_verified = all(
                call["ok"] and call["response"]["diagnostics"]["question_details"][0]["cached_tokens"] == 0
                for call in (first, second)
            )
            independent.append(
                {
                    "question_id": question_id,
                    "first": first,
                    "second": second,
                    "comparison": comparison,
                    "cold_verified": cold_verified,
                }
            )
            if first["ok"]:
                baseline.update(question_results(first))
        batch = capture_request(_payload(payload, payload["questions"]))
        reversed_batch = capture_request(_payload(payload, reversed(payload["questions"])))
        duplicate_payload = _payload(payload, payload["questions"])
        duplicate_payload["questions"][duplicate_id] = payload["questions"][args.single_question]
        duplicated = capture_request(duplicate_payload)
        mapping = {question_id: question_id for question_id in payload["questions"]}
        mapping[duplicate_id] = args.single_question
        comparisons = {
            "batch": _against_baseline(baseline, batch, args.atol, args.rtol),
            "reversed": _against_baseline(baseline, reversed_batch, args.atol, args.rtol),
            "duplicated": _against_baseline(baseline, duplicated, args.atol, args.rtol, mapping),
        }
        trials.append(
            {
                "repeat": repeat,
                "independent": independent,
                "batch": batch,
                "reversed": reversed_batch,
                "duplicated": duplicated,
                "comparisons": comparisons,
                "passed": all(item["comparison"]["passed"] and item["cold_verified"] for item in independent)
                and all(comparison["passed"] for comparison in comparisons.values()),
            }
        )
    return {"passed": all(trial["passed"] for trial in trials), "trials": trials}


def percentile(values, percent):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def summarize_pairs(pairs):
    summary = {}
    for cache in ("logical_cold", "warm_repeat"):
        selected = [pair for pair in pairs if pair["cache"] == cache]
        ratios = [pair["ratio"] for pair in selected if pair["ratio"] is not None]
        summary[cache] = {
            "pairs": len(selected),
            "passed_pairs": sum(pair["passed"] for pair in selected),
            "failed_pairs": sum(not pair["passed"] for pair in selected),
            "data_failed_pairs": sum(not pair["data_passed"] for pair in selected),
            "target_failed_valid_pairs": sum(pair["data_passed"] and not pair["target_passed"] for pair in selected),
            "failed_repeats": [pair["repeat"] for pair in selected if not pair["passed"]],
            "ratio_percentiles_valid_pairs": {f"p{p}": percentile(ratios, p) for p in PERCENTILES},
            "latency_ms_successful_requests": {
                label: {
                    f"p{p}": percentile([pair[label]["elapsed_ms"] for pair in selected if pair[label]["ok"]], p)
                    for p in PERCENTILES
                }
                for label in ("one", "many")
            },
        }
    return summary


def run_performance(payload, args, capture_request):
    baseline_calls = [capture_request(_payload(payload, [question_id])) for question_id in payload["questions"]]
    baseline = {}
    for call in baseline_calls:
        if call["ok"]:
            baseline.update(question_results(call))
    pairs = []
    rng = random.Random(args.seed)
    for repeat in range(args.repeats):
        order = ["one", "many"]
        rng.shuffle(order)
        cold = {"repeat": repeat, "cache": "logical_cold", "order": order}
        warm = {"repeat": repeat, "cache": "warm_repeat", "order": order}
        for label in order:
            question_ids = [args.single_question] if label == "one" else list(payload["questions"])
            measured = _payload(payload, question_ids)
            cold[label] = capture_request(measured)
            warm[label] = capture_request(measured)
        for pair in (cold, warm):
            comparisons = {
                label: _against_baseline(baseline, pair[label], args.atol, args.rtol) for label in ("one", "many")
            }
            predecessor_passed = pair is cold or cold["data_passed"]
            valid = predecessor_passed and all(comparison["passed"] for comparison in comparisons.values())
            cache_verified = all(
                pair[label]["ok"]
                and (
                    all(
                        detail["cached_tokens"] > 0
                        for detail in pair[label]["response"]["diagnostics"]["question_details"]
                    )
                    if pair is warm
                    else label == "many"
                    or pair[label]["response"]["diagnostics"]["question_details"][0]["cached_tokens"] == 0
                )
                for label in ("one", "many")
            )
            valid = valid and cache_verified
            ratio = pair["many"]["elapsed_ms"] / pair["one"]["elapsed_ms"] if valid else None
            pair.update(
                comparisons=comparisons,
                cold_predecessor_passed=predecessor_passed,
                cache_verified=cache_verified,
                data_passed=valid,
                ratio=ratio,
                target_passed=ratio is not None and ratio <= args.ratio_limit,
                passed=ratio is not None and ratio <= args.ratio_limit,
            )
            pairs.append(pair)
    baseline_passed = all(
        call["ok"] and call["response"]["diagnostics"]["question_details"][0]["cached_tokens"] == 0
        for call in baseline_calls
    )
    data_passed = baseline_passed and all(pair["data_passed"] for pair in pairs)
    target_passed = all(pair["target_passed"] for pair in pairs)
    return {
        "passed": data_passed and target_passed,
        "data_passed": data_passed,
        "target_passed": target_passed,
        "baseline_passed": baseline_passed,
        "baseline": baseline_calls,
        "pairs": pairs,
        "summary": summarize_pairs(pairs),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, required=True, help="Existing SystemOne request JSON; never synthesize questions"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8181/v1/systemone")
    parser.add_argument("--expected-strategy", choices=("direct", "producer", "first_question"), required=True)
    parser.add_argument("--phase", choices=("salt", "bridge", "correctness", "performance", "all"), default="all")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--single-question", default="department")
    parser.add_argument("--bridge-questions", nargs="+", default=list(BRIDGE_QUESTIONS))
    parser.add_argument("--question-counts", nargs="+", type=int, default=[3, TARGET_QUESTIONS])
    parser.add_argument("--concurrency", nargs="+", type=int, default=[1, 3])
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--ratio-limit", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args(argv)
    if args.repeats <= 0 or any(count <= 0 for count in args.question_counts + args.concurrency):
        parser.error("repeats, question counts, and concurrency must be positive")
    if any(not math.isfinite(value) or value < 0 for value in (args.atol, args.rtol)):
        parser.error("atol and rtol must be finite and nonnegative; choose them before the run")
    if any(not math.isfinite(value) or value <= 0 for value in (args.timeout, args.ratio_limit)):
        parser.error("timeout and ratio-limit must be finite and positive")
    source = args.input.read_bytes()
    payload = json.loads(source)
    questions = payload.get("questions", {})
    if not isinstance(questions, dict) or not questions or args.single_question not in questions:
        parser.error("input must contain questions including --single-question")
    if args.phase in ("all", "performance") and len(questions) != TARGET_QUESTIONS:
        parser.error("performance requires the original input with exactly 20 questions")
    if args.phase in ("all", "bridge"):
        if len(set(args.bridge_questions)) != len(args.bridge_questions):
            parser.error("--bridge-questions must be distinct")
        if any(question_id not in questions for question_id in args.bridge_questions):
            parser.error("input does not contain the selected --bridge-questions")
        if max(args.question_counts) > len(questions):
            parser.error("input has fewer original questions than --question-counts; no questions will be fabricated")
    return args, source, payload


def main(argv=None):
    args, source, payload = parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    journal_path = args.output.with_suffix(args.output.suffix + ".requests.jsonl")
    parameters = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    report = {
        "schema_version": 1,
        "input_path": str(args.input.resolve()),
        "input_sha256": hashlib.sha256(source).hexdigest(),
        "question_ids": list(payload["questions"]),
        "parameters": parameters,
        "journal_path": str(journal_path.resolve()),
        "scope": "HTTP latency; endpoint submission clocks are not engine scheduler timestamps",
        "started_wall_ns": time.time_ns(),
        "status": "running",
        "phases": {},
        "passed": False,
    }

    def save():
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )

    save()
    journal_lock = threading.Lock()
    phases = ("salt", "bridge", "correctness", "performance") if args.phase == "all" else (args.phase,)
    runners = {"salt": run_salt, "bridge": run_bridge, "correctness": run_correctness, "performance": run_performance}
    with journal_path.open("w", encoding="utf-8") as journal:

        def capture_request(measured):
            record = capture(args.url, measured, args.timeout, args.expected_strategy)
            record["phase"] = report["phase_in_progress"]
            with journal_lock:
                journal.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                journal.flush()
            return record

        try:
            for phase in phases:
                report["phase_in_progress"] = phase
                if any(not result["passed"] for result in report["phases"].values()):
                    report["phases"][phase] = {"passed": False, "skipped": "A preceding acceptance gate failed"}
                else:
                    print(f"Running {phase} ({args.expected_strategy}, repeats={args.repeats})", flush=True)
                    report["phases"][phase] = runners[phase](payload, args, capture_request)
                save()
            report["passed"] = all(result["passed"] for result in report["phases"].values())
            report["data_passed"] = all(
                result.get("data_passed", result["passed"]) for result in report["phases"].values()
            )
            report["target_passed"] = all(result.get("target_passed", True) for result in report["phases"].values())
            report["status"] = "completed"
        except (Exception, KeyboardInterrupt) as exc:
            report["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
            report["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            report["completed_wall_ns"] = time.time_ns()
            save()
    print(json.dumps({"passed": report["passed"], "status": report["status"], "output": str(args.output)}, indent=2))
    return 0 if report["passed"] else 1 if report.get("data_passed", False) else 2


if __name__ == "__main__":
    raise SystemExit(main())
