# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Paired, salt-isolated SystemOne HTTP latency capture (run against an NPU server)."""

import argparse
import hashlib
import json
import math
import random
import time
import uuid
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

TARGET_QUESTIONS = 20
LATENCY_RATIO_LIMIT = 1.5
PERCENTILES = (50, 95, 99)


def capture(url, payload, timeout):
    request_id = f"shared-prefix-bench-{uuid.uuid4().hex}"
    request = Request(
        url,
        data=json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "X-Request-Id": request_id},
        method="POST",
    )
    started = time.perf_counter()
    record = {"request_id": request_id, "ok": False}
    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read()
            record["http_status"] = response.status
        # Stop at the full HTTP body; JSON parsing is outside the latency timer.
        record["elapsed_ms"] = (time.perf_counter() - started) * 1000
        result = json.loads(body)
        if not isinstance(result, dict):
            raise ValueError("Response must be a JSON object")
        record["response"] = result
        answers = result.get("answers", {})
        if not isinstance(answers, dict) or set(answers) != set(payload["questions"]):
            raise ValueError("Response does not contain exactly the requested question IDs")
        record["ok"] = True
    except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
        record.setdefault("elapsed_ms", (time.perf_counter() - started) * 1000)
        record["error"] = str(exc)
        if isinstance(exc, HTTPError):
            record["http_status"] = exc.code
    return record


def percentile(values, percent):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100
    low = math.floor(position)
    high = math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def summarize(pairs):
    groups = {}
    for cache in sorted({pair["cache"] for pair in pairs}):
        selected = [pair for pair in pairs if pair["cache"] == cache]
        ratios = [pair["ratio"] for pair in selected if pair["ratio"] is not None]
        groups[cache] = {
            "pairs": len(selected),
            "passed_pairs": sum(pair["passed"] for pair in selected),
            "failed_or_over_limit_pairs": sum(not pair["passed"] for pair in selected),
            "http_failed_pairs": sum(pair["ratio"] is None for pair in selected),
            "all_pairs_passed": all(pair["passed"] for pair in selected),
            "ratio_percentiles_successful_pairs": {f"p{p}": percentile(ratios, p) for p in PERCENTILES},
            "latency_ms_successful_requests": {
                str(count): {
                    f"p{p}": percentile(
                        [pair[str(count)]["elapsed_ms"] for pair in selected if pair[str(count)]["ok"]], p
                    )
                    for p in PERCENTILES
                }
                for count in (1, TARGET_QUESTIONS)
            },
            "failed_repeats": [pair["repeat"] for pair in selected if not pair["passed"]],
        }
    return groups


def run_pairs(payload, single_question, repeats, seed, url, timeout, include_warm, capture_request=capture):
    """Each cold measurement owns a new salt; only its warm repeat reuses it."""
    rng = random.Random(seed)
    pairs = []
    for repeat in range(repeats):
        order = [1, TARGET_QUESTIONS]
        rng.shuffle(order)
        cold = {"repeat": repeat, "cache": "logical_cold", "order": order}
        warm = {"repeat": repeat, "cache": "warm_repeat", "order": order}
        for count in order:
            questions = (
                payload["questions"]
                if count == TARGET_QUESTIONS
                else {single_question: payload["questions"][single_question]}
            )
            measured = {**payload, "questions": questions, "cache_salt": uuid.uuid4().hex}
            cold[str(count)] = capture_request(url, measured, timeout)
            if include_warm:
                warm[str(count)] = capture_request(url, measured, timeout)
                # A failed cold request did not establish a valid warm baseline.
                if not cold[str(count)]["ok"]:
                    warm[str(count)]["ok"] = False
                    warm[str(count)]["error"] = "Cold predecessor failed; warm state is unverified"
        for pair in [cold, warm] if include_warm else [cold]:
            one, many = pair["1"], pair[str(TARGET_QUESTIONS)]
            pair["ratio"] = many["elapsed_ms"] / one["elapsed_ms"] if one["ok"] and many["ok"] else None
            pair["passed"] = pair["ratio"] is not None and pair["ratio"] <= LATENCY_RATIO_LIMIT
            pairs.append(pair)
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Original SystemOne JSON with exactly 20 questions")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8010/v1/systemone")
    parser.add_argument("--single-question", required=True, help="Preselect a representative question ID from the 20")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--include-warm", action="store_true")
    args = parser.parse_args()
    if args.repeats <= 0 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("repeats and timeout must be positive and finite")
    source = args.input.read_bytes()
    payload = json.loads(source)
    questions = payload.get("questions", {})
    if len(questions) != TARGET_QUESTIONS or args.single_question not in questions:
        parser.error("input must have exactly 20 questions including --single-question")
    pairs = run_pairs(payload, args.single_question, args.repeats, args.seed, args.url, args.timeout, args.include_warm)
    report = {
        "input_path": str(args.input.resolve()),
        "input_sha256": hashlib.sha256(source).hexdigest(),
        "single_question": args.single_question,
        "question_ids": list(questions),
        "url": args.url,
        "seed": args.seed,
        "ratio_limit": LATENCY_RATIO_LIMIT,
        "cache_isolation": "New salt per cold request; APC must remain enabled",
        "scope": "HTTP latency only; verify numerical correctness and device reuse separately with server traces",
        "summary": summarize(pairs),
        "pairs": pairs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    return 0 if all(pair["passed"] for pair in pairs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
