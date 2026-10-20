#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
set -euo pipefail

# Run from the checkout, in the same activated environment as the existing server.
stage=${1:?Usage: bash benchmarks/scripts/run_systemone_apc_round.sh diagnostic|performance direct|producer|first_question [output-root] [input-dir]}
strategy=${2:?A strategy is required}
output_root=${3:-benchmarks/results/systemone_apc}
input_dir=${4:-/opt/p00906181/260929_plumb/benchmark_inputs}
case "$stage" in diagnostic|performance) ;; *) echo "Unknown stage: $stage" >&2; exit 2 ;; esac
case "$strategy" in direct|producer|first_question) ;; *) echo "Unknown strategy: $strategy" >&2; exit 2 ;; esac
run_dir="$output_root/$strategy-$stage"
mkdir -p "$run_dir"

for length in 16k 32k; do
    input="$input_dir/decision_20_questions_$length.json"
    common=(--input "$input" --expected-strategy "$strategy" --single-question department
            --url http://127.0.0.1:8181/v1/systemone --timeout 300 --seed 42
            --atol 0.00001 --rtol 0.0001)
    if [[ "$stage" == diagnostic ]]; then
        python benchmarks/scripts/benchmark_systemone_apc.py "${common[@]}" \
            --phase salt --repeats 3 --output "$run_dir/$length-salt.json"
        python benchmarks/scripts/benchmark_systemone_apc.py "${common[@]}" \
            --phase bridge --question-counts 3 20 --concurrency 1 3 --repeats 3 \
            --output "$run_dir/$length-bridge.json"
        python benchmarks/scripts/benchmark_systemone_apc.py "${common[@]}" \
            --phase correctness --repeats 1 --output "$run_dir/$length-correctness.json"
        python benchmarks/scripts/analyze_systemone_apc.py --log "$run_dir/server.log" \
            --results "$run_dir/$length-salt.json" "$run_dir/$length-bridge.json" \
                      "$run_dir/$length-correctness.json" \
            --output "$run_dir/$length-trace.json"
    else
        # Check salt again after restarting without per-step logging.
        python benchmarks/scripts/benchmark_systemone_apc.py "${common[@]}" \
            --phase salt --repeats 1 --output "$run_dir/$length-salt.json"
        result=0
        python benchmarks/scripts/benchmark_systemone_apc.py "${common[@]}" \
            --phase performance --repeats 30 --output "$run_dir/$length-performance.json" || result=$?
        if [[ "$result" -gt 1 ]]; then
            exit "$result"
        fi
        # Exit 1 means valid data but T20/T1 exceeded 1.5; keep the other length.
        if [[ "$result" == 1 ]]; then
            echo "$length: valid measurements saved; latency ratio target not met."
        fi
    fi
done
