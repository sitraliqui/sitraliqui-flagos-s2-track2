"""Read-only outer validator for the unchanged official 4K/16K CSV protocol.

API: validate_benchmark(raw_csv, summary_csv, official_script, cases=("4k",)).
Raises ValidationError on any missing/failed/inconsistent round, including warmup.
Only stdlib; does not launch processes, import GPU frameworks or modify inputs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import tempfile
from pathlib import Path

OFFICIAL_SCRIPT_SHA256 = "0813eba574e9d1eb50c297c515baa426da1bcb61ce8c78a398b482f246005565"
CASES = {"4k": (4096, 1024, 64, 256), "16k": (16384, 1024, 64, 128)}
SHAPE_COLUMNS = ["Prefill", "Decode", "Conc", "Num Prompts"]
METRICS = [
    "Benchmark Duration (s)", "Total Input Tokens", "Total Output Tokens",
    "Req/s", "Output tok/s", "Peak Output tok/s", "Total tok/s",
    "Mean TTFT (ms)", "Median TTFT (ms)", "P99 TTFT (ms)",
    "Mean TPOT (ms)", "Median TPOT (ms)", "P99 TPOT (ms)",
    "Mean ITL (ms)", "Median ITL (ms)", "P99 ITL (ms)",
]
RAW_COLUMNS = SHAPE_COLUMNS + ["Successful Requests", "Run Status"] + METRICS
SUMMARY_COLUMNS = SHAPE_COLUMNS + METRICS


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path, columns):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == columns, f"{path}: unexpected/duplicate/missing CSV columns")
        rows = list(reader)
    for index, row in enumerate(rows, 1):
        require(set(row) == set(columns), f"{path}: row {index} extra/missing fields")
        require(all(isinstance(row[c], str) and row[c].strip() for c in columns),
                f"{path}: row {index} contains an empty/truncated cell")
    return rows


def number(value, label):
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{label}: not numeric") from exc
    require(math.isfinite(result), f"{label}: nonfinite value")
    return result


def integer(value, label):
    result = number(value, label)
    require(result.is_integer(), f"{label}: not an integer")
    return int(result)


def shape(row):
    return tuple(integer(row[c], c) for c in SHAPE_COLUMNS)


def rate_consistent(rate, count, duration, label):
    # Official stdout prints both rates and duration at 2 decimal places.
    # Compare rounding intervals rather than inventing an arbitrary percent gate.
    require(duration > .005, f"{label}: invalid duration")
    low = count / (duration + .005)
    high = count / (duration - .005)
    require(rate + .005 + 1e-9 >= low and rate - .005 - 1e-9 <= high,
            f"{label}: rate inconsistent with exact count and rounded duration")


def validate_benchmark(raw_csv, summary_csv, official_script, cases=("4k",)):
    """Validate one configuration's complete case(s); never compare two configs.

    Each requested case must occur in the given order, with exactly four raw
    rows and one summary row. Summary must equal the official mean of runs 2-4.
    The caller separately verifies server/variant identity, request logs, process
    exit, no overlap, accuracy prerequisite, and owned cleanup/resource gates.
    """
    cases = tuple(cases)
    require(bool(cases) and len(cases) == len(set(cases)) and all(c in CASES for c in cases),
            "cases must be unique names from 4k/16k")
    require(digest(official_script) == OFFICIAL_SCRIPT_SHA256,
            "Official benchmark script SHA changed; do not modify it to satisfy this validator")
    raw = read_rows(raw_csv, RAW_COLUMNS)
    summaries = read_rows(summary_csv, SUMMARY_COLUMNS)
    require(len(raw) == 4 * len(cases), "Exactly four raw rows per requested case required, including first round")
    require(len(summaries) == len(cases), "Exactly one summary row per requested case required")
    require([shape(row) for row in raw] == [CASES[c] for c in cases for _ in range(4)],
            "Raw case order/shape/request count differs from frozen cases")
    require([shape(row) for row in summaries] == [CASES[c] for c in cases],
            "Summary case order/shape/request count differs from raw cases")
    checked = []
    for case_index, name in enumerate(cases):
        expected = CASES[name]
        input_len, output_len, _, count = expected
        token_in, token_out = input_len * count, output_len * count
        runs = []
        for round_index, row in enumerate(raw[case_index * 4:case_index * 4 + 4], 1):
            prefix = f"{name} round {round_index}"
            require(row["Run Status"] == "SUCCESS", prefix + ": non-SUCCESS row, including first round")
            require(integer(row["Successful Requests"], prefix) == count, prefix + ": incomplete requests")
            metrics = {column: number(row[column], prefix + " " + column) for column in METRICS}
            require(integer(row["Total Input Tokens"], prefix) == token_in, prefix + ": incorrect input token total")
            require(integer(row["Total Output Tokens"], prefix) == token_out, prefix + ": incorrect output token total")
            require(all(v >= 0 for v in metrics.values()), prefix + ": negative metric")
            require(metrics["Benchmark Duration (s)"] > 0 and metrics["Output tok/s"] > 0
                    and metrics["Total tok/s"] > 0 and metrics["Peak Output tok/s"] > 0,
                    prefix + ": zero duration or token throughput")
            for kind in ("TTFT", "TPOT", "ITL"):
                require(metrics[f"P99 {kind} (ms)"] + .01 >= metrics[f"Median {kind} (ms)"],
                        prefix + f": P99 {kind} below median beyond rounding")
            duration = metrics["Benchmark Duration (s)"]
            for column, quantity in (("Req/s", count), ("Output tok/s", token_out),
                                     ("Total tok/s", token_in + token_out)):
                rate_consistent(metrics[column], quantity, duration, prefix + " " + column)
            runs.append(metrics)
        summary = {column: number(summaries[case_index][column], name + " summary " + column)
                   for column in METRICS}
        # This is exactly the unchanged script's statistics.mean + round(., 2).
        expected_summary = {column: round(statistics.mean(run[column] for run in runs[1:]), 2)
                            for column in METRICS}
        for column in METRICS:
            require(abs(summary[column] - expected_summary[column]) <= 1e-8,
                    f"{name}: summary {column} differs from mean of retained rounds 2-4")
        checked.append(dict(case=name, shape=list(expected), raw_rounds=4,
                            all_four_successful=True, successful_requests_per_round=count,
                            input_tokens_per_round=token_in, output_tokens_per_round=token_out,
                            retained_rounds=[2, 3, 4], summary=summary,
                            benchmark_duration_sum_seconds=sum(x["Benchmark Duration (s)"] for x in runs),
                            raw_metrics=runs))
    return dict(status="csv_protocol_passed", cases=checked,
                official_script_SHA256=OFFICIAL_SCRIPT_SHA256,
                raw_csv=dict(path=str(Path(raw_csv).resolve()), sha256=digest(raw_csv)),
                summary_csv=dict(path=str(Path(summary_csv).resolve()), sha256=digest(summary_csv)),
                limitations=["CSV protocol validation, not independent per-request trace validation",
                             "Caller must verify variant/source, accuracy, no-overlap, process exits and cleanup",
                             "P99 summary is mean of per-round P99, not pooled-request P99",
                             "A subset is not the complete official 4K+16K benchmark suite"])


def self_test(raw_csv, summary_csv, official_script):
    """Real historical positive plus counterexamples, all in temporary files."""
    historical = validate_benchmark(raw_csv, summary_csv, official_script, ("4k", "16k"))
    original_raw = read_rows(raw_csv, RAW_COLUMNS)
    original_summary = read_rows(summary_csv, SUMMARY_COLUMNS)
    rejected = []
    with tempfile.TemporaryDirectory(prefix="tianshu-csv-validator-") as tmp:
        root = Path(tmp)
        def write(path, rows, columns):
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=columns)
                writer.writeheader(); writer.writerows(rows)
        def reject(name, mutate_raw=None, mutate_summary=None, script=None):
            raw_rows = [dict(row) for row in original_raw]
            summary_rows = [dict(row) for row in original_summary]
            if mutate_raw: mutate_raw(raw_rows)
            if mutate_summary: mutate_summary(summary_rows)
            rp, sp = root / "raw.csv", root / "summary.csv"
            write(rp, raw_rows, RAW_COLUMNS); write(sp, summary_rows, SUMMARY_COLUMNS)
            try:
                validate_benchmark(rp, sp, script or official_script, ("4k", "16k"))
            except ValidationError:
                rejected.append(name); return
            raise AssertionError("Counterexample accepted: " + name)
        reject("first_round_failure", lambda rows: rows[0].update({"Run Status": "FAILED"}))
        reject("missing_round", lambda rows: rows.pop())
        reject("extra_round", lambda rows: rows.append(dict(rows[-1])))
        reject("incomplete_success_count", lambda rows: rows[0].update({"Successful Requests": "255"}))
        reject("wrong_shape", lambda rows: rows[0].update({"Conc": "32"}))
        reject("wrong_input_tokens", lambda rows: rows[0].update({"Total Input Tokens": "1048575"}))
        reject("short_output", lambda rows: rows[4].update({"Total Output Tokens": "131071"}))
        reject("nonfinite_TPOT", lambda rows: rows[1].update({"Mean TPOT (ms)": "nan"}))
        reject("missing_TTFT", lambda rows: rows[2].update({"P99 TTFT (ms)": ""}))
        reject("inconsistent_throughput", lambda rows: rows[0].update({"Output tok/s": "5000"}))
        reject("wrong_retained_mean", mutate_summary=lambda rows: rows[0].update({"Mean TPOT (ms)": "1"}))
        reject("wrong_summary_case", mutate_summary=lambda rows: rows[1].update({"Prefill": "4096"}))
        altered = root / "altered.py"
        altered.write_bytes(Path(official_script).read_bytes() + b"\n# altered\n")
        reject("changed_official_script", script=altered)
    return dict(status="self_test_passed", historical_cases=len(historical["cases"]),
                historical_raw_rounds=8, rejected_counterexamples=rejected,
                GPU_executed=False, torch_imported=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--official-script", type=Path, required=True)
    parser.add_argument("--cases", choices=tuple(CASES), nargs="+", default=["4k"])
    parser.add_argument("--self-test", action="store_true",
                        help="Use the historical two-case CSV as positive and test strict rejection cases")
    args = parser.parse_args()
    try:
        result = (self_test(args.raw, args.summary, args.official_script) if args.self_test else
                  validate_benchmark(args.raw, args.summary, args.official_script, args.cases))
    except (ValidationError, OSError, csv.Error) as exc:
        print(json.dumps(dict(status="failed", error=type(exc).__name__ + ": " + str(exc))))
        return 2
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
