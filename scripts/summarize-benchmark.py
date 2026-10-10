"""Validate the untouched official CSV and summarize retained runs 2 through 4."""
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

EXPECTED = {(4096, 1024, 64, 256), (16384, 1024, 64, 128)}
CASE_KEYS = ("Prefill", "Decode", "Conc", "Num Prompts")
METRICS = ("Total tok/s", "Output tok/s", "Mean TTFT (ms)", "P99 TTFT (ms)",
           "Mean TPOT (ms)", "Benchmark Duration (s)")


def summarize(raw_path, summary_path):
    grouped = defaultdict(list)
    with raw_path.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            case = tuple(int(row[k]) for k in CASE_KEYS)
            grouped[case].append(row)
    errors = []
    if set(grouped) != EXPECTED:
        errors.append("Raw CSV does not contain exactly the two official cases")
    official = {}
    if summary_path is not None and summary_path.exists():
        with summary_path.open(newline="", encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                case = tuple(int(row[k]) for k in CASE_KEYS)
                if case in official:
                    errors.append(f"Duplicate summary row: {case}")
                official[case] = row
    if set(official) != EXPECTED:
        errors.append("Official summary does not contain exactly the two official cases")
    cases = []
    for case, rows in sorted(grouped.items()):
        case_errors = []
        if len(rows) != 4:
            case_errors.append(f"Expected 4 ordered runs, found {len(rows)}")
        checked = []
        for index, row in enumerate(rows, 1):
            ok = (row.get("Run Status") == "SUCCESS" and
                  float(row.get("Successful Requests") or -1) == case[3])
            checked.append({"run": index, "successful_requests": row.get("Successful Requests"),
                            "status": row.get("Run Status"), "valid": ok})
            if not ok:
                case_errors.append(f"Run {index} did not complete all expected requests")
        metrics = {}
        retained = rows[1:4]
        for metric in METRICS:
            try:
                values = [float(row[metric]) for row in retained]
                if len(values) != 3 or not all(math.isfinite(v) and v > 0 for v in values):
                    raise ValueError("missing, nonfinite or nonpositive values")
                avg = mean(values)
                metrics[metric] = {"runs_2_to_4": values, "mean": avg,
                                   "sample_stddev": stdev(values), "min": min(values),
                                   "max": max(values), "cv_percent": stdev(values) / avg * 100}
                expected_avg = float(official[case][metric])
                if abs(expected_avg - round(avg, 2)) > 0.010001:
                    case_errors.append(f"Official summary mean disagrees for {metric}")
            except (ValueError, TypeError, KeyError) as exc:
                case_errors.append(f"Cannot validate {metric}: {exc}")
        errors.extend(f"{case}: {message}" for message in case_errors)
        cases.append({"case": list(case), "valid": not case_errors,
                      "requests_by_run": checked, "metrics": metrics})
    return {"protocol": {"runs": 4, "skip_first": 1, "retained_runs": [2, 3, 4],
                         "run_id_basis": "row order within each case in official raw CSV"},
            "raw_file": raw_path.name,
            "official_summary_file": summary_path.name if summary_path else None,
            "valid": not errors, "errors": errors, "cases": cases}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.raw, args.summary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"valid": result["valid"], "errors": result["errors"]}, ensure_ascii=False))
    raise SystemExit(0 if result["valid"] else 1)
