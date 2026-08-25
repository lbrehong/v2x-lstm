"""
Aggregate every validation type 1 prediction-error CSV in output/validation/
into a single summary table, so results across all RAT x model combinations
can be compared at a glance instead of opening each report individually.

Usage:
    python -m validation.aggregate_reports
"""
import csv
import glob
import os
import re

import config

FILENAME_RE = re.compile(
    r"(?P<rat>5g|pc5|dsrc)_(?P<model>lstm|gru|rnn)_(?P<stage>init|trained)_prediction_errors\.csv")


def main():
    out_dir = os.path.join(config.OUTPUT_DIR, "validation")
    rows = []
    for path in sorted(glob.glob(os.path.join(out_dir, "*_prediction_errors.csv"))):
        match = FILENAME_RE.match(os.path.basename(path))
        if not match:
            continue
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                row.update(match.groupdict())
                rows.append(row)

    if not rows:
        print("No prediction-error reports found in", out_dir)
        return

    fieldnames = ["rat", "model", "stage", "output", "mean_rel_error", "max_rel_error", "p50", "p95", "p99"]
    out_path = os.path.join(out_dir, "summary_all.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    print(f"Aggregated {len(rows)} rows from {len(set((r['rat'], r['model'], r['stage']) for r in rows))} "
          f"reports into {out_path}")


if __name__ == "__main__":
    main()
