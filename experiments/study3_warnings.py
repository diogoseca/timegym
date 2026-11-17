"""Counts of the warnings in the unit records of a finished run of Part 1 or Part 2 (protocol 3.0).

Every unit writes, with each case, how many warnings of each category the case raised (study3_harness). The runner
does not add these numbers to results.json, and part_2_learners.learners.hyperparameter_justification asks that the
ConvergenceWarnings be counted and reported. This module reads the unit records under ``<run>/units`` and writes
``warnings.csv``: one row per learner, subject (the clean pipeline or the mutants), unit phase and stage, with the
number of records, the number that raised a warning, and the number of warnings of each category.

    python -m experiments.study3_warnings <run folder> [--output warnings.csv]

The table is made after the run from the unit records, which are not kept in the repository; the manifest of the
run does not list it.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

FIXED = ("learner", "subject", "phase", "stage", "records", "records_with_warnings")


def tally(units: Path) -> list[dict]:
    """Rows of the table for the unit records under ``units`` (``<learner>/<source or clean>/<phase>-<i>.jsonl``)."""
    counts: dict[tuple, dict] = {}
    for path in sorted(Path(units).glob("*/*/*.jsonl")):
        learner, source = path.parts[-3], path.parts[-2]
        subject = "clean" if source == "clean" else "mutants"
        phase = path.name.rsplit("-", 1)[0]
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if record.get("kind") not in ("case", "warmup"):
                    continue
                stage = record["stage"] if record["kind"] == "case" else "warmup"
                row = counts.setdefault((learner, subject, phase, stage), {"records": 0, "records_with_warnings": 0})
                raised = record.get("warnings") or {}
                row["records"] += 1
                row["records_with_warnings"] += bool(raised)
                for category, number in raised.items():
                    row[category] = row.get(category, 0) + number
    categories = sorted({name for row in counts.values() for name in row} - set(FIXED))
    return [{**dict(zip(FIXED, key)), "records": row["records"],
             "records_with_warnings": row["records_with_warnings"],
             **{category: row.get(category, 0) for category in categories}}
            for key, row in sorted(counts.items())]


def write(rows: list[dict], path: Path) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else list(FIXED), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path, help="The output folder of a finished run; it holds units/")
    parser.add_argument("--output", type=Path, help="Where to write the table (default: <run>/warnings.csv)")
    args = parser.parse_args(argv)
    units = args.run / "units"
    if not units.is_dir():
        raise SystemExit(f"No unit records in {units}")
    rows = tally(units)
    output = args.output or args.run / "warnings.csv"
    write(rows, output)
    print(json.dumps({"output": str(output), "rows": len(rows),
                      "warnings": sum(value for row in rows for name, value in row.items() if name not in FIXED)}))


if __name__ == "__main__":
    main()
