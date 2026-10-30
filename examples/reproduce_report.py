"""Safely replay a built-in model case: python examples/reproduce_report.py REPORT."""

import argparse
import json

from timegym import TestReport
from timegym.pipeline import replay_case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", help="Saved TimeGym JSON report")
    parser.add_argument("--case-index", type=int, default=0)
    parser.add_argument("--output", help="Optional path for the regenerated one-case report")
    args = parser.parse_args()
    try:
        payload = TestReport.read(args.report)
        report = replay_case(payload.to_dict(), args.case_index)
        if args.output:
            report.to_json(args.output)
    except (OSError, ValueError, TypeError) as exc:
        parser.exit(2, f"Replay failed: {exc}\n")
    comparison = report.metadata["replay"]
    print(json.dumps(comparison, indent=2, sort_keys=True))
    return 0 if all(comparison[key] for key in ("metrics_match", "diagnostics_match", "manifest_match")) else 1


if __name__ == "__main__":
    raise SystemExit(main())

