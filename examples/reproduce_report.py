"""Replay one case of a saved TimeGym report and print the comparison.

    python examples/reproduce_report.py REPORT [--case-index N] [--output PATH]
        [--require-same-environment]

Replay rebuilds allowlisted built-in models only; a report never imports code.
Exit status 0 means the case was reproduced (with --require-same-environment,
also on the same source, dependencies and Python version), 1 means a comparison
differs, and 2 means the report or arguments are invalid.
"""

import argparse
import json
from pathlib import Path

from timegym import TestReport


def main(argv=None):
    parser = argparse.ArgumentParser(description="Replay one case of a saved TimeGym report.")
    parser.add_argument("report", type=Path, help="saved TimeGym JSON report")
    parser.add_argument("--case-index", type=int, default=0)
    parser.add_argument("--output", type=Path, help="also save the regenerated one-case report")
    parser.add_argument("--require-same-environment", action="store_true",
                        help="also require matching source, dependencies and Python version")
    args = parser.parse_args(argv)
    try:
        replayed = TestReport.read(args.report).replay_case(args.case_index)
        if args.output is not None:
            replayed.to_json(args.output)
    except (OSError, ValueError, TypeError, ImportError, OverflowError) as exc:
        parser.exit(2, f"Replay failed: {exc}\n")
    comparison = replayed.metadata["replay"]
    print(json.dumps(comparison, indent=2, sort_keys=True))
    environment_ok = comparison["same_environment"] or not args.require_same_environment
    return 0 if comparison["reproduced"] and environment_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
