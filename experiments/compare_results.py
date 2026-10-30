"""Compare a fresh numerical record with a frozen manifest's SHA256."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform

from .common import sha256


def compare(results: Path, manifest: Path) -> dict:
    expected = json.loads(manifest.read_text(encoding="utf-8"))["results_sha256"]
    actual = sha256(results)
    return {"expected_sha256": expected, "actual_sha256": actual, "matches": actual == expected,
            "results": str(results), "manifest": str(manifest), "platform": platform.platform()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--warn", action="store_true", help="Record platform differences without failing")
    args = parser.parse_args()
    result = compare(args.results, args.manifest)
    text = json.dumps(result, indent=2, sort_keys=True)+"\n"
    if args.report:
        args.report.write_text(text, encoding="utf-8")
    print(text, end="")
    if not result["matches"]:
        if args.warn:
            print("::warning::Numerical SHA256 differs from the frozen record; inspect the saved comparison report")
        else:
            raise SystemExit("Numerical SHA256 differs from the frozen record")


if __name__ == "__main__":
    main()
