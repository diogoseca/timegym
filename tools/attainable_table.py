"""Print or check the scorecard's stored attainable scores: python tools/attainable_table.py [--check]

For each of the 20 cells of the default scorecard grid (ten patterns at noise sd 0.5 and
2.0, 120 steps of history, 24 ahead), fits the pattern's own model family
(``Pattern.attainable_by``) to 200 fresh series and scores it as a cell is scored. Without
arguments it prints every value with the seconds it took, then the table as the dict
literal that source/13_scorecard.ipynb stores in ``_ATTAINABLE``.

``--check`` recomputes the 20 values and exits with status 1 when any differs from the
stored one by more than 0.3 points. The four maximum-likelihood fits stop at slightly
different points on other SciPy versions and platforms, so the values are not expected to
repeat to the last decimal.
"""
import argparse
import sys
import time

from timegym.grid import PATTERNS, scenario_grid
from timegym.scorecard import _ATTAINABLE, _attainable_score

TOLERANCE = 0.3


def compute():
    """Yield ``(pattern, noise_sd, attainable score, seconds)`` for every default cell."""
    for cell in scenario_grid():
        started = time.perf_counter()
        value = _attainable_score(PATTERNS[cell.pattern], cell.scenario)
        yield cell.pattern, cell.noise_sd, value, time.perf_counter() - started


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true",
                        help=f"fail when a stored value is more than {TOLERANCE} points off")
    check = parser.parse_args(argv).check
    table, failures, total, count = {}, [], 0.0, 0
    for key, sd, value, seconds in compute():
        total, count = total + seconds, count + 1
        table.setdefault(key, {})[sd] = round(value, 1)
        stored = _ATTAINABLE.get(key, {}).get(sd)
        line = f"{key:18s} sd {sd:<4g} {value:6.2f}  {seconds:5.1f} s"
        if check:
            off = float("inf") if stored is None else abs(value - stored)
            line += f"  stored {stored}  difference {off:.2f}"
            if off > TOLERANCE:
                failures.append(f"{key} at sd {sd:g}: computed {value:.2f}, stored {stored}")
        print(line, flush=True)
    print(f"{count} cells in {total:.1f} s")
    if check:
        if failures:
            print(f"More than {TOLERANCE} points from the stored table:\n  " + "\n  ".join(failures))
            return 1
        print(f"Every stored attainable score is within {TOLERANCE} points")
        return 0
    print("_ATTAINABLE = {")
    for key, values in table.items():
        print(f"    {key!r}: {{" + ", ".join(f"{sd!r}: {value!r}" for sd, value in values.items()) + "},")
    print("}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
