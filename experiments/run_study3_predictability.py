"""Run the predictability estimators of protocol 3.0, Part 5 (part_5_predictability_estimators).

    python -m experiments.run_study3_predictability --plan
    python -m experiments.run_study3_predictability

For each of the 20 default scorecard cells and the 200 registered series (seeds 13000000 + i, the same
in every cell) the runner records the exact floor (the mean over the horizon of the reference
variance), the spectral bound with the primary inputs (the last 24 context values and the 24 realized
values) and with the history-only inputs (context values 73 to 96 and 97 to 120, 1-based), and the
weighted permutation entropy of the 120 context values. It then writes the table of
part_5_predictability_estimators.table and the two Spearman correlations. The run refuses to start
until study3_common.require_start allows Part 5; --plan prints what would run and computes nothing.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.stats import spearmanr
from threadpoolctl import threadpool_info, threadpool_limits

from timegym.grid import scenario_grid

from . import study3_common as common
from .study3_predictability import spectral_bound, weighted_permutation_entropy

PART = "part-5"
CONTEXT, HORIZON, SEGMENT = 120, 24, 24
# Rows whose process has no stationary spectrum; their bound is reported and labelled.
NONSTATIONARY = ("random_walk", "random_walk_drift")


def _finite(value):
    return None if value is None or not math.isfinite(value) else float(value)


def series_inputs(case):
    """The context (120 values), the realized future (24 values) and the exact floor of a one-target case."""
    context = np.asarray(case.task.history, dtype=float)
    observed = np.asarray(case.observed, dtype=float)
    if context.shape != (CONTEXT, 1) or observed.shape != (HORIZON, 1):
        raise ValueError(f"Expected one target, {CONTEXT} context values and {HORIZON} future values")
    floor = float(np.mean(np.asarray(case.reference.variance, dtype=float)))
    return context[:, 0], observed[:, 0], floor


def series_row(case, bound=spectral_bound, entropy=weighted_permutation_entropy):
    """The exact floor, both bounds, their ratios to the floor and the entropy of one series."""
    context, observed, floor = series_inputs(case)
    primary = bound(context[-SEGMENT:], observed)
    history = bound(context[72:96], context[96:120])
    row = {"floor": floor, "context_variance": float(np.var(context)), "wpe": _finite(entropy(context))}
    for name, found in (("primary", primary), ("history", history)):
        defined = bool(found["defined"]) and floor > 0
        row[f"{name}_bound"] = _finite(found["bound"]) if found["defined"] else None
        row[f"{name}_ratio"] = found["bound"] / floor if defined else None
    return row


def _spread(values):
    values = [value for value in values if value is not None]
    if not values:
        return {"median": None, "p05": None, "p95": None, "defined": 0}
    p05, median, p95 = np.percentile(values, [5, 50, 95])
    return {"median": float(median), "p05": float(p05), "p95": float(p95), "defined": len(values)}


def cell_summary(cell, rows):
    """One row of the table: ratio spreads for both inputs, undefined counts, median WPE and floor summaries."""
    entropies = [row["wpe"] for row in rows if row["wpe"] is not None]
    scale_free = [row["floor"] / row["context_variance"] for row in rows if row["context_variance"] > 0]
    out = {"pattern": cell.pattern, "noise_sd": float(cell.noise_sd), "series": len(rows),
           "label": "no stationary spectrum" if cell.pattern in NONSTATIONARY else "",
           "median_floor": float(np.median([row["floor"] for row in rows])),
           "median_scale_free_floor": float(np.median(scale_free)) if scale_free else None,
           "median_wpe": float(np.median(entropies)) if entropies else None,
           "wpe_undefined": len(rows) - len(entropies)}
    for name in ("primary", "history"):
        spread = _spread([row[f"{name}_ratio"] for row in rows])
        out[f"{name}_ratio"] = {key: spread[key] for key in ("median", "p05", "p95")}
        out[f"{name}_undefined"] = len(rows) - spread["defined"]
    return out


def correlations(summaries):
    """Spearman correlations across rows of median WPE with the median floor and with the scale-free floor."""
    out = {}
    for name, key in (("wpe_vs_floor", "median_floor"), ("wpe_vs_scale_free_floor", "median_scale_free_floor")):
        pairs = [(row["median_wpe"], row[key]) for row in summaries
                 if row["median_wpe"] is not None and row[key] is not None]
        if len(pairs) < 3:
            out[name] = {"rho": None, "p_value": None, "rows": len(pairs)}
            continue
        found = spearmanr([a for a, _ in pairs], [b for _, b in pairs])
        out[name] = {"rho": _finite(float(found.statistic)), "p_value": _finite(float(found.pvalue)),
                     "rows": len(pairs)}
    return out


def run(cells, seeds, bound=spectral_bound, entropy=weighted_permutation_entropy):
    """Per-series rows, the table and the correlations for ``cells`` (GridCell-like) and case ``seeds``."""
    series, summaries = [], []
    for cell in cells:
        rows = []
        for index, seed in enumerate(seeds):
            row = series_row(cell.scenario.generate(seed=seed), bound, entropy)
            rows.append({"pattern": cell.pattern, "noise_sd": float(cell.noise_sd), "series": index,
                         "seed": int(seed), **row})
        series.extend(rows)
        summaries.append(cell_summary(cell, rows))
    return {"series": series, "table": summaries, "correlations": correlations(summaries)}


def registered_inputs(protocol):
    part = protocol["part_5_predictability_estimators"]
    cells = scenario_grid()
    seeds = common.case_seeds(protocol, "predictability_series")
    if len(cells) != 20 or len(seeds) != part["series_per_cell"]:
        raise ValueError("Part 5 needs the 20 default scorecard cells and 200 series per cell")
    for cell in cells:
        scenario = cell.scenario
        if (scenario.history_length, scenario.horizon, scenario.n_targets) != (CONTEXT, HORIZON, 1):
            raise ValueError(f"{cell.pattern}: expected context 120, horizon 24 and one target")
    if part["permutation_entropy"]["word_length"] != 3:
        raise ValueError("The registered word length is 3")
    return cells, seeds


def write_tables(results, output):
    with (output / "table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["pattern", "noise_sd", "label", "series", "primary_median", "primary_p05", "primary_p95",
                         "primary_undefined", "history_median", "history_p05", "history_p95", "history_undefined",
                         "median_wpe", "wpe_undefined", "median_floor", "median_scale_free_floor"])
        for row in results["table"]:
            writer.writerow([row["pattern"], row["noise_sd"], row["label"], row["series"],
                             *[row["primary_ratio"][key] for key in ("median", "p05", "p95")],
                             row["primary_undefined"],
                             *[row["history_ratio"][key] for key in ("median", "p05", "p95")],
                             row["history_undefined"], row["median_wpe"], row["wpe_undefined"],
                             row["median_floor"], row["median_scale_free_floor"]])
    with (output / "series.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results["series"][0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(results["series"])


def plan(protocol):
    cells, seeds = registered_inputs(protocol)
    return {"part": PART, "cells": [f"{cell.pattern} sd {cell.noise_sd}" for cell in cells],
            "series_per_cell": len(seeds), "seeds": [seeds[0], seeds[-1]], "series": len(cells) * len(seeds),
            "estimator_calls": {"spectral_bound": 2 * len(cells) * len(seeds),
                                "weighted_permutation_entropy": len(cells) * len(seeds)},
            "cpu_hours_estimate": protocol["runtime_estimate_cpu_hours"]["part_5"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=common.RESULTS / PART)
    parser.add_argument("--force", action="store_true", help="Replace existing study artifacts")
    parser.add_argument("--plan", action="store_true", help="Print what would run and compute nothing")
    args = parser.parse_args()
    protocol = common.load_protocol()
    if args.plan:
        print(json.dumps(plan(protocol), indent=2))
        return
    common.require_start(protocol, PART)
    common.protect_output(args.output, force=args.force)
    cells, seeds = registered_inputs(protocol)
    started = time.perf_counter()
    with threadpool_limits(limits=1):
        found = run(cells, seeds)
        pools = threadpool_info()
    results = {"protocol_version": protocol["protocol_version"], "study_id": protocol["study_id"], "part": PART,
               "seeds": seeds, **found}
    runtime = {"seconds": time.perf_counter() - started, "thread_limit": 1, "threadpools": pools}
    sources = tuple(common.ROOT / "experiments" / name for name in
                    ("study3_common.py", "study3_predictability.py", "run_study3_predictability.py", "common.py"))
    manifest = common.write_record(args.output, results, sources=sources, runtime=runtime, force=args.force)
    write_tables(results, args.output)
    print(json.dumps({"output": str(args.output), "results_sha256": manifest["results_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
