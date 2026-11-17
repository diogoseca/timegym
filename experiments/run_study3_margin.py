"""Run the margin simulation of protocol 3.0, Part 4 (part_4_joint_spread_margin).

    python -m experiments.run_study3_margin --plan
    python -m experiments.run_study3_margin --workers 4

For every cell (condition, margin, true distance and direction, number of paths) and every
repetition the runner builds an ensemble whose covariance is exactly c^2 times the reference
covariance, so that the true relative distance is d, applies the decision of
experiments/study3_margin.py and counts the three verdicts. Cells may run in parallel worker
processes; every random number of a repetition comes from SeedSequence([12000000, j, r]) alone, so
the record does not depend on the number of workers. The run refuses to start until
study3_common.require_start allows Part 4; --plan prints what would run and computes nothing.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import json
import math
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

from timegym.statistics import wilson_interval

from . import study3_common as common
from .study3_margin import VERDICTS, estimate, decide
from .run_study3_suites import part3_conditions
from .run_supplement import scenario_from_condition

PART = "part-4"


def distance_grid(margin, multiples):
    """The distances of one margin: d = 0 once, then each nonzero multiple shrunk and inflated, in that order."""
    grid = []
    for multiple in multiples:
        if multiple == 0:
            grid.append({"multiple": float(multiple), "direction": "none", "d": 0.0})
            continue
        for direction in ("shrink", "inflate"):
            grid.append({"multiple": float(multiple), "direction": direction, "d": float(multiple) * margin})
    return grid


def spread_multiplier(d, direction):
    """c with |c^2 - 1| = d: sqrt(1 - d) to shrink, sqrt(1 + d) to inflate."""
    if direction == "none":
        if d != 0:
            raise ValueError("Only d = 0 has no direction")
        return 1.0
    if direction == "shrink":
        if not 0 < d < 1:
            raise ValueError("A shrink needs 0 < d < 1")
        return math.sqrt(1 - d)
    if direction == "inflate":
        return math.sqrt(1 + d)
    raise ValueError(f"Unknown direction {direction!r}")


def build_cells(condition_ids, margins, multiples, sample_paths):
    """All cells, indexed j in the nesting order conditions x margins x distances x path counts."""
    cells = []
    for condition_index, condition in enumerate(condition_ids):
        for margin in margins:
            for distance in distance_grid(float(margin), multiples):
                for n_paths in sample_paths:
                    cells.append({"j": len(cells), "condition_index": condition_index, "condition": condition,
                                  "margin": float(margin), **distance, "n_paths": int(n_paths)})
    return cells


def ensemble_seed(tag, j, r):
    """The sampling seed of cell j, repetition r: SeedSequence([tag, j, r]).generate_state(1)[0]."""
    return int(np.random.SeedSequence([int(tag), int(j), int(r)]).generate_state(1)[0])


def ensemble(reference, d, direction, n_paths, seed):
    """reference.mean + c (Z - reference.mean) with Z the n_paths draws reference.sample(n_paths, seed)."""
    c = spread_multiplier(d, direction)
    mean = np.asarray(reference.mean, dtype=float)
    draws = np.asarray(reference.sample(n_paths, seed), dtype=float)
    return mean[None] + c * (draws - mean[None])


def simulate_cell(reference, cell, repetitions, tag):
    """Verdict counts with Wilson intervals, and the values of T and its standard error, for one cell."""
    covariance = np.asarray(reference.covariance, dtype=float)
    estimates, errors, verdicts = [], [], []
    for r in range(repetitions):
        paths = ensemble(reference, cell["d"], cell["direction"], cell["n_paths"], ensemble_seed(tag, cell["j"], r))
        found = estimate(paths, covariance)
        decision = decide(found["T"], found["standard_error"], cell["margin"])
        estimates.append(found["T"])
        errors.append(found["standard_error"])
        verdicts.append(decision["verdict"])
    counts = {name: wilson_interval([verdict == name for verdict in verdicts]) for name in VERDICTS}
    return {"j": cell["j"], "condition": cell["condition"], "margin": cell["margin"],
            "multiple": cell["multiple"], "direction": cell["direction"], "d": cell["d"],
            "d_squared": cell["d"] ** 2, "n_paths": cell["n_paths"], "repetitions": repetitions,
            "verdicts": counts, "mean_T": float(np.mean(estimates)),
            "mean_T_minus_d_squared": float(np.mean(estimates)) - cell["d"] ** 2,
            "mean_standard_error": float(np.mean(errors)), "T": estimates, "standard_error": errors}


def _worker(arguments):
    reference, cell, repetitions, tag = arguments
    with threadpool_limits(limits=1):
        return simulate_cell(reference, cell, repetitions, tag)


def run_simulation(references, cells, repetitions, tag, workers=1):
    """Every cell, in order of j; ``references[cell['condition_index']]`` is the reference of a cell's condition.

    ``workers`` > 1 runs cells in that many processes. The result is the same for any number of workers.
    """
    tasks = [(references[cell["condition_index"]], cell, int(repetitions), int(tag)) for cell in cells]
    if workers <= 1:
        rows = [_worker(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(_worker, tasks))
    return sorted(rows, key=lambda row: row["j"])


def settings(protocol):
    """Conditions, margins, grid, path counts, repetitions, the generator tag and the case seeds of Part 4."""
    part = protocol["part_4_joint_spread_margin"]
    conditions = part3_conditions(protocol)
    found = {"conditions": conditions, "margins": [float(value) for value in part["margins"]["values"]],
             "multiples": [float(value) for value in part["distance_grid"]["multiples_of_margin"]],
             "sample_paths": [int(value) for value in part["sample_paths"]],
             "repetitions": int(part["repetitions"]["value"]),
             "tag": common.generator_tag(protocol, "margin_ensembles"),
             "case_seeds": [common.case_seed(protocol, "margin_cases", index=index)
                            for index in range(len(conditions))]}
    cells = build_cells([row["id"] for row in conditions], found["margins"], found["multiples"],
                        found["sample_paths"])
    per_margin = len(distance_grid(1.0, found["multiples"]))
    if per_margin != 15 or len(cells) != 240:
        raise ValueError(f"Expected 15 distances per margin and 240 cells; got {per_margin} and {len(cells)}")
    return found, cells


def references_for(conditions, case_seeds):
    """The reference law of the one fixed case of each condition."""
    return [scenario_from_condition(condition).generate(seed=seed).reference
            for condition, seed in zip(conditions, case_seeds)]


def write_tables(rows, output):
    with (output / "cells.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["j", "condition", "margin", "multiple", "direction", "d", "n_paths", "repetitions"]
                        + [f"{name}_{field}" for name in VERDICTS for field in ("count", "lower95", "upper95")]
                        + ["mean_T", "d_squared", "mean_standard_error"])
        for row in rows:
            verdicts = [value for name in VERDICTS for value in (row["verdicts"][name]["count"],
                                                                 *row["verdicts"][name]["ci95"])]
            writer.writerow([row["j"], row["condition"], row["margin"], row["multiple"], row["direction"], row["d"],
                             row["n_paths"], row["repetitions"], *verdicts, row["mean_T"], row["d_squared"],
                             row["mean_standard_error"]])


def plan(protocol):
    found, cells = settings(protocol)
    estimate_text = protocol["runtime_estimate_cpu_hours"]["part_4"]
    return {"part": PART, "cells": len(cells), "repetitions_per_cell": found["repetitions"],
            "ensembles": len(cells) * found["repetitions"], "margins": found["margins"],
            "distances_per_margin": len(distance_grid(1.0, found["multiples"])),
            "sample_paths": found["sample_paths"],
            "case_seeds": found["case_seeds"], "generator_tag": found["tag"],
            "cpu_hours_estimate": estimate_text,
            "first_cells": [{key: cell[key] for key in ("j", "condition_index", "margin", "direction", "d",
                                                        "n_paths")} for cell in cells[:3]]}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=common.RESULTS / PART)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--force", action="store_true", help="Replace existing study artifacts")
    parser.add_argument("--plan", action="store_true", help="Print what would run and compute nothing")
    args = parser.parse_args()
    protocol = common.load_protocol()
    if args.plan:
        print(json.dumps(plan(protocol), indent=2))
        return
    common.require_start(protocol, PART)
    common.protect_output(args.output, force=args.force)
    found, cells = settings(protocol)
    started = time.perf_counter()
    with threadpool_limits(limits=1):
        references = references_for(found["conditions"], found["case_seeds"])
        rows = run_simulation(references, cells, found["repetitions"], found["tag"], workers=args.workers)
        pools = threadpool_info()
    results = {"protocol_version": protocol["protocol_version"], "study_id": protocol["study_id"], "part": PART,
               "case_seeds": found["case_seeds"], "generator_tag": found["tag"],
               "repetitions": found["repetitions"], "cells": rows}
    runtime = {"seconds": time.perf_counter() - started, "workers": args.workers, "thread_limit": 1,
               "threadpools": pools}
    sources = tuple(common.ROOT / "experiments" / name for name in
                    ("study3_common.py", "study3_margin.py", "run_study3_margin.py", "run_study3_suites.py",
                     "run_supplement.py", "common.py"))
    manifest = common.write_record(args.output, results, sources=sources, runtime=runtime, force=args.force)
    write_tables(rows, args.output)
    print(json.dumps({"output": str(args.output), "results_sha256": manifest["results_sha256"],
                      "cells": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
