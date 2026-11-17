"""Run the two suite questions of protocol 3.0, Part 3 (part_3_suite_questions).

    python -m experiments.run_study3_suites --plan
    python -m experiments.run_study3_suites

The covariance study of protocol 2.0 is repeated through experiments.run_supplement.run_covariance_study
with the Part 3 settings (239 calibration draws, 93 suites, new seeds). Each suite and variant is then
asked two questions with one declared method each: Fisher's combination of its four p-values
(experiments.run_addendum.fisher_combination, flagged when the probability is at most 0.05) for "does
the suite contain any failure", and Holm at 0.05 (experiments.run_supplement.holm_rejections) for
"which condition failed". The run refuses to start until study3_common.require_start allows Part 3;
--plan prints what would run and computes nothing.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
import time

from threadpoolctl import threadpool_info, threadpool_limits

from timegym.statistics import wilson_interval

from . import study3_common as common
from .run_addendum import fisher_combination
from .run_supplement import holm_rejections, run_covariance_study

PART = "part-3"
STUDY2_PROTOCOL = common.ROOT / "experiments/protocol-v2.json"
OFFSETS = (("covariance_reference_samples", "reference_sample_seed_offset"),
           ("covariance_calibration", "calibration_seed_offset"),
           ("covariance_shuffle", "mutation_seed_offset"))
METHODS = ("raw", "holm", "fisher")


def part3_conditions(protocol):
    """The protocol 2.0 covariance conditions, checked against part_3_suite_questions.condition_ids."""
    named = list(protocol["part_3_suite_questions"]["condition_ids"])
    conditions = json.loads(STUDY2_PROTOCOL.read_text())["covariance_study"]["conditions"]
    if [row["id"] for row in conditions] != named:
        raise ValueError(f"Part 3 names the conditions {named}; protocol 2.0 has {[row['id'] for row in conditions]}")
    return conditions


def registered_spec(protocol):
    """The settings dictionary of run_covariance_study for Part 3, with every seed checked against the seed table.

    run_covariance_study derives the seed of suite s and condition c as seed_base + s * suite_seed_stride + c
    and adds the three offsets; this function checks that each of those seeds is the one that
    study3_common.case_seed gives in the namespaces covariance_cases, covariance_reference_samples,
    covariance_calibration and covariance_shuffle.
    """
    part = protocol["part_3_suite_questions"]
    spec = copy.deepcopy(json.loads(STUDY2_PROTOCOL.read_text())["covariance_study"])
    spec["conditions"] = part3_conditions(protocol)
    spec.update(copy.deepcopy(part["settings"]))
    spec["variants"] = list(part["variants"])
    if spec["permutation_invariant_conditions"] != ["factorized_white_noise"]:
        raise ValueError("Part 3 declares factorized_white_noise as the only true null under the partial shuffle")
    if spec["alpha"] != protocol["shared"]["alpha"]:
        raise ValueError("Part 3 uses the shared alpha")
    if spec["n_calibration"] != protocol["shared"]["monte_carlo_rank_checks"]["calibration_draws"]:
        raise ValueError("Part 3 uses the shared number of calibration draws")
    for suite in range(spec["independent_suites"]):
        for index in range(len(spec["conditions"])):
            seed = spec["seed_base"] + suite * spec["suite_seed_stride"] + index
            if seed != common.case_seed(protocol, "covariance_cases", index=index, block=suite):
                raise ValueError(f"Suite {suite}, condition {index}: seed {seed} is not the registered case seed")
            for namespace, offset in OFFSETS:
                if seed + spec[offset] != common.case_seed(protocol, namespace, index=index, block=suite):
                    raise ValueError(f"Suite {suite}, condition {index}: {offset} leaves the {namespace} block")
    return spec


def _suite_row(variant, suite, rows, alpha):
    p = [row["p_value"] for row in rows]
    raw = [bool(value <= alpha) for value in p]
    holm = [bool(flag) for flag in holm_rejections(p, alpha)]
    statistic, fisher_p = fisher_combination(p)
    return {"variant": variant, "suite": suite, "conditions": [row["condition"] for row in rows], "p_values": p,
            "null_true": [bool(row["null_true"]) for row in rows], "raw": raw, "holm": holm,
            "fisher_statistic": statistic, "fisher_p_value": fisher_p, "fisher": bool(fisher_p <= alpha)}


def analyse(records, conditions, variants, alpha):
    """Fisher, Holm and raw decisions per suite and variant, and the counts of ``reported`` with Wilson intervals.

    ``records`` holds one row per suite, condition and variant with ``p_value`` and ``null_true``
    (the rows of run_covariance_study, or hand-made ones). A row that already carries ``raw_alarm`` or
    ``holm_alarm`` must agree with the decision recomputed here.
    """
    by_key = {(row["variant"], row["suite"], row["condition"]): row for row in records}
    if len(by_key) != len(records):
        raise ValueError("Records repeat a suite, condition and variant")
    suites = sorted({row["suite"] for row in records})
    suite_rows = []
    for variant in variants:
        for suite in suites:
            rows = [by_key[variant, suite, condition] for condition in conditions]
            found = _suite_row(variant, suite, rows, alpha)
            for row, raw, holm in zip(rows, found["raw"], found["holm"]):
                if row.get("raw_alarm", raw) != raw or row.get("holm_alarm", holm) != holm:
                    raise RuntimeError(f"Recomputed decision differs for {variant}, suite {suite}, "
                                       f"{row['condition']}")
            suite_rows.append(found)
    variant_summaries, condition_summaries = [], []
    for variant in variants:
        rows = [row for row in suite_rows if row["variant"] == variant]
        nulls = rows[0]["null_true"]
        if any(row["null_true"] != nulls for row in rows):
            raise ValueError(f"True nulls differ between suites of {variant}")
        flagged = {"raw": wilson_interval([any(row["raw"]) for row in rows]),
                   "holm": wilson_interval([any(row["holm"]) for row in rows]),
                   "fisher": wilson_interval([row["fisher"] for row in rows])}
        if any(nulls):
            false = {method: wilson_interval([any(flag and null for flag, null in zip(row[method], nulls))
                                              for row in rows]) for method in ("raw", "holm")}
            # Fisher tests the intersection of the four nulls; it rejects a true null only when all four hold.
            false["fisher"] = wilson_interval([row["fisher"] for row in rows]) if all(nulls) else None
        else:
            false = {method: None for method in METHODS}
        variant_summaries.append({"variant": variant, "n_suites": len(rows),
                                  "true_null_conditions": [c for c, null in zip(conditions, nulls) if null],
                                  "suites_flagged": flagged, "suites_rejecting_a_true_null": false})
        for index, condition in enumerate(conditions):
            condition_summaries.append({"variant": variant, "condition": condition, "null_true": nulls[index],
                                        "raw": wilson_interval([row["raw"][index] for row in rows]),
                                        "holm": wilson_interval([row["holm"][index] for row in rows])})
    return {"alpha": alpha, "conditions": list(conditions), "variants": list(variants), "suites": len(suites),
            "suite_rows": suite_rows, "variant_summaries": variant_summaries,
            "condition_summaries": condition_summaries}


def run(spec):
    """run_covariance_study with ``spec``, then ``analyse`` of its records."""
    covariance = run_covariance_study(spec)
    conditions = [condition["id"] for condition in spec["conditions"]]
    analysis = analyse(covariance["records"], conditions, spec["variants"], spec["alpha"])
    return {"covariance": covariance, "analysis": analysis}


def _write(path, header, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def write_tables(results, output):
    analysis = results["analysis"]
    conditions = analysis["conditions"]
    _write(output / "suites.csv",
           ["variant", "suite", *[f"p_{c}" for c in conditions], *[f"raw_{c}" for c in conditions],
            *[f"holm_{c}" for c in conditions], "fisher_statistic", "fisher_p_value", "fisher_flagged"],
           [[row["variant"], row["suite"], *row["p_values"], *row["raw"], *row["holm"], row["fisher_statistic"],
             row["fisher_p_value"], row["fisher"]] for row in analysis["suite_rows"]])
    rows = []
    for row in analysis["variant_summaries"]:
        for method in METHODS:
            flagged, false = row["suites_flagged"][method], row["suites_rejecting_a_true_null"][method]
            rows.append([row["variant"], method, flagged["count"], flagged["n"], *flagged["ci95"],
                         *(["", "", ""] if false is None else [false["count"], *false["ci95"]])])
    _write(output / "variants.csv", ["variant", "method", "suites_flagged", "n", "lower95", "upper95",
                                     "true_null_rejections", "true_null_lower95", "true_null_upper95"], rows)
    _write(output / "conditions.csv",
           ["variant", "condition", "null_true", "raw", "raw_lower95", "raw_upper95", "holm", "holm_lower95",
            "holm_upper95", "n"],
           [[row["variant"], row["condition"], row["null_true"], row["raw"]["count"], *row["raw"]["ci95"],
             row["holm"]["count"], *row["holm"]["ci95"], row["raw"]["n"]] for row in analysis["condition_summaries"]])
    _write(output / "p-values.csv",
           ["suite", "condition", "variant", "seed", "reference_sample_seed", "calibration_seed", "statistic",
            "p_value", "raw_alarm", "holm_alarm", "null_true"],
           [[row[key] for key in ("suite", "condition", "variant", "seed", "reference_sample_seed",
                                  "calibration_seed", "statistic", "p_value", "raw_alarm", "holm_alarm", "null_true")]
            for row in results["covariance"]["records"]])


def plan(protocol):
    spec = registered_spec(protocol)
    checks = spec["independent_suites"] * len(spec["conditions"]) * len(spec["variants"])
    last = spec["independent_suites"] - 1
    return {"part": PART, "suites": spec["independent_suites"], "conditions": [c["id"] for c in spec["conditions"]],
            "variants": spec["variants"], "covariance_checks": checks, "n_samples": spec["n_samples"],
            "calibration_ensembles": checks * spec["n_calibration"],
            "case_seeds": [common.case_seed(protocol, "covariance_cases", index=0, block=0),
                           common.case_seed(protocol, "covariance_cases", index=3, block=last)],
            "cpu_hours_estimate": protocol["runtime_estimate_cpu_hours"]["part_3"]}


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
    spec = registered_spec(protocol)
    started = time.perf_counter()
    with threadpool_limits(limits=1):
        found = run(spec)
        pools = threadpool_info()
    results = {"protocol_version": protocol["protocol_version"], "study_id": protocol["study_id"], "part": PART,
               "settings": spec, **found}
    runtime = {"seconds": time.perf_counter() - started, "thread_limit": 1, "threadpools": pools}
    sources = tuple(common.ROOT / "experiments" / name for name in
                    ("study3_common.py", "run_study3_suites.py", "run_supplement.py", "run_addendum.py", "common.py"))
    manifest = common.write_record(args.output, results, sources=sources, runtime=runtime, force=args.force)
    write_tables(results, args.output)
    print(json.dumps({"output": str(args.output), "results_sha256": manifest["results_sha256"],
                      "suites": found["analysis"]["suites"]}, indent=2))


if __name__ == "__main__":
    main()
