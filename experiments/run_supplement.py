"""Run the separately versioned CPU supplement without replacing pilot results.

    python -m experiments.run_supplement --output /tmp/timegym-supplement
    python -m experiments.run_supplement --output /tmp/timegym-supplement-repeat

Numerical records exclude runtime and environment metadata. The original
experiments/protocol.json and experiments/results/ are never overwritten.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import time

import numpy as np
from timegym.statistics import paired_t_interval as mean_interval
from timegym.statistics import wilson_interval as wilson

from timegym import Pipeline, Scenario
from timegym.checks import check_shift_equivariance
from timegym.diagnostics import covariance_dependence_check
from timegym.metrics import crps, energy_score, mse, variogram_score
from timegym.mutations import mutate_forecast
from threadpoolctl import threadpool_info, threadpool_limits

from .common import protect_output, sha256

from .model_faults import (FAULTS, LagRidgeImplementation, normal_equation_forecast,
                           point_agreement, target_shift_agreement)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "experiments/protocol-v2.json"
CHECKS = ("contract", "scalar_shift", "target_shift", "conformance", "loss")


def conformal_threshold(calibration_scores, alpha):
    """Uninterpolated split-conformal order statistic; strict exceedance alarm."""
    scores = np.asarray(calibration_scores, dtype=float)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("Calibration scores must be a finite nonempty vector")
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    rank = math.ceil((len(scores)+1) * (1-alpha))
    if rank > len(scores):
        raise ValueError("Too few calibration cases for a finite threshold at this alpha")
    return float(np.sort(scores)[rank-1]), rank


def conformal_pvalue(calibration_scores, score):
    scores = np.asarray(calibration_scores, dtype=float)
    if not np.isfinite(score) or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("Scores must be finite")
    return float((1 + np.count_nonzero(scores >= score)) / (len(scores)+1))


def holm_rejections(pvalues, alpha):
    """Holm step-down rejection mask, retaining the input hypothesis order."""
    pvalues = np.asarray(pvalues, dtype=float)
    if pvalues.ndim != 1 or not len(pvalues) or not np.isfinite(pvalues).all():
        raise ValueError("pvalues must be a finite nonempty vector")
    if np.any((pvalues < 0) | (pvalues > 1)) or not 0 < alpha < 1:
        raise ValueError("Invalid p-values or alpha")
    rejected = np.zeros(len(pvalues), dtype=bool)
    for rank, index in enumerate(np.argsort(pvalues, kind="stable")):
        if pvalues[index] > alpha / (len(pvalues)-rank):
            break
        rejected[index] = True
    return rejected


def scenario_from_condition(condition):
    return Scenario(**{key: value for key, value in condition.items() if key != "id"})


def validate_protocol(protocol):
    if protocol.get("protocol_version") != "2.0":
        raise ValueError("Expected separately versioned supplement protocol 2.0")
    model, covariance = protocol["model_study"], protocol["covariance_study"]
    if tuple(model["faults"]) != FAULTS:
        raise ValueError("Protocol faults must match the implemented selected fixtures")
    if model["evaluation_cases"] < 2 or covariance["independent_suites"] < 2:
        raise ValueError("At least two independent evaluation units are required")
    conformal_threshold(np.zeros(model["calibration_cases"]), model["loss_alpha"])
    if covariance["n_samples"] < 20:
        raise ValueError("Covariance check requires at least 20 predictive samples")
    if 1 / (covariance["n_calibration"]+1) > covariance["alpha"] / len(covariance["conditions"]):
        raise ValueError("Monte Carlo resolution cannot support the first Holm threshold")
    if covariance["suite_seed_stride"] < len(covariance["conditions"]):
        raise ValueError("Suite seed namespaces overlap")
    if model["condition_seed_stride"] < max(model["calibration_cases"], model["evaluation_cases"]):
        raise ValueError("Condition seed namespaces overlap")
    namespaces = []
    for index, condition in enumerate(model["conditions"]):
        scenario_from_condition(condition)
        if condition["n_targets"] != len(model["target_shifts"]):
            raise ValueError("Each condition needs the declared target-specific shifts")
        for stage, count in (("calibration", model["calibration_cases"]), ("evaluation", model["evaluation_cases"])):
            first = model[f"{stage}_seed_base"] + index*model["condition_seed_stride"]
            namespaces.append(set(range(first, first+count)))
    for index, left in enumerate(namespaces):
        if any(left & right for right in namespaces[index+1:]):
            raise ValueError("Model calibration/evaluation seed namespaces overlap")
    for condition in covariance["conditions"]:
        scenario_from_condition(condition)
    if len({c["id"] for c in covariance["conditions"]}) != len(covariance["conditions"]):
        raise ValueError("Covariance conditions must have distinct identifiers")
    conditions = {row["id"]: row for row in covariance["conditions"]}
    nulls = covariance["permutation_invariant_conditions"]
    if len(set(nulls)) != len(nulls) or set(nulls) - conditions.keys():
        raise ValueError("Permutation-invariant conditions must name distinct existing conditions")
    for name in nulls:
        condition = conditions[name]
        if condition["kind"] != "white_noise" or condition["parameters"].get("correlation") != 0:
            raise ValueError("Permutation-invariant conditions require white_noise with correlation 0")


def evaluate_implementation(case, branch, independent, calibration_scores, threshold, spec):
    """All comparisons are made on freshly fitted models and the same case."""
    pipeline = Pipeline(LagRidgeImplementation(branch, spec["window"], spec["ridge_alpha"]))
    forecast = pipeline.predict(case.task.context, case.task.future_index)
    violations = forecast.validate(case.task)
    contract_alarm = bool(violations)
    if contract_alarm:
        raise RuntimeError(f"The semantic-fault fixture unexpectedly violated its contract: {violations}")
    scalar = check_shift_equivariance(pipeline, case.task.context, case.task.future_index,
                                      shift=spec["scalar_shift"], atol=spec["atol"], rtol=spec["rtol"])
    if scalar.status not in {"PASS", "FAIL"}:
        raise RuntimeError(f"Unexpected scalar-shift outcome: {scalar.to_dict()}")
    target = target_shift_agreement(pipeline, case.task.context, case.task.future_index, forecast,
                                    shifts=spec["target_shifts"], atol=spec["atol"], rtol=spec["rtol"])
    differential = point_agreement(forecast.point, independent.point, atol=spec["atol"], rtol=spec["rtol"])
    loss = mse(case.observed.to_numpy(), forecast.point)
    pvalue = conformal_pvalue(calibration_scores, loss)
    loss_alarm = bool(loss > threshold)
    if loss_alarm != (pvalue <= spec["loss_alpha"]):
        raise RuntimeError("Conformal rank and threshold decisions disagree")
    return {
        "branch": branch,
        "alarms": {"contract": contract_alarm, "scalar_shift": scalar.status == "FAIL",
                   "target_shift": target["alarm"], "conformance": differential["alarm"],
                   "loss": loss_alarm},
        "realized_mse": loss,
        "mean_reference_mse": mse(case.reference.mean, forecast.point),
        "loss_rank_pvalue": pvalue,
        "scalar_shift_max_error": scalar.evidence["max_absolute_error"],
        "target_shift_max_error": target["max_absolute_error"],
        "conformance_max_error": differential["max_absolute_error"],
        "forecast_sha256": hashlib.sha256(np.ascontiguousarray(forecast.point).tobytes()).hexdigest(),
    }, forecast


def run_model_study(spec):
    calibration, records, examples = [], [], []
    condition_summaries, loss_summaries = [], []
    for condition_index, condition in enumerate(spec["conditions"]):
        scenario = scenario_from_condition(condition)
        calibration_scores = []
        for index in range(spec["calibration_cases"]):
            seed = spec["calibration_seed_base"] + condition_index*spec["condition_seed_stride"] + index
            case = scenario.generate(seed=seed)
            model = Pipeline(LagRidgeImplementation(window=spec["window"], alpha=spec["ridge_alpha"]))
            output = model.predict(case.task.context, case.task.future_index)
            score = mse(case.observed.to_numpy(), output.point)
            calibration_scores.append(score)
            calibration.append({"condition": condition["id"], "seed": seed, "realized_mse": score})
        threshold, rank = conformal_threshold(calibration_scores, spec["loss_alpha"])
        paired = []
        for index in range(spec["evaluation_cases"]):
            seed = spec["evaluation_seed_base"] + condition_index*spec["condition_seed_stride"] + index
            case = scenario.generate(seed=seed)
            independent = normal_equation_forecast(case.task.context, case.task.future_index,
                                                   window=spec["window"], alpha=spec["ridge_alpha"])
            branches, outputs = {}, {}
            for branch in ("clean",) + tuple(spec["faults"]):
                result, output = evaluate_implementation(case, branch, independent, calibration_scores, threshold, spec)
                result.update(condition=condition["id"], seed=seed)
                branches[branch], outputs[branch] = result, output
                records.append(result)
            paired.append(branches)
            if index == 0:
                examples.append({"condition": condition["id"], "manifest": case.manifest,
                                 "independent_point": independent.point.tolist(),
                                 "branch_points": {name: forecast.point.tolist() for name, forecast in outputs.items()}})
        for fault in spec["faults"]:
            for check in CHECKS:
                clean = np.asarray([row["clean"]["alarms"][check] for row in paired])
                broken = np.asarray([row[fault]["alarms"][check] for row in paired])
                condition_summaries.append({
                    "condition": condition["id"], "fault": fault, "check": check,
                    "clean": wilson(clean), "faulty": wilson(broken),
                    "paired_alarm_gain": int(np.sum(broken & ~clean)),
                    "paired_alarm_loss": int(np.sum(clean & ~broken)),
                    "loss_calibration_threshold": threshold, "loss_calibration_rank": rank,
                })
            loss_summaries.append({
                "condition": condition["id"], "fault": fault,
                "clean_mse": mean_interval([row["clean"]["realized_mse"] for row in paired]),
                "faulty_mse": mean_interval([row[fault]["realized_mse"] for row in paired]),
                "repair_minus_faulty_mse": mean_interval([row["clean"]["realized_mse"]-row[fault]["realized_mse"] for row in paired]),
                "repair_minus_faulty_reference_mse": mean_interval([row["clean"]["mean_reference_mse"]-row[fault]["mean_reference_mse"] for row in paired]),
                "cases_improved_by_repair": int(sum(row["clean"]["realized_mse"] < row[fault]["realized_mse"] for row in paired)),
                "n": len(paired),
            })
        print(f"Model condition finished: {condition['id']}", flush=True)
    ablation = []
    for check in CHECKS:
        clean = [row for row in records if row["branch"] == "clean"]
        faulty = [row for row in records if row["branch"] in FAULTS]
        ablation.append({"check": check,
                         "clean_alarms": sum(row["alarms"][check] for row in clean), "clean_n": len(clean),
                         "fault_alarms": sum(row["alarms"][check] for row in faulty), "fault_n": len(faulty),
                         "scope": "Descriptive counts over the fixed equally weighted condition corpus; no pooled IID interval"})
    return {"calibration_records": calibration, "records": records,
            "condition_summaries": condition_summaries, "loss_summaries": loss_summaries,
            "ablation_counts": ablation, "examples": examples,
            "cost_counts": {"sklearn_fits": len(calibration)+4*len(records),
                            "independent_ridge_solves": len(spec["conditions"])*spec["evaluation_cases"]}}


def run_covariance_study(spec):
    records, suite_records = [], []
    for suite in range(spec["independent_suites"]):
        rows_by_variant = {name: [] for name in spec["variants"]}
        for condition_index, condition in enumerate(spec["conditions"]):
            seed = spec["seed_base"] + suite*spec["suite_seed_stride"] + condition_index
            case = scenario_from_condition(condition).generate(seed=seed)
            clean = case.reference.forecast(spec["n_samples"], seed+spec["reference_sample_seed_offset"])
            shuffled = mutate_forecast(clean, "dependence_shuffle",
                                      seed=seed+spec["mutation_seed_offset"], strength=spec["shuffle_strength"])
            np.testing.assert_array_equal(np.sort(clean.samples, axis=0), np.sort(shuffled.samples, axis=0))
            samples = {"clean": clean.samples, "partial_shuffle": shuffled.samples,
                       "reduced_spread": clean.point[None] + spec["spread_multiplier"]*(clean.samples-clean.point[None])}
            score_values = {}
            for variant, ensemble in samples.items():
                check = covariance_dependence_check(case.reference, ensemble,
                                                     seed=seed+spec["calibration_seed_offset"],
                                                     n_calibration=spec["n_calibration"], alpha=spec["alpha"])
                if check.status not in {"PASS", "FAIL"}:
                    raise RuntimeError(f"Unexpected covariance outcome: {check.to_dict()}")
                observed = case.observed.to_numpy()
                scores = {"crps": crps(observed, ensemble), "energy": energy_score(observed, ensemble),
                          "variogram": variogram_score(observed, ensemble)}
                score_values[variant] = scores
                null_true = variant == "clean" or (variant == "partial_shuffle" and condition["id"] in spec["permutation_invariant_conditions"])
                row = {"suite": suite, "condition": condition["id"], "seed": seed, "variant": variant,
                       "reference_sample_seed": seed+spec["reference_sample_seed_offset"],
                       "calibration_seed": seed+spec["calibration_seed_offset"],
                       "p_value": check.evidence["p_value"], "statistic": check.evidence["statistic"],
                       "raw_alarm": check.status == "FAIL", "null_true": null_true, "scores": scores,
                       "score_delta_from_clean": {key: value-score_values["clean"][key] for key, value in scores.items()}}
                rows_by_variant[variant].append(row)
            np.testing.assert_allclose(score_values["clean"]["crps"], score_values["partial_shuffle"]["crps"], atol=1e-12, rtol=1e-12)
        for variant, rows in rows_by_variant.items():
            rejected = holm_rejections([row["p_value"] for row in rows], spec["alpha"])
            for row, alarm in zip(rows, rejected):
                row["holm_alarm"] = bool(alarm)
                records.append(row)
            nulls = [row for row in rows if row["null_true"]]
            alternatives = [row for row in rows if not row["null_true"]]
            suite_records.append({"suite": suite, "variant": variant, "family_size": len(rows),
                                  "raw_any_alarm": any(row["raw_alarm"] for row in rows),
                                  "holm_any_alarm": any(row["holm_alarm"] for row in rows),
                                  "null_hypotheses": len(nulls), "alternative_hypotheses": len(alternatives),
                                  "raw_any_false_alarm": any(row["raw_alarm"] for row in nulls) if nulls else None,
                                  "holm_any_false_alarm": any(row["holm_alarm"] for row in nulls) if nulls else None})
    condition_summaries = []
    for condition in spec["conditions"]:
        for variant in spec["variants"]:
            rows = [row for row in records if row["condition"] == condition["id"] and row["variant"] == variant]
            condition_summaries.append({"condition": condition["id"], "variant": variant,
                                        "null_true": rows[0]["null_true"],
                                        "raw": wilson([row["raw_alarm"] for row in rows]),
                                        "holm": wilson([row["holm_alarm"] for row in rows]),
                                        "score_deltas": {key: mean_interval([row["score_delta_from_clean"][key] for row in rows]) for key in ("crps", "energy", "variogram")}})
    suite_summaries = []
    for variant in spec["variants"]:
        rows = [row for row in suite_records if row["variant"] == variant]
        suite_summaries.append({"variant": variant, "family_size": rows[0]["family_size"],
                                "null_hypotheses": rows[0]["null_hypotheses"],
                                **{key: (wilson([row[key] for row in rows]) if rows[0][key] is not None else None)
                                   for key in ("raw_any_alarm", "holm_any_alarm", "raw_any_false_alarm", "holm_any_false_alarm")}})
    checks = len(records)
    return {"records": records, "suite_records": suite_records, "condition_summaries": condition_summaries,
            "suite_summaries": suite_summaries,
            "cost_counts": {"covariance_checks": checks, "calibration_ensembles": checks*spec["n_calibration"],
                            "calibration_trajectories": checks*spec["n_calibration"]*spec["n_samples"]}}


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def export_tables(results, output):
    write_csv(output / "model-ablation.csv", results["model"]["ablation_counts"])
    write_csv(output / "model-conditions.csv", [{
        "condition": row["condition"], "fault": row["fault"], "check": row["check"],
        "clean_alarms": row["clean"]["count"], "fault_alarms": row["faulty"]["count"],
        "n": row["clean"]["n"],
        "gained_alarms": row["paired_alarm_gain"], "lost_alarms": row["paired_alarm_loss"],
    } for row in results["model"]["condition_summaries"]])
    write_csv(output / "model-repair-losses.csv", [{
        "condition": row["condition"], "fault": row["fault"],
        "clean_mse": row["clean_mse"]["mean"], "faulty_mse": row["faulty_mse"]["mean"],
        "repair_minus_faulty_mse": row["repair_minus_faulty_mse"]["mean"],
        "paired_lower95": row["repair_minus_faulty_mse"]["ci95"][0],
        "paired_upper95": row["repair_minus_faulty_mse"]["ci95"][1],
        "improved_cases": row["cases_improved_by_repair"], "n": row["n"],
    } for row in results["model"]["loss_summaries"]])
    write_csv(output / "covariance-conditions.csv", [{
        "condition": row["condition"], "variant": row["variant"], "null_true": row["null_true"],
        "raw_alarms": row["raw"]["count"], "holm_alarms": row["holm"]["count"], "n": row["raw"]["n"],
        "raw_lower95": row["raw"]["ci95"][0], "raw_upper95": row["raw"]["ci95"][1],
        "holm_lower95": row["holm"]["ci95"][0], "holm_upper95": row["holm"]["ci95"][1],
    } for row in results["covariance"]["condition_summaries"]])


def source_fingerprints():
    paths = sorted((ROOT / "timegym").rglob("*.py"))
    paths += [ROOT / "experiments/model_faults.py", ROOT / "experiments/common.py", Path(__file__).resolve(), PROTOCOL]
    return {str(path.relative_to(ROOT)): sha256(path) for path in paths}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/results-v2")
    parser.add_argument("--force", action="store_true", help="Replace existing study artifacts")
    args = parser.parse_args()
    if args.output.resolve() == (ROOT / "experiments/results").resolve():
        parser.error("The original pilot results directory is protected; choose a separate output")
    protect_output(args.output, force=args.force)
    protocol = json.loads(PROTOCOL.read_text())
    validate_protocol(protocol)
    fingerprints = source_fingerprints()
    started = time.perf_counter()
    with threadpool_limits(limits=1):
        model = run_model_study(protocol["model_study"])
        model_seconds = time.perf_counter()-started
        covariance_started = time.perf_counter()
        covariance = run_covariance_study(protocol["covariance_study"])
        covariance_seconds = time.perf_counter()-covariance_started
        pools = threadpool_info()
    if fingerprints != source_fingerprints():
        raise RuntimeError("Source files changed during the study; rerun after the shared export completes")
    results = {"study_id": protocol["study_id"], "protocol_version": protocol["protocol_version"],
               "protocol_sha256": sha256(PROTOCOL), "model": model, "covariance": covariance}
    args.output.mkdir(parents=True, exist_ok=True)
    numerical = args.output / "results.json"
    numerical.write_text(json.dumps(results, indent=2, sort_keys=True, allow_nan=False)+"\n")
    manifest = {"protocol": protocol, "protocol_sha256": sha256(PROTOCOL), "results_sha256": sha256(numerical),
                "source_files": fingerprints, "python": platform.python_version(), "platform": platform.platform(),
                "versions": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn")},
                "original_pilot_results_sha256": sha256(ROOT / "experiments/results/results.json"),
                "source_note": "Hashes identify the working source, including generated package files; no uncommitted changes are represented by a base Git revision alone."}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False)+"\n")
    runtime = {"model_seconds": model_seconds, "covariance_seconds": covariance_seconds,
               "total_seconds": model_seconds+covariance_seconds,
               "thread_limit": 1, "threadpools": pools,
               "threads": {name: os.environ.get(name) for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
               "note": "Observed wall time for this execution; excluded from deterministic numerical results."}
    (args.output / "runtime.json").write_text(json.dumps(runtime, indent=2, sort_keys=True)+"\n")
    export_tables(results, args.output)
    print(json.dumps({"output": str(args.output), "results_sha256": sha256(numerical),
                      "runtime": runtime, "model_ablation": model["ablation_counts"],
                      "covariance_suites": covariance["suite_summaries"]}, indent=2))


if __name__ == "__main__":
    main()
