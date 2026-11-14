"""Run the exploratory follow-up to Study 2 (protocol 2.1) without touching frozen records.

    python -m experiments.run_followup --output experiments/results-v2-followup

The analyses were chosen on 2026-11-13, after the protocol 2.0 results were
known (see experiments/protocol-v2-followup.json). The runner imports the
frozen Study 1 and Study 2 code rather than copying it, re-runs only the
fits it needs, checks every re-run value against the frozen records, and
writes results.json and manifest.json. Frozen records are read, never written.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import platform

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import timegym
from timegym import Pipeline, Scenario
from timegym.metrics import mse

from .common import protect_output, sha256
from .model_faults import LagRidgeImplementation
from .run_study import capability_models
from .run_supplement import conformal_threshold, scenario_from_condition

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "experiments/protocol-v2-followup.json"
STUDY1_PROTOCOL = ROOT / "experiments/protocol.json"
STUDY1_RESULTS = ROOT / "experiments/results/results.json"
STUDY2_PROTOCOL = ROOT / "experiments/protocol-v2.json"
STUDY2_RESULTS = ROOT / "experiments/results-v2/results.json"
VERSION = "2.1"


def validate_protocol(protocol, root=ROOT):
    if protocol.get("protocol_version") != VERSION or protocol.get("status") != "exploratory":
        raise ValueError("Expected the exploratory follow-up protocol 2.1")
    for name, item in protocol["frozen_inputs"].items():
        if sha256(root / item["path"]) != item["sha256"]:
            raise ValueError(f"Frozen input {item['path']} differs from the digest the follow-up declares")


def answer_key_counts(calibration_distances, records, alpha, conditions, branches):
    """Thresholds and alarm counts from stored calibration distances and frozen evaluation records.

    This is the cheap part of the answer-key alarm: it needs no model fit, so
    the tests recompute it from the committed record.
    """
    rows = []
    for condition in conditions:
        threshold, rank = conformal_threshold(calibration_distances[condition], alpha)
        for branch in branches:
            cases = [row for row in records if row["condition"] == condition and row["branch"] == branch]
            rows.append({"condition": condition, "branch": branch, "n": len(cases),
                         "answer_key_threshold": threshold, "answer_key_rank": rank,
                         "answer_key_alarms": int(sum(row["mean_reference_mse"] > threshold for row in cases)),
                         "loss_alarms": int(sum(row["alarms"]["loss"] for row in cases))})
    return rows


def answer_key_totals(rows, faults, excluded):
    def total(branches):
        chosen = [row for row in rows if row["branch"] in branches and row["condition"] not in excluded]
        return {"answer_key_alarms": sum(row["answer_key_alarms"] for row in chosen),
                "loss_alarms": sum(row["loss_alarms"] for row in chosen),
                "n": sum(row["n"] for row in chosen)}
    return {"subtle_faults": {"faults": list(faults), **total(set(faults))},
            "clean": total({"clean"}), "excluded_conditions": list(excluded)}


def answer_key_alarm(spec, study2_protocol, study2):
    model = study2_protocol["model_study"]
    frozen = {(row["condition"], row["seed"]): row["realized_mse"] for row in study2["model"]["calibration_records"]}
    frozen_loss = {row["condition"]: (row["loss_calibration_threshold"], row["loss_calibration_rank"])
                   for row in study2["model"]["condition_summaries"] if row["check"] == "loss"}
    distances, largest = {}, 0.0
    loss_thresholds = {}
    for index, condition in enumerate(model["conditions"]):
        scenario = scenario_from_condition(condition)
        realized, distance = [], []
        for case_number in range(model["calibration_cases"]):
            seed = model["calibration_seed_base"] + index * model["condition_seed_stride"] + case_number
            case = scenario.generate(seed=seed)
            forecast = Pipeline(LagRidgeImplementation(window=model["window"], alpha=model["ridge_alpha"])).predict(
                case.task.context, case.task.future_index)
            loss = mse(case.observed.to_numpy(), forecast.point)
            difference = abs(loss - frozen[condition["id"], seed])
            largest = max(largest, difference)
            if difference > spec["reproduction_tolerance"]:
                raise RuntimeError(f"Re-run calibration case {condition['id']} seed {seed} differs from the "
                                   f"frozen realized MSE by {difference:.3e}")
            realized.append(loss)
            distance.append(mse(case.reference.mean, forecast.point))
        threshold, rank = conformal_threshold(realized, model["loss_alpha"])
        frozen_threshold, frozen_rank = frozen_loss[condition["id"]]
        if rank != frozen_rank or abs(threshold - frozen_threshold) > spec["reproduction_tolerance"]:
            raise RuntimeError(f"Re-run loss threshold for {condition['id']} differs from the frozen one")
        distances[condition["id"]] = distance
        loss_thresholds[condition["id"]] = {"threshold": frozen_threshold, "rank": frozen_rank}
        print(f"Calibration re-run finished: {condition['id']}", flush=True)
    conditions = [condition["id"] for condition in model["conditions"]]
    rows = answer_key_counts(distances, study2["model"]["records"], spec["alpha"], conditions, spec["branches"])
    for row in rows:
        row["loss_threshold"] = loss_thresholds[row["condition"]]["threshold"]
    return {"alpha": spec["alpha"], "conditions": conditions,
            "largest_calibration_difference": largest,
            "calibration_distances": distances, "rows": rows,
            "totals": answer_key_totals(rows, spec["summary_faults"], spec["summary_excluded_conditions"])}


def repair_mechanism(spec, tolerance, study2_protocol, study2):
    model = study2_protocol["model_study"]
    ids = [condition["id"] for condition in model["conditions"]]
    index = ids.index(spec["condition"])
    condition = model["conditions"][index]
    scenario = scenario_from_condition(condition)
    frozen = {(row["branch"], row["seed"]): row for row in study2["model"]["records"]
              if row["condition"] == spec["condition"]}
    sums, squared = [], {branch: [] for branch in spec["branches"]}
    for case_number in range(model["evaluation_cases"]):
        seed = model["evaluation_seed_base"] + index * model["condition_seed_stride"] + case_number
        case = scenario.generate(seed=seed)
        answer = np.asarray(case.reference.mean, dtype=float)
        for branch in spec["branches"]:
            forecast = Pipeline(LagRidgeImplementation(branch, model["window"], model["ridge_alpha"])).predict(
                case.task.context, case.task.future_index)
            distance = mse(answer, forecast.point)
            if abs(distance - frozen[branch, seed]["mean_reference_mse"]) > tolerance:
                raise RuntimeError(f"Re-run {branch} case {seed} differs from the frozen distance to the answer key")
            squared[branch].append(((np.asarray(forecast.point) - answer) ** 2).mean(axis=1))
        fitted = LagRidgeImplementation("clean", model["window"], model["ridge_alpha"]).fit(case.task.context)
        targets = fitted.history_.shape[1]
        weights = fitted.regressor_.coef_ * fitted.targets_.scale_[:, None] / fitted.features_.scale_[None, :]
        weights = weights.reshape(targets, model["window"], targets)
        sums.append([float(weights[d, :, d].sum()) for d in range(targets)])
    sums = np.asarray(sums)
    clean = {seed: row["mean_reference_mse"] for (branch, seed), row in frozen.items() if branch == "clean"}
    closer = sum(frozen[spec["branches"][1], seed]["mean_reference_mse"] < value for seed, value in clean.items())
    summary = next(row for row in study2["model"]["loss_summaries"]
                   if row["condition"] == spec["condition"] and row["fault"] == spec["branches"][1])
    return {"condition": spec["condition"], "fault": spec["branches"][1], "n": len(sums),
            "own_lag_coefficient_sum": {"mean": sums.mean(axis=0).tolist(), "min": sums.min(axis=0).tolist(),
                                        "max": sums.max(axis=0).tolist()},
            "per_step_squared_distance": {branch: np.mean(values, axis=0).tolist()
                                          for branch, values in squared.items()},
            "faulty_closer_to_answer_key": int(closer),
            "frozen_clean_minus_faulty_reference_mse": summary["clean_minus_faulty_reference_mse"]}


def scorecard_pipeline():
    return Pipeline(window_length=24).add(StandardScaler()).add(Ridge(alpha=1.0))


def cell_score(unavoidable, avoidable):
    from timegym.scorecard import _score
    return _score(float(np.mean(unavoidable)), float(np.mean(avoidable)))


def scorecard_precision(spec):
    from timegym.scorecard import _case_numbers
    card = scorecard_pipeline().scorecard(**spec["scorecard"])
    figure = scorecard_pipeline().scorecard()
    entries = {}
    for entry in card.run.cases:
        numbers = _case_numbers(entry)
        if numbers is None:
            raise RuntimeError("Every scorecard case must be scorable")
        entries.setdefault(entry.scenario_index, []).append((entry.seed_index, numbers[0], numbers[1]))
    boot = spec["bootstrap"]
    generator = np.random.default_rng(boot["seed"])
    cells = []
    for index, row in enumerate(card.table.itertuples()):
        values = sorted(entries[index])
        if [value[0] for value in values] != list(range(spec["scorecard"]["instances"])):
            raise RuntimeError("Scorecard instances are missing or out of order")
        unavoidable = np.asarray([value[1] for value in values])
        avoidable = np.asarray([value[2] for value in values])
        first = spec["figure_instances"]
        score_first = cell_score(unavoidable[:first], avoidable[:first])
        shown = figure.table.iloc[index]
        if (shown.pattern, shown.noise) != (row.pattern, row.noise) or abs(shown.score - score_first) > 1e-9:
            raise RuntimeError(f"First-ten score of {row.pattern} / {row.noise} differs from the Figure 1 scorecard")
        scores = []
        for _ in range(boot["resamples"]):
            pick = generator.integers(0, len(values), boot["size"])
            scores.append(cell_score(unavoidable[pick], avoidable[pick]))
        low, high = np.percentile(scores, [2.5, 97.5])
        cells.append({"pattern": row.pattern, "title": row.title, "noise": row.noise,
                      "noise_sd": float(row.noise_sd), "score_first_10": score_first,
                      "figure_score": float(shown.score), "score_30": cell_score(unavoidable, avoidable),
                      "bootstrap_sd": float(np.std(scores)), "bootstrap_p2_5": float(low),
                      "bootstrap_p97_5": float(high)})
    sds = [cell["bootstrap_sd"] for cell in cells]
    return {"cells": cells, "smallest_sd": min(sds), "largest_sd": max(sds),
            "smallest_sd_cell": cells[int(np.argmin(sds))]["pattern"],
            "largest_sd_cell": cells[int(np.argmax(sds))]["pattern"]}


def trend_ar1(spec, tolerance, study1_protocol, study1):
    task = study1_protocol["capability_task"]
    frozen = {(row["sigma"], row["seed"]): row["mean_reference_mse"] for row in study1["capability_records"]
              if row["scenario"] == spec["scenario"] and row["model"] == "gaussian_ar1"}
    rows = []
    for sigma in spec["sigma"]:
        scenario = Scenario(spec["scenario"], parameters={"sigma": sigma},
                            **{key: task[key] for key in ("context_length", "horizon", "n_targets")})
        pipeline = Pipeline(capability_models(task)["gaussian_ar1"]())
        phis, offsets = [], []
        for seed in range(*spec["seeds"]):
            case = scenario.generate(seed=seed)
            forecast = pipeline.predict(case.task.context, case.task.future_index)
            answer = np.asarray(case.reference.mean, dtype=float)
            if abs(mse(answer, forecast.point) - frozen[sigma, seed]) > tolerance:
                raise RuntimeError(f"Re-run AR(1) trend case sigma={sigma} seed={seed} differs from Study 1")
            model = capability_models(task)["gaussian_ar1"]().fit(case.task.context)
            phis.append(float(np.mean(model.phi_)))
            offsets.append(float(np.mean(np.asarray(forecast.point) - answer)))
        rows.append({"sigma": sigma, "n": len(phis), "mean_phi": float(np.mean(phis)),
                     "min_phi": float(np.min(phis)), "max_phi": float(np.max(phis)),
                     "mean_forecast_minus_answer_key": float(np.mean(offsets))})
    return {"scenario": spec["scenario"], "model": "gaussian_ar1", "rows": rows}


def source_fingerprints():
    paths = sorted((ROOT / "timegym").rglob("*.py"))
    paths += [ROOT / "experiments" / name for name in
              ("common.py", "model_faults.py", "run_study.py", "run_supplement.py", "run_followup.py")]
    paths += [PROTOCOL]
    return {str(path.relative_to(ROOT)): sha256(path) for path in paths}


def run(protocol):
    tolerance = protocol["answer_key_alarm"]["reproduction_tolerance"]
    study1_protocol = json.loads(STUDY1_PROTOCOL.read_text())
    study1 = json.loads(STUDY1_RESULTS.read_text())
    study2_protocol = json.loads(STUDY2_PROTOCOL.read_text())
    study2 = json.loads(STUDY2_RESULTS.read_text())
    return {
        "study_id": protocol["study_id"], "protocol_version": protocol["protocol_version"],
        "protocol_sha256": sha256(PROTOCOL), "status": protocol["status"],
        "answer_key_alarm": answer_key_alarm(protocol["answer_key_alarm"], study2_protocol, study2),
        "repair_mechanism": repair_mechanism(protocol["repair_mechanism"], tolerance, study2_protocol, study2),
        "scorecard_precision": scorecard_precision(protocol["scorecard_precision"]),
        "trend_ar1": trend_ar1(protocol["trend_ar1"], tolerance, study1_protocol, study1),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/results-v2-followup")
    parser.add_argument("--force", action="store_true", help="Replace an existing follow-up record")
    args = parser.parse_args()
    frozen = {(ROOT / "experiments" / name).resolve() for name in ("results", "results-v2")}
    if args.output.resolve() in frozen:
        parser.error("The frozen study directories are protected; choose a separate output")
    protect_output(args.output, force=args.force)
    protocol = json.loads(PROTOCOL.read_text())
    validate_protocol(protocol)
    fingerprints = source_fingerprints()
    with threadpool_limits(limits=1):
        results = run(protocol)
    if fingerprints != source_fingerprints():
        raise RuntimeError("Source files changed during the follow-up; rerun it")
    args.output.mkdir(parents=True, exist_ok=True)
    numerical = args.output / "results.json"
    numerical.write_text(json.dumps(results, indent=2, sort_keys=True, allow_nan=False) + "\n")
    manifest = {"protocol": protocol, "protocol_sha256": sha256(PROTOCOL), "results_sha256": sha256(numerical),
                "frozen_inputs_sha256": {item["path"]: sha256(ROOT / item["path"])
                                         for item in protocol["frozen_inputs"].values()},
                "source_files": fingerprints, "python": platform.python_version(), "platform": platform.platform(),
                "package_version": timegym.__version__,
                "versions": {name: importlib.metadata.version(name)
                             for name in ("numpy", "pandas", "scipy", "scikit-learn")},
                "thread_limit": 1,
                "source_note": "Hashes identify the working source, including generated package files. "
                               "The follow-up was chosen after the protocol 2.0 results were known and is exploratory."}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "results_sha256": manifest["results_sha256"],
                      "answer_key_totals": results["answer_key_alarm"]["totals"]}, indent=2))


if __name__ == "__main__":
    main()
