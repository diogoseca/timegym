"""Run the frozen CPU pilot: python -m experiments.run_study.

Use --output /tmp/timegym-repeat for a verification rerun without overwriting
the canonical artifacts. Numerical results exclude timestamps and runtime.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import os
import subprocess

import numpy as np
import pandas as pd
from timegym.statistics import paired_t_interval as mean_interval
from timegym.statistics import wilson_interval as rate_interval

from timegym import Forecast, Pipeline, Scenario, NaiveForecaster, GaussianAR1Forecaster
from timegym.checks import check_shift_equivariance
from timegym.diagnostics import covariance_dependence_check, evaluate_forecast
from timegym.metrics import mse, mae, crps, energy_score, variogram_score
from timegym.mutations import MUTATIONS, mutate_forecast
from threadpoolctl import threadpool_info, threadpool_limits

from .common import protect_output, sha256 as fingerprint

ROOT = Path(__file__).resolve().parents[1]


class StandardizedMeanForecaster:
    """One explicit bug/repair pair: forgetting to invert target scaling."""
    def __init__(self, inverse=True):
        self.inverse = inverse

    def clone_unfitted(self):
        return type(self)(inverse=self.inverse)

    def fit(self, context):
        self.names_ = tuple(context.columns)
        y = context.to_numpy(dtype=float)
        self.location_ = y.mean(axis=0)
        self.scale_ = y.std(axis=0)
        self.scale_ = np.where(self.scale_ == 0, 1., self.scale_)
        self.standard_mean_ = ((y - self.location_) / self.scale_).mean(axis=0)
        return self

    def predict(self, future_index):
        value = self.standard_mean_
        if self.inverse:
            value = value * self.scale_ + self.location_
        return Forecast(future_index.copy(), point=np.tile(value, (len(future_index), 1)),
                        target_names=self.names_)


def source_fingerprints():
    paths = sorted(list((ROOT / "timegym").rglob("*.py")) + list((ROOT / "experiments").glob("*.py")))
    paths += [ROOT / "experiments/protocol.json", ROOT / "experiments/data/AirPassengers.csv"]
    return {str(path.relative_to(ROOT)): fingerprint(path) for path in paths}


def capability_models(task):
    return {"last_value": lambda: NaiveForecaster(),
            "seasonal_12": lambda: NaiveForecaster("seasonal", task["seasonal_period"]),
            "gaussian_ar1": lambda: GaussianAR1Forecaster(n_samples=task["n_samples"])}


def validate_protocol(protocol):
    if protocol.get("protocol_version") != "1.1":
        raise ValueError("Expected pilot protocol 1.1")
    for name in ("development_seeds", "evaluation_seeds", "capability_seeds"):
        bounds = protocol[name]
        if (len(bounds) != 2 or any(type(value) is not int or value < 0 for value in bounds)
                or bounds[1] - bounds[0] < 2):
            raise ValueError(f"{name} must be a half-open range of at least two nonnegative seeds")
    if set(range(*protocol["development_seeds"])) & set(range(*protocol["evaluation_seeds"])):
        raise ValueError("Development and evaluation seed ranges overlap")
    if not 0 < protocol["alpha_per_check"] < 1:
        raise ValueError("alpha_per_check must be in (0, 1)")
    if protocol["n_samples"] < 20 or protocol["n_calibration"] < 1:
        raise ValueError("Covariance checks require at least 20 samples and positive calibration count")
    mutations = protocol["mutations"]
    if not mutations or len(set(mutations)) != len(mutations) or set(mutations) - set(MUTATIONS):
        raise ValueError("Mutations must be distinct implemented fixtures")
    task = protocol["capability_task"]
    for key in ("context_length", "horizon", "n_targets", "n_samples", "seasonal_period"):
        if type(task[key]) is not int or task[key] < 1:
            raise ValueError(f"capability_task.{key} must be a positive integer")
    names = protocol["capability_models"]
    if not names or len(set(names)) != len(names) or set(names) - capability_models(task).keys():
        raise ValueError("capability_models must name distinct implemented models")
    for name in ("scenario", "inverse_transform_synthetic"):
        for dimension in ("context_length", "horizon", "n_targets"):
            if dimension not in protocol[name]:
                raise ValueError(f"{name}.{dimension} must be explicit")
    Scenario(**protocol["scenario"])
    spec = protocol["inverse_transform_synthetic"]
    Scenario(**{key: value for key, value in spec.items() if key != "shift"})
    for kind in protocol["capability_scenarios"]:
        for sigma in protocol["capability_sigma"]:
            Scenario(kind, parameters={"sigma": sigma}, **{key: task[key] for key in ("context_length", "horizon", "n_targets")})
    strengths = protocol["dependence_strengths"]
    if not strengths or any(not 0 <= strength <= 1 for strength in strengths):
        raise ValueError("dependence_strengths must lie in [0, 1]")


def provenance(protocol, fingerprints=None):
    paths = sorted(list((ROOT / "timegym").rglob("*.py")) + list((ROOT / "experiments").glob("*.py")))
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = None
    return {"protocol_sha256": fingerprint(ROOT / "experiments/protocol.json"),
            "source_sha256": digest.hexdigest(), "source_files": fingerprints or source_fingerprints(), "base_git_commit": commit,
            "source_note": "Source fingerprint identifies working files; base commit alone does not include uncommitted v2 changes.",
            "python": platform.python_version(), "platform": platform.platform(),
            "versions": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scipy")},
            "dataset_sha256": fingerprint(ROOT / "experiments/data/AirPassengers.csv"), "protocol": protocol}


def run(protocol):
    scenario = Scenario(**protocol["scenario"])
    seeds = range(*protocol["evaluation_seeds"])
    development = range(*protocol["development_seeds"])
    sample_count = protocol["n_samples"]
    threshold = float(np.quantile([mse((c := scenario.generate(seed=s)).observed.to_numpy(), c.reference.mean)
                                   for s in development], .95))
    records, curves, examples = [], [], {}
    for seed in seeds:
        case = scenario.generate(seed=seed)
        clean = case.reference.forecast(n_samples=sample_count, seed=seed+protocol["seed_offsets"]["reference_samples"])
        observed = case.observed.to_numpy()
        checks = {"seed": seed+protocol["seed_offsets"]["calibration"], "n_calibration": protocol["n_calibration"], "alpha": protocol["alpha_per_check"]}
        clean_covariance = covariance_dependence_check(case.reference, clean.samples, **checks)
        for name in ("clean",) + tuple(protocol["mutations"]):
            forecast = clean if name == "clean" else mutate_forecast(clean, name, seed=seed+protocol["seed_offsets"]["mutation"])
            violations = forecast.validate(case.task)
            try:
                point_mse = mse(observed, forecast.point)
                mse_alarm = point_mse > threshold
            except (ValueError, TypeError):
                point_mse, mse_alarm = None, True
            if violations:
                covariance_status, covariance_evidence = "UNSUPPORTED", {}
            else:
                if name not in {"variance_shrink", "dependence_shuffle"}:
                    assert np.array_equal(forecast.samples, clean.samples), "Cannot reuse a covariance decision for different samples"
                    diagnostic = clean_covariance
                else:
                    diagnostic = covariance_dependence_check(case.reference, forecast.samples, **checks)
                covariance_status, covariance_evidence = diagnostic.status, diagnostic.evidence
            if name == "clean":
                repair_contract, repair_covariance = False, clean_covariance.status
            else:
                repaired = mutate_forecast(clean, name, enabled=False)
                repair_contract = bool(repaired.validate(case.task))
                # Same exact ensemble, so re-use its paired clean covariance decision.
                assert np.array_equal(repaired.samples, clean.samples)
                repair_covariance = clean_covariance.status
            records.append({"seed": seed, "mutation": name, "contract_alarm": bool(violations),
                            "covariance_status": covariance_status, "covariance_evidence": covariance_evidence,
                            "mse_alarm": bool(mse_alarm), "point_mse": point_mse,
                            "repair_contract_alarm": repair_contract, "repair_covariance_status": repair_covariance,
                            "violations": violations, "manifest": case.manifest})
            if seed == seeds.start and name in {"index_shift", "dependence_shuffle"}:
                report = evaluate_forecast(case, forecast, check_dependence=not violations,
                                           dependence_seed=seed+protocol["seed_offsets"]["calibration"],
                                           n_calibration=protocol["n_calibration"],
                                           alpha=protocol["alpha_per_check"])
                examples[name] = report.to_dict()
        clean_scores = {"crps": crps(observed, clean.samples), "energy": energy_score(observed, clean.samples),
                        "variogram": variogram_score(observed, clean.samples)}
        for strength in protocol["dependence_strengths"]:
            corrupted = mutate_forecast(clean, "dependence_shuffle", seed=seed+protocol["seed_offsets"]["mutation"], strength=strength)
            assert np.array_equal(np.sort(corrupted.samples, axis=0), np.sort(clean.samples, axis=0))
            scores = {"crps": crps(observed, corrupted.samples), "energy": energy_score(observed, corrupted.samples),
                      "variogram": variogram_score(observed, corrupted.samples)}
            curves.append({"seed": seed, "strength": strength, **scores,
                           **{key+"_delta": scores[key]-clean_scores[key] for key in scores}})
    summary = {}
    for name in ("clean",) + tuple(protocol["mutations"]):
        rows = [r for r in records if r["mutation"] == name]
        eligible = [r for r in rows if r["covariance_status"] in {"PASS", "FAIL"}]
        summary[name] = {"contract": rate_interval([r["contract_alarm"] for r in rows]),
                         "mse": rate_interval([r["mse_alarm"] for r in rows]),
                         "covariance": rate_interval([r["covariance_status"] == "FAIL" for r in eligible]) if eligible else None,
                         "combined": rate_interval([r["contract_alarm"] or r["covariance_status"] == "FAIL" for r in rows]),
                         "repair_combined": rate_interval([r["repair_contract_alarm"] or r["repair_covariance_status"] == "FAIL" for r in rows])}
    curve_summary = {str(s): {key: mean_interval([r[key+"_delta"] for r in curves if r["strength"] == s])
                             for key in ("crps", "energy", "variogram")} for s in protocol["dependence_strengths"]}
    return {"mse_threshold": threshold, "fault_summary": summary, "fault_records": records,
            "dependence_summary": curve_summary, "dependence_records": curves, "example_reports": examples}


def capability_study(protocol):
    rows = []
    task = protocol["capability_task"]
    constructors = capability_models(task)
    for kind in protocol["capability_scenarios"]:
        for sigma in protocol["capability_sigma"]:
            scenario = Scenario(kind, parameters={"sigma": sigma},
                                **{key: task[key] for key in ("context_length", "horizon", "n_targets")})
            for name in protocol["capability_models"]:
                pipeline = Pipeline(constructors[name]())
                for seed in range(*protocol["capability_seeds"]):
                    case = scenario.generate(seed=seed)
                    forecast = pipeline.predict(case.task.context, case.task.future_index)
                    rows.append({"scenario": kind, "sigma": sigma, "model": name, "seed": seed,
                                 "mean_reference_mse": mse(case.reference.mean, forecast.point)})
    return rows


def inverse_repair_study(protocol):
    defective = Pipeline(StandardizedMeanForecaster(inverse=False))
    repaired = Pipeline(StandardizedMeanForecaster(inverse=True))
    rows = []
    spec = protocol["inverse_transform_synthetic"]
    scenario = Scenario(**{key: value for key, value in spec.items() if key != "shift"})
    for seed in range(*protocol["evaluation_seeds"]):
        case = scenario.generate(seed=seed)
        checks = [check_shift_equivariance(p, case.task.context, case.task.future_index, shift=spec["shift"]) for p in (defective, repaired)]
        scores = [mae(case.observed.to_numpy(), p.predict(case.task.context, case.task.future_index).point) for p in (defective, repaired)]
        rows.append({"seed": seed, "faulty_status": checks[0].status, "repaired_status": checks[1].status,
                     "faulty_mae": scores[0], "repaired_mae": scores[1]})
    data = pd.read_csv(ROOT / "experiments/data/AirPassengers.csv")
    y = pd.DataFrame({"passengers": data["value"].to_numpy(dtype=float)}, index=pd.period_range("1949-01", periods=len(data), freq="M"))
    real_rows = []
    for origin in protocol["real_data"]["origins"]:
        context, heldout = y.iloc[:origin], y.iloc[origin:origin+protocol["real_data"]["horizon"]]
        checks = [check_shift_equivariance(p, context, heldout.index) for p in (defective, repaired)]
        scores = [mae(heldout.to_numpy(), p.predict(context, heldout.index).point) for p in (defective, repaired)]
        real_rows.append({"origin": str(context.index[-1]), "faulty_status": checks[0].status,
                          "repaired_status": checks[1].status, "faulty_mae": scores[0], "repaired_mae": scores[1]})
    return {"synthetic": rows, "real": real_rows,
            "synthetic_fault_detection": rate_interval([r["faulty_status"] == "FAIL" for r in rows]),
            "synthetic_repair_failure": rate_interval([r["repaired_status"] == "FAIL" for r in rows]),
            "synthetic_paired_mae_delta": mean_interval([r["repaired_mae"]-r["faulty_mae"] for r in rows])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/results")
    parser.add_argument("--force", action="store_true", help="Replace existing study artifacts")
    args = parser.parse_args()
    protect_output(args.output, force=args.force)
    protocol = json.loads((ROOT / "experiments/protocol.json").read_text())
    validate_protocol(protocol)
    fingerprints = source_fingerprints()
    with threadpool_limits(limits=1):
        results = run(protocol)
        results["capability_records"] = capability_study(protocol)
        results["inverse_repair"] = inverse_repair_study(protocol)
        runtime = {"threads": {name: os.environ.get(name) for name in
                               ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
                   "threadpools": threadpool_info(), "thread_limit": 1}
    manifest = provenance(protocol, fingerprints)
    if fingerprints != source_fingerprints():
        raise RuntimeError("Source files changed during the pilot; no artifacts were written")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(json.dumps(results, indent=2, sort_keys=True, allow_nan=False) + "\n")
    manifest["results_sha256"] = fingerprint(args.output / "results.json")
    (args.output / "runtime.json").write_text(json.dumps(runtime, indent=2, sort_keys=True)+"\n")
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
    for name, report in results["example_reports"].items():
        (args.output / f"example-{name}.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "fault_summary": results["fault_summary"],
                      "dependence": results["dependence_summary"]["1.0"], "inverse_repair": results["inverse_repair"]["real"]}, indent=2))


if __name__ == "__main__":
    main()
