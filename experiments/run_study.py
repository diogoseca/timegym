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
import subprocess

import numpy as np
import pandas as pd
from scipy.stats import norm, t

from timegym import Forecast, Pipeline, Scenario, NaiveForecaster, GaussianAR1Forecaster
from timegym.checks import check_shift_equivariance
from timegym.diagnostics import covariance_dependence_check, evaluate_forecast
from timegym.metrics import mse, mae, crps, energy_score, variogram_score
from timegym.mutations import MUTATIONS, mutate_forecast

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


def rate_interval(values):
    n, k = len(values), int(sum(values))
    z = norm.ppf(.975)
    p = k / n
    center = (p + z*z/(2*n)) / (1 + z*z/n)
    half = z * np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return {"count": k, "n": n, "rate": p, "ci95": [float(max(0, center-half)), float(min(1, center+half))]}


def mean_interval(values):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    half = float(t.ppf(.975, len(values)-1) * values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.
    return {"mean": mean, "ci95": [mean-half, mean+half], "n": len(values)}


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def provenance(protocol):
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
            "source_sha256": digest.hexdigest(), "base_git_commit": commit,
            "source_note": "Source fingerprint identifies working files; base commit alone does not include uncommitted v2 changes.",
            "python": platform.python_version(), "platform": platform.platform(),
            "versions": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn")},
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
                diagnostic = clean_covariance if name not in {"variance_shrink", "dependence_shuffle"} else covariance_dependence_check(case.reference, forecast.samples, **checks)
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
    for kind in protocol["capability_scenarios"]:
        for sigma in protocol["capability_sigma"]:
            scenario = Scenario(kind, parameters={"sigma": sigma}, context_length=120, horizon=8)
            for name, model in [("last_value", NaiveForecaster()), ("seasonal_12", NaiveForecaster("seasonal", 12)),
                                ("gaussian_ar1", GaussianAR1Forecaster(n_samples=128))]:
                pipeline = Pipeline(model)
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
    scenario = Scenario(spec["kind"], parameters=spec["parameters"])
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
    args = parser.parse_args()
    protocol = json.loads((ROOT / "experiments/protocol.json").read_text())
    results = run(protocol)
    results["capability_records"] = capability_study(protocol)
    results["inverse_repair"] = inverse_repair_study(protocol)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(json.dumps(results, indent=2, sort_keys=True, allow_nan=False) + "\n")
    (args.output / "manifest.json").write_text(json.dumps(provenance(protocol), indent=2, sort_keys=True, allow_nan=False) + "\n")
    for name, report in results["example_reports"].items():
        (args.output / f"example-{name}.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "fault_summary": results["fault_summary"],
                      "dependence": results["dependence_summary"]["1.0"], "inverse_repair": results["inverse_repair"]["real"]}, indent=2))


if __name__ == "__main__":
    main()
