"""Replay one controlled fault: python -m experiments.reproduce_fault --mutation index_shift --seed 10000."""
import argparse
import json
from pathlib import Path

from timegym import Scenario
from timegym.diagnostics import evaluate_forecast
from timegym.mutations import mutate_forecast

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mutation", default="index_shift")
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "experiments/results")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    results = json.loads((args.results_dir / "results.json").read_text())
    provenance = json.loads((args.results_dir / "manifest.json").read_text())
    protocol = provenance["protocol"]
    rows = [r for r in results["fault_records"] if r["mutation"] == args.mutation and r["seed"] == args.seed]
    if len(rows) != 1:
        parser.error("Requested mutation and seed must identify one saved case")
    recorded = rows[0]
    keys = ("kind", "context_length", "horizon", "parameters", "n_targets", "seed")
    case = Scenario(**{key: recorded["manifest"][key] for key in keys}).generate()
    clean = case.reference.forecast(n_samples=protocol["n_samples"],
                                    seed=args.seed + protocol["seed_offsets"]["reference_samples"])
    forecast = clean if args.mutation == "clean" else mutate_forecast(
        clean, args.mutation, seed=args.seed + protocol["seed_offsets"]["mutation"])
    violations = forecast.validate(case.task)
    result = evaluate_forecast(case, forecast, check_dependence=not violations,
                               dependence_seed=args.seed + protocol["seed_offsets"]["calibration"],
                               n_calibration=protocol["n_calibration"], alpha=protocol["alpha_per_check"])
    covariance = [d for d in result.diagnostics if d.name == "reference_covariance"]
    covariance_status = covariance[0].status if covariance else "UNSUPPORTED"
    matches = violations == recorded["violations"] and covariance_status == recorded["covariance_status"]
    payload = {"recorded_decisions_match": matches, "mutation": args.mutation, "case": result.to_dict()}
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
    if args.output:
        args.output.write_text(serialized + "\n")
    print(serialized)
    return 0 if matches else 1


if __name__ == "__main__":
    raise SystemExit(main())
