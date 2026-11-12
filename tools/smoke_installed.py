"""Exercise the installed wheel, with lazily imported libraries deliberately unavailable.

Matplotlib and scikit-learn are installed with TimeGym but imported only when a
plot is drawn or an sklearn object is used, so scoring must work without them.

Run with `python -I tools/smoke_installed.py` after installing a built wheel.
--report PATH also saves the replayed report for replay in a separate process.
For an offline local check only, --dependency-path can expose scientific
dependencies from another environment without executing its editable .pth hooks.
"""
from __future__ import annotations

import argparse
import importlib.abc
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dependency-path", type=Path)
    parser.add_argument("--report", type=Path, help="Also save the replayed report here")
    args = parser.parse_args()
    if args.dependency_path:
        sys.path.append(str(args.dependency_path.resolve()))
    blocked = {"matplotlib", "sklearn", "nbdev", "nbformat", "nbclient", "nbconvert",
               "IPython", "tsfel", "yfinance", "jupyterlab"}
    attempts = []

    class BlockOptional(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.partition(".")[0] in blocked:
                attempts.append(fullname)
                raise ImportError(f"Optional dependency deliberately unavailable: {fullname}")

    sys.meta_path.insert(0, BlockOptional())
    import numpy as np
    import timegym
    from timegym import (GaussianAR1Forecaster, LastValueForecaster, NaiveForecaster, Pipeline,
                         Scenario, Scorecard, TestReport)

    installed_path = Path(timegym.__file__).resolve()
    repository = Path(__file__).resolve().parents[1]
    if installed_path.is_relative_to(repository / "timegym"):
        raise AssertionError("The smoke check imported the checkout instead of an installed wheel")
    if not installed_path.is_relative_to(Path(sys.prefix).resolve()):
        raise AssertionError("TimeGym must be installed into the checking interpreter's environment")
    scenario = Scenario("ar1", context_length=48, horizon=6, n_targets=2,
                        parameters={"correlation": .5})
    case = scenario.generate(seed=17)
    point = Pipeline(NaiveForecaster()).predict(case.task.context, case.task.future_index)
    assert point.point.shape == (6, 2) and point.validate(case.task) == []
    joint = Pipeline(GaussianAR1Forecaster(n_samples=32, seed=11)).predict(
        case.task.context, case.task.future_index)
    assert joint.samples.shape == (32, 6, 2) and joint.validate(case.task) == []
    assert np.isfinite(joint.samples).all()
    run = Pipeline(NaiveForecaster()).run([scenario], seeds=[0, 1, 2])
    run.report.assert_passed()
    assert len(run.cases) == 3
    from timegym.diagnostics import covariance_dependence_check, evaluate_forecast
    from timegym.mutations import mutate_forecast
    shifted = evaluate_forecast(case, mutate_forecast(point, "index_shift"))
    fault_codes = [d.evidence.get("code") for d in shifted.diagnostics if d.failed]
    assert "index_mismatch" in fault_codes and shifted.metrics == {}
    with TemporaryDirectory(prefix="timegym-wheel-report-") as temporary:
        report_path = args.report or Path(temporary) / "report.json"
        run.report.to_json(report_path)
        replay = TestReport.read(report_path).replay_case(0)
        assert replay.metadata["replay"]["reproduced"]
    joint_processes = []
    for kind in ("composite", "merton_jump_diffusion", "garch", "markov_switching"):
        spec = Scenario(kind, context_length=24, horizon=3, n_targets=2,
                        parameters={"correlation": .6})
        generated = spec.generate(seed=17)
        distribution = generated.reference.forecast(n_samples=64, seed=91)
        assert distribution.samples.shape == (64, 3, 2)
        assert distribution.validate(generated.task) == []
        retained = Pipeline(NaiveForecaster()).run([spec], seeds=[17])
        rechecked = TestReport.from_json(retained.report.to_json()).replay_case()
        assert rechecked.metadata["replay"]["reproduced"]
        if kind == "garch":
            assert generated.reference.moments_kind == "analytic_mean_monte_carlo_covariance"
            assert generated.reference.sampler_kind == "exact_conditional"
        joint_processes.append(kind)
    approximate = Scenario("markov_switching", 24, 3, n_targets=2,
                           parameters={"regime_jump_rate": [1., 4.]}).generate(seed=17)
    guard = covariance_dependence_check(approximate.reference,
                                       approximate.reference.sample(64, seed=91))
    assert guard.status == "UNSUPPORTED" and "p_value" not in guard.evidence
    card = Pipeline(LastValueForecaster()).scorecard(["random_walk", "seasonal"], instances=2)
    assert isinstance(card, Scorecard) and len(card.table) == 4
    assert str(card).startswith("## LastValueForecaster: ")
    assert "Random walk" in str(card) and all(row["failed"] == 0 for row in card.checks)
    assert not attempts, attempts
    assert not (set(sys.modules) & blocked)
    print(json.dumps({"imported_from": str(installed_path), "version": timegym.__version__,
                      "point_shape": list(point.point.shape),
                      "joint_sample_shape": list(joint.samples.shape),
                      "retained_cases": len(run.cases), "index_shift_fault_codes": fault_codes,
                      "joint_processes_verified": joint_processes,
                      "scorecard_score": round(card.score, 1),
                      "optional_import_attempts": attempts}, indent=2))


if __name__ == "__main__":
    main()
