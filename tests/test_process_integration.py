"""Controlled process families work through the public evaluation and replay API."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from timegym import Pipeline, Scenario, TestReport
from timegym.diagnostics import covariance_dependence_check, evaluate_forecast
from timegym.scenarios import scenario_label


PROCESS_CASES = [
    ("composite", {"slope": 0.03, "amplitude": 2.0, "period": 12.0,
                   "change_point": 20, "changes": {"sigma": 2.0}}),
    ("merton_jump_diffusion", {"output": "price", "dt": 0.1}),
    ("garch", {"output": "log_price"}),
    ("markov_switching", {"output": "log_return", "dt": 0.1}),
]


@pytest.mark.parametrize("kind,parameters", PROCESS_CASES)
def test_public_pipeline_report_round_trip_and_replay_each_new_family(kind, parameters):
    scenario = Scenario(kind, context_length=24, horizon=3, parameters=parameters)
    pipeline = Pipeline()
    run = pipeline.run([scenario], seeds=[3, 5])
    assert len(run.cases) == 2
    for entry in run.cases:
        assert entry.result.scenario == entry.case.manifest["kind"] == kind
        assert entry.forecast.validate(entry.case.task) == []
        assert {"mse", "mae", "mase", "mean_reference_mse"} <= entry.result.metrics.keys()
        assert all(np.isfinite(value) for value in entry.result.metrics.values())
        assert not any(d.status == "ERROR" for d in entry.result.diagnostics)
        assert scenario.display_name == scenario_label(kind, entry.case.manifest["parameters"])
        assert entry.case.reference.conditioning["future_observations_used"] is False
    direct_report = pipeline.test([scenario], seeds=[3, 5])
    assert direct_report.to_dict() == run.report.to_dict()
    encoded = run.report.to_json()
    assert json.loads(encoded)["schema_version"] == "2.0"
    loaded = TestReport.from_json(encoded)
    assert loaded.to_dict() == run.report.to_dict()
    replay = loaded.replay_case(1)
    assert replay.metadata["replay"]["metrics_match"]
    assert replay.metadata["replay"]["diagnostics_match"]
    assert replay.metadata["replay"]["manifest_match"]
    assert replay.metadata["replay"]["source_matches"]
    assert replay.metadata["replay"]["dependencies_match"]


def jump_regimes():
    return Scenario("markov_switching", context_length=24, horizon=3, parameters={
        "dt": 0.1, "regime_jump_rate": [0.2, 1.0],
        "regime_jump_mean": [-0.02, -0.1], "regime_jump_sigma": [0.02, 0.1],
    })


def test_composed_regime_jump_report_preserves_approximation_and_quantile_settings():
    run = Pipeline().run([jump_regimes()], seeds=[3])
    generated = run.cases[0].case
    metadata = generated.manifest["conditioning"]
    assert metadata["sampler_kind"] == "approximate_conditional"
    assert metadata["moments_kind"] == "analytic"
    assert metadata["quantiles_kind"] == "monte_carlo"
    assert metadata["quantile_samples"] == generated.reference.quantile_samples
    assert metadata["quantile_seed"] == generated.reference.quantile_seed
    assert metadata["quantile_method"] == "linear"
    assert metadata["state_inference"] == "forward_filter_observed_returns_only"
    assert "not_a_posterior_error_bound" in metadata["filter_note"]
    assert metadata["hidden_states_exposed_to_model"] is False
    assert not hasattr(generated.task, "state_posterior")
    restored = TestReport.from_json(run.report.to_json())
    assert restored.cases[0].manifest["conditioning"] == metadata
    replay = restored.replay_case()
    assert replay.metadata["replay"]["manifest_match"]
    forecast = generated.reference.forecast(n_samples=32, seed=93)
    result = evaluate_forecast(generated, forecast, check_dependence=True,
                               dependence_seed=101, n_calibration=19)
    diagnostic = next(item for item in result.diagnostics if item.name == "reference_covariance")
    assert diagnostic.status == "UNSUPPORTED"
    assert diagnostic.evidence["reference"]["sampler_kind"] == "approximate_conditional"
    assert "p_value" not in diagnostic.evidence
    # Unsupported null calibration does not suppress ordinary predictive scores.
    assert {"mse", "pinball", "crps", "energy_score", "variogram_score"} <= result.metrics.keys()
    assert all(np.isfinite(value) for value in result.metrics.values())


def test_exact_merton_reference_exposes_a_calibrated_covariance_diagnostic():
    generated = Scenario("merton_jump_diffusion", context_length=24, horizon=3,
                         parameters={"output": "log_return", "dt": 0.1}).generate(seed=3)
    forecast = generated.reference.forecast(n_samples=64, seed=93)
    result = evaluate_forecast(generated, forecast, check_dependence=True,
                               dependence_seed=101, n_calibration=19)
    diagnostic = next(item for item in result.diagnostics if item.name == "reference_covariance")
    # A valid null may reject at its declared error rate; test availability,
    # finite evidence and reproducibility rather than requiring a lucky PASS.
    assert diagnostic.status in {"PASS", "FAIL"}
    assert 1 / 20 <= diagnostic.evidence["p_value"] <= 1
    assert np.isfinite(diagnostic.evidence["statistic"])
    assert diagnostic.evidence["reference"]["sampler_kind"] == "exact_conditional"
    repeated = covariance_dependence_check(generated.reference, forecast.samples,
                                           seed=101, n_calibration=19)
    assert repeated.to_dict() == diagnostic.to_dict()


@pytest.mark.parametrize("kind,output,unit", [
    ("merton_jump_diffusion", "price", "price"),
    ("merton_jump_diffusion", "log_price", "log price"),
    ("merton_jump_diffusion", "log_return", "log returns"),
    ("garch", "log_price", "log price"),
    ("garch", "log_return", "log returns"),
    ("markov_switching", "price", "price"),
])
def test_public_scenario_labels_carry_output_units(kind, output, unit):
    specification = Scenario(kind, context_length=12, horizon=2, parameters={"output": output})
    generated = specification.generate(seed=3)
    expected = scenario_label(kind, generated.manifest["parameters"])
    assert specification.display_name == expected
    assert expected.endswith(f"({unit})")
    assert generated.manifest["kind"] == kind


def test_controlled_generation_and_native_pipeline_do_not_import_optional_packages():
    script = '''
import importlib.abc
import sys

blocked = {"matplotlib", "sklearn", "IPython", "nbdev", "yfinance", "tsfel"}
class CoreOnly(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in blocked:
            raise AssertionError("optional dependency imported: " + fullname)
sys.meta_path.insert(0, CoreOnly())
from timegym import Pipeline, Scenario
for kind in ("composite", "merton_jump_diffusion", "garch", "markov_switching"):
    run = Pipeline().run([Scenario(kind, context_length=12, horizon=2)], seeds=[3])
    assert len(run.cases) == 1
    assert "mse" in run.cases[0].result.metrics
    assert not any(d.status == "ERROR" for d in run.cases[0].result.diagnostics)
assert not any(name.split(".")[0] in blocked for name in sys.modules)
'''
    completed = subprocess.run([sys.executable, "-c", script],
                               cwd=Path(__file__).resolve().parents[1],
                               text=True, capture_output=True, timeout=30)
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_plot_legend_distinguishes_approximate_filtered_mean_from_exact_reference():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run = Pipeline().run([
        jump_regimes(),
        Scenario("merton_jump_diffusion", context_length=24, horizon=3),
    ], seeds=[3])
    figure = run.plot_forecasts()
    try:
        approximate_labels = figure.axes[0].get_legend_handles_labels()[1]
        exact_labels = figure.axes[1].get_legend_handles_labels()[1]
        assert "Conditional mean (approx. filter)" in approximate_labels
        assert "Known conditional mean" not in approximate_labels
        assert "Known conditional mean" in exact_labels
        for axis in figure.axes:
            predictions = next(line for line in axis.lines if line.get_label() == "Predictive mean")
            assert predictions.get_linestyle() == "None"
    finally:
        plt.close(figure)
