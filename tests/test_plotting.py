"""Behavioral checks for retained runs and the public plotting protocol."""

import os
from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from timegym import Forecast, NaiveForecaster, Pipeline, Scenario
from timegym.evaluation import ForecastRun


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_run_fits_once_per_case_and_plots_retained_outputs_without_refitting():
    models = []

    class CountingModel:
        def __init__(self):
            self.fit_count = 0
            self.predict_count = 0
            models.append(self)

        def fit(self, context):
            self.fit_count += 1
            self.names = tuple(context.columns)
            self.last = context.iloc[-1].to_numpy()
            return self

        def predict(self, future_index):
            self.predict_count += 1
            return Forecast(future_index, point=np.tile(self.last, (len(future_index), 1)), target_names=self.names)

    run = Pipeline(model_factory=CountingModel).run(
        [Scenario("trend", context_length=12, horizon=3), Scenario("ar1", context_length=12, horizon=3)], seeds=[5, 7])
    assert isinstance(run, ForecastRun)
    assert len(models) == len(run.cases) == 4
    assert all(model.fit_count == model.predict_count == 1 for model in models)
    assert len({id(model) for model in models}) == 4
    assert run.cases[0].forecast.index.equals(run.cases[0].case.task.future_index)
    assert run.report.cases[0] is run.cases[0].result
    run.plot_forecasts()
    run.plot_errors()
    run.plot_forecasts()
    assert len(models) == 4
    assert all(model.fit_count == model.predict_count == 1 for model in models)


def test_test_schema_and_replay_remain_identical_to_run_report():
    pipeline = Pipeline(NaiveForecaster("mean"))
    scenarios = [Scenario("white_noise", context_length=12, horizon=3)]
    run = pipeline.run(scenarios, seeds=[2])
    report = pipeline.test(scenarios, seeds=[2])
    assert run.report.to_dict() == report.to_dict()
    assert run.report.to_dict()["schema_version"] == "2.0"
    assert run.report.replay_case().metadata["replay"]["metrics_match"]
    assert "forecast" not in run.report.to_dict()["cases"][0]


def test_forecasts_are_unconnected_blue_crosses_and_observations_gray_dots():
    run = Pipeline().run([Scenario("trend", context_length=24, horizon=9)], seeds=[8])
    figure = run.plot_forecasts()
    axis = figure.axes[0]
    lines = {line.get_label(): line for line in axis.lines}
    point = lines["Predictive mean"]
    assert point.get_marker() == "x"
    assert point.get_linestyle() == "None"
    assert point.get_color() == "blue"
    assert len(point.get_xdata()) == 9
    for name in ["Observed context", "Held-out observations"]:
        assert lines[name].get_marker() == "."
        assert lines[name].get_linestyle() == "None"
        assert lines[name].get_color() == "gray"
    reference = lines["Known conditional mean"]
    np.testing.assert_array_equal(reference.get_xdata(), np.arange(24, 33))
    assert lines["Forecast origin"].get_linestyle() == "--"
    np.testing.assert_allclose(lines["Forecast origin"].get_xdata(), [23.5, 23.5])
    np.testing.assert_allclose(axis.get_xlim(), [0, 32])
    assert "vs what happened " in axis.get_title(loc="left")
    assert "error vs answer key " in axis.get_title(loc="left")
    assert "/100" in axis.get_title(loc="left")


def test_multitarget_single_seed_and_duplicate_scenario_kinds_have_distinct_rows():
    run = Pipeline().run([
        Scenario("ar1", context_length=12, horizon=3, n_targets=2),
        Scenario("ar1", context_length=20, horizon=4, n_targets=1),
    ], seeds=[5])
    figure = run.plot_forecasts()
    assert len(figure.axes) == 3
    assert [axis.get_ylabel() for axis in figure.axes] == ["y0", "y1", "y"]
    assert figure.axes[0].get_title(loc="left").startswith("1. Stationary AR(1)")
    assert figure.axes[2].get_title(loc="left").startswith("2. Stationary AR(1)")
    error_figure = run.plot_errors()
    assert len(error_figure.axes) == 2


def test_two_seed_columns_preserve_duplicate_seed_positions():
    run = Pipeline().run([Scenario("white_noise", context_length=10, horizon=3, n_targets=2)], seeds=[5, 5])
    figure = run.plot_forecasts()
    assert len(figure.axes) == 4
    assert [axis.get_ylabel() for axis in figure.axes] == ["y0", "y0", "y1", "y1"]


@pytest.mark.parametrize("index", [
    pd.date_range("2026-01-01", periods=10, freq="D", tz="Europe/Lisbon"),
    pd.period_range("2026-01", periods=10, freq="M"),
    pd.timedelta_range("0h", periods=10, freq="2h"),
    pd.RangeIndex(10, 30, 2),
])
def test_date_period_timedelta_and_numeric_axes_preserve_bounds(index):
    import matplotlib.dates as mdates

    class IndexedScenario(Scenario):
        def generate(self, seed=None):
            case = super().generate(seed)
            case.task.context.index = index[:7]
            case.task.future_index = index[7:]
            case.observed.index = index[7:]
            case.reference.index = index[7:]
            return case

    run = Pipeline().run([IndexedScenario("white_noise", context_length=7, horizon=3)], seeds=[2])
    figure = run.plot_forecasts()
    figure.canvas.draw()
    axis = figure.axes[0]
    if isinstance(index, pd.PeriodIndex):
        expected = mdates.date2num(index.to_timestamp().to_pydatetime())
    elif isinstance(index, pd.DatetimeIndex):
        expected = mdates.date2num(index.to_pydatetime())
    elif isinstance(index, pd.TimedeltaIndex):
        expected = index.total_seconds().to_numpy()
    else:
        expected = index.to_numpy()
    np.testing.assert_allclose(axis.get_xlim(), [expected[0], expected[-1]])
    boundary = next(line for line in axis.lines if line.get_label() == "Forecast origin")
    np.testing.assert_allclose(boundary.get_xdata(), [(expected[6] + expected[7]) / 2] * 2)


def test_execution_errors_are_retained_and_annotated():
    class Broken:
        def fit(self, context):
            raise RuntimeError("intentional failure")

    run = Pipeline(model_factory=Broken).run([Scenario("ar1", context_length=12, horizon=3)], seeds=[7])
    assert run.cases[0].forecast is None
    assert run.cases[0].result.diagnostics[0].status == "ERROR"
    axis = run.plot_forecasts().axes[0]
    assert any("intentional failure" in text.get_text() for text in axis.texts)
    assert not any(line.get_label() == "Predictive mean" for line in axis.lines)
    with pytest.raises(ValueError, match="missing or nonfinite"):
        run.plot_errors()
    axis = run.plot_errors(missing="annotate").axes[0]
    assert any("not in the histogram" in " ".join(text.get_text().split()) for text in axis.texts)
    assert not axis.patches


def test_invalid_forecast_contract_is_not_drawn_as_valid_predictions():
    class SwappedLabels(NaiveForecaster):
        def predict(self, future_index):
            result = super().predict(future_index)
            result.target_names = tuple(reversed(result.target_names))
            return result

    run = Pipeline(SwappedLabels()).run([Scenario("ar1", context_length=12, horizon=3, n_targets=2)], seeds=[4])
    assert run.cases[0].forecast is not None
    assert any(d.status == "FAIL" for d in run.cases[0].result.diagnostics)
    for axis in run.plot_forecasts().axes:
        assert not any(line.get_label() == "Predictive mean" for line in axis.lines)
        assert any("target_mismatch" in text.get_text() for text in axis.texts)


def test_samples_are_not_silently_converted_to_point_or_quantile_outputs():
    class SampleOnly(NaiveForecaster):
        def predict(self, future_index):
            point = super().predict(future_index)
            return Forecast(point.index, samples=np.tile(point.point, (8, 1, 1)), target_names=point.target_names)

    run = Pipeline(SampleOnly()).run([Scenario("ar1", context_length=12, horizon=3)], seeds=[3])
    axis = run.plot_forecasts(show_quantiles=True).axes[0]
    assert not any(line.get_label() == "Predictive mean" for line in axis.lines)
    assert not axis.collections
    assert any("No explicit predictive mean" in text.get_text() for text in axis.texts)
    assert any("Marginal interval unavailable" in text.get_text() for text in axis.texts)


def test_explicit_quantile_bands_are_opt_in_and_labeled_marginal():
    from timegym import GaussianAR1Forecaster

    run = Pipeline(GaussianAR1Forecaster(n_samples=24, seed=3)).run(
        [Scenario("ar1", context_length=12, horizon=3)], seeds=[4])
    assert not run.plot_forecasts().axes[0].collections
    axis = run.plot_forecasts(show_quantiles=True).axes[0]
    assert len(axis.collections) == 1
    assert axis.collections[0].get_label() == "80% marginal interval"


def test_unknown_or_partly_missing_metrics_never_disappear_silently():
    run = Pipeline().run([Scenario("ar1", context_length=12, horizon=3)], seeds=[1, 2])
    with pytest.raises(ValueError, match="unknown_score"):
        run.plot_errors("unknown_score")
    del run.cases[1].result.metrics["mean_reference_mse"]
    with pytest.raises(ValueError, match="seed=2"):
        run.plot_errors()
    axis = run.plot_errors(missing="annotate").axes[0]
    assert "1 scored / 2 evaluated cases" in axis.get_title(loc="left")
    assert any("1 case(s) have no finite" in text.get_text() for text in axis.texts)
    assert sum(patch.get_height() for patch in axis.patches) == 1


def test_plot_style_is_local_and_pipeline_plots_return_figures():
    scenarios = [Scenario("white_noise", context_length=10, horizon=3)]
    before = dict(plt.rcParams)
    pipeline = Pipeline()
    first = pipeline.plot_tests(scenarios, seeds=[4])
    second = pipeline.plot_errors(scenarios, seeds=[4, 5], metric="mse")
    assert len(first.axes) == len(second.axes) == 1
    assert dict(plt.rcParams) == before


def test_importing_package_does_not_import_matplotlib():
    source = "import sys; import timegym; from timegym.evaluation import ForecastRun; assert 'matplotlib' not in sys.modules"
    completed = subprocess.run([sys.executable, "-c", source], cwd=Path(__file__).resolve().parents[1],
                               env=os.environ.copy(), capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr


def test_mixed_random_walk_drift_is_named_per_target_and_aggregated_in_error_plot():
    run = Pipeline().run([Scenario("random_walk", context_length=12, horizon=3, n_targets=2,
                                   parameters={"drift": [0., 1e-15]})], seeds=[7])
    axes = run.plot_forecasts().axes
    assert "Random walk without drift · y0" in axes[0].get_title(loc="left")
    assert "Random walk with drift · y1" in axes[1].get_title(loc="left")
    assert "Random walk (mixed drift across targets)" in run.plot_errors().axes[0].get_title(loc="left")
    assert run.report.cases[0].scenario == "random_walk"
    assert run.report.cases[0].manifest["kind"] == "random_walk"


def test_custom_case_kind_remains_plotable_without_a_builtin_scenario_label():
    class MeasuredCase(Scenario):
        def generate(self, seed=None):
            case = super().generate(seed)
            case.manifest["kind"] = "Measured sensor signal"
            case.manifest["parameters"] = {"source": "custom dataset"}
            return case

    run = Pipeline().run([MeasuredCase("white_noise", context_length=12, horizon=3)], seeds=[7])
    assert "Measured sensor signal" in run.plot_forecasts().axes[0].get_title(loc="left")
    assert "Measured sensor signal" in run.plot_errors().axes[0].get_title(loc="left")
