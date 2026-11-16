"""Your own cases: cutting a series into cases, scoring them, and scoring forecasts made earlier."""

import json

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from timegym import (
    Case,
    Forecast,
    ForecastRun,
    LastValueForecaster,
    Pipeline,
    Scenario,
    TestReport,
    cases_from_series,
)
from timegym.diagnostics import evaluate_forecast


def daily_rain(periods=120, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=periods, freq="D")
    return pd.Series(rng.gamma(2.0, 2.0, periods), index=dates, name="rain_mm")


def repeat_last(case):
    last = case.task.context.iloc[-1].to_numpy()
    return Forecast(case.task.future_index, point=np.tile(last, (len(case.task.future_index), 1)))


def test_three_ways_to_say_where_a_history_ends():
    rain = daily_rain()
    by_rows, by_share, by_label = cases_from_series(rain, [60, 0.75, "2024-04-10"], horizon=7)
    assert [case.manifest["context_length"] for case in (by_rows, by_share, by_label)] == [60, 90, 101]
    assert by_label.manifest["origin"] == "2024-04-10"
    assert by_rows.manifest["origin"] == "2024-02-29"
    for case in (by_rows, by_share, by_label):
        assert case.reference is None
        assert case.task.context.index[-1] < case.task.future_index[0]
        assert len(case.task.future_index) == 7
        pd.testing.assert_index_equal(case.observed.index, case.task.future_index)
        rows = case.manifest["context_length"]
        pd.testing.assert_frame_equal(case.task.context, rain.to_frame().iloc[:rows])
        pd.testing.assert_frame_equal(case.observed, rain.to_frame().iloc[rows:rows + 7])


def test_a_label_between_two_rows_keeps_every_row_at_or_before_it():
    readings = pd.Series([1.0, 2.0, 3.0, 4.0], index=pd.to_datetime(
        ["2024-01-01 10:00", "2024-01-01 10:50", "2024-01-01 12:10", "2024-01-01 13:00"]))
    (case,) = cases_from_series(readings, [pd.Timestamp("2024-01-01 12:00")], horizon="rest")
    assert list(case.task.context.iloc[:, 0]) == [1.0, 2.0]
    assert list(case.observed.iloc[:, 0]) == [3.0, 4.0]
    assert case.manifest["horizon"] == 2


def test_manifest_names_the_series_and_fingerprints_it():
    rain = daily_rain()
    (case,) = cases_from_series(rain, [60], 7, name="rain", manifest={"station": "north"})
    assert case.manifest["kind"] == "rain" and case.manifest["seed"] is None
    assert case.manifest["target_names"] == ["rain_mm"] and case.manifest["station"] == "north"
    again = cases_from_series(rain.copy(), [60], 7, name="rain", manifest={"station": "north"})[0]
    assert again.manifest == case.manifest
    changed = rain.copy()
    changed.iloc[-1] += 1.0  # a value after the case, still part of the series
    assert cases_from_series(changed, [60], 7)[0].manifest["series_sha256"] != case.manifest["series_sha256"]
    unnamed = cases_from_series(rain.rename(None), [60], 7, name="rain")[0]
    assert unnamed.task.target_names == ("rain",)


def test_several_series_stay_together():
    rng = np.random.default_rng(1)
    stores = pd.DataFrame(rng.poisson(20, size=(50, 2)).astype(float), columns=["north", "south"],
                          index=pd.date_range("2024-01-01", periods=50, freq="D"))
    cases = cases_from_series(stores, range(20, 44, 8), horizon=5, name="stores")
    assert len(cases) == 3
    assert all(case.task.target_names == ("north", "south") for case in cases)
    assert [case.manifest["origin"] for case in cases] == ["2024-01-20", "2024-01-28", "2024-02-05"]


def test_missing_values_are_refused_with_the_origin_named():
    rain = daily_rain()
    rain.iloc[70] = np.nan
    assert len(cases_from_series(rain, [60], 7)) == 1  # the gap lies after this case
    with pytest.raises(ValueError, match="origin 80: the history has missing"):
        cases_from_series(rain, [60, 80], 7)
    with pytest.raises(ValueError, match="origin 65: the future has missing"):
        cases_from_series(rain, [65], 7)


@pytest.mark.parametrize("origins, horizon, message", [
    ([0], 7, "at least one row of history"),
    ([120], 7, "at least one row of history"),
    (["2023-01-01"], 7, "at least one row of history"),
    ([115], 7, "fewer than the horizon"),
    ([], 7, "at least one place"),
    ([60], 0, "horizon"),
    ([60], "all", "horizon"),
])
def test_impossible_cuts_are_refused(origins, horizon, message):
    with pytest.raises(ValueError, match=message):
        cases_from_series(daily_rain(), origins, horizon)


def test_bad_inputs_are_refused():
    rain = daily_rain()
    with pytest.raises(TypeError, match="row count"):
        cases_from_series(rain, [True], 7)
    with pytest.raises(TypeError, match="Series or DataFrame"):
        cases_from_series(rain.to_numpy(), [60], 7)
    with pytest.raises(ValueError, match="set by cases_from_series"):
        cases_from_series(rain, [60], 7, manifest={"origin": "today"})
    with pytest.raises(ValueError, match="unique and increasing"):
        cases_from_series(rain.iloc[::-1], [60], 7)


def test_run_and_test_take_your_own_cases():
    rain = daily_rain()
    cases = cases_from_series(rain, [60, 80, 100], 7, name="rain")
    run = Pipeline(LastValueForecaster()).run(cases=cases)
    assert [(entry.scenario_index, entry.seed_index) for entry in run.cases] == [(0, 0), (0, 1), (0, 2)]
    assert all(entry.result.scenario == "rain" and entry.result.seed is None for entry in run.cases)
    for entry in run.cases:
        expected = np.mean((entry.case.observed.to_numpy() - entry.case.task.context.iloc[-1].to_numpy()) ** 2)
        assert entry.result.metrics["mse"] == pytest.approx(expected)
        assert "mean_reference_mse" not in entry.result.metrics  # no answer key for real data
    report = Pipeline(LastValueForecaster()).test(cases=cases)
    assert [case.metrics for case in report.cases] == [entry.result.metrics for entry in run.cases]
    report.assert_passed()
    loaded = TestReport.from_json(report.to_json())
    assert loaded.cases[0].manifest == cases[0].manifest


def test_each_case_gets_a_fresh_model():
    class Counting:
        def fit(self, context):
            self.fitted_on = len(context)
            return self

        def predict(self, future_index):
            return np.full(len(future_index), float(self.fitted_on))

    cases = cases_from_series(daily_rain(), [30, 60], 3)
    model = Counting()
    run = Pipeline(model).run(cases=cases)
    assert [float(entry.forecast.point[0, 0]) for entry in run.cases] == [30.0, 60.0]
    assert not hasattr(model, "fitted_on")


def test_hand_built_cases_need_no_kind_or_seed():
    rain = daily_rain().to_frame()
    from timegym import ForecastTask

    case = Case(ForecastTask(rain.iloc[:50], rain.index[50:55]), rain.iloc[50:55], None, {"site": "north"})
    result = evaluate_forecast(case, repeat_last(case))
    assert (result.scenario, result.seed) == ("case", None)
    report = Pipeline(LastValueForecaster()).test(cases=[case, case])
    assert [c.scenario for c in report.cases] == ["case", "case"]
    assert report.cases[0].manifest == {"site": "north"}


def test_drawn_cases_keep_their_identity_and_still_replay():
    drawn = [Scenario("ar1", context_length=40, horizon=4).generate(seed) for seed in (3, 4)]
    report = Pipeline().test(cases=drawn)
    assert [(c.scenario, c.seed) for c in report.cases] == [("ar1", 3), ("ar1", 4)]
    replayed = report.replay_case(1)
    assert replayed.metadata["replay"]["reproduced"]


def test_cases_cannot_be_mixed_with_drawn_scenarios():
    cases = cases_from_series(daily_rain(), [60], 7)
    pipeline = Pipeline(LastValueForecaster())
    with pytest.raises(ValueError, match="not both"):
        pipeline.run([Scenario("ar1")], cases=cases)
    with pytest.raises(ValueError, match="not both"):
        pipeline.test(cases=cases, seeds=[1])
    with pytest.raises(TypeError, match="not one Case"):
        pipeline.run(cases=cases[0])
    with pytest.raises(TypeError, match=r"cases\[1\]"):
        pipeline.run(cases=[cases[0], "case"])
    with pytest.raises(ValueError, match="at least one case"):
        pipeline.run(cases=[])


def test_a_model_that_crashes_on_one_case_does_not_stop_the_run():
    class Fragile:
        def fit(self, context):
            self.rows = len(context)
            return self

        def predict(self, future_index):
            if self.rows == 60:
                raise RuntimeError("not enough memory")
            return np.zeros(len(future_index))

    report = Pipeline(Fragile()).test(cases=cases_from_series(daily_rain(), [30, 60, 90], 3))
    assert [c.diagnostics[0].status for c in report.cases] == ["PASS", "ERROR", "PASS"]
    assert "not enough memory" in report.cases[1].diagnostics[0].message


def test_forecasts_made_earlier_are_scored_like_a_run():
    cases = cases_from_series(daily_rain(), [60, 80, 100], 7, name="rain")
    forecasts = [repeat_last(case) for case in cases]
    recorded = ForecastRun.from_forecasts(cases, forecasts, metadata={"model": "repeat last", "commit": "abc"})
    fitted = Pipeline(LastValueForecaster()).run(cases=cases)
    assert [e.result.metrics for e in recorded.cases] == [e.result.metrics for e in fitted.cases]
    assert recorded.metadata["model"] == "repeat last" and recorded.metadata["commit"] == "abc"
    assert "pipeline_config" not in recorded.metadata
    assert set(recorded.metadata["diagnostic_settings"]) == {
        "check_dependence", "n_calibration", "alpha", "dependence_seed_offset"}
    json.dumps(recorded.report.to_dict(), allow_nan=False)
    figure = recorded.plot_forecasts()
    assert "history ends 2024-02-29" in figure.axes[0].get_title(loc="left")


def test_forecasts_made_earlier_accept_plain_arrays_and_gaps():
    cases = cases_from_series(daily_rain(), [60, 80], 5)
    recorded = ForecastRun.from_forecasts(cases, [np.zeros(5), None])
    assert recorded.cases[0].result.metrics["mse"] > 0
    assert recorded.cases[1].forecast is None
    (problem,) = recorded.cases[1].result.diagnostics
    assert (problem.name, problem.status) == ("model_execution", "ERROR")
    assert "No forecast was recorded" in problem.message
    with pytest.raises(ValueError, match="Got 1 forecasts for 2 cases"):
        ForecastRun.from_forecasts(cases, [np.zeros(5)])


def test_replay_explains_why_your_own_cases_cannot_be_replayed():
    cases = cases_from_series(daily_rain(), [60], 7)
    report = Pipeline(LastValueForecaster()).test(cases=cases)
    with pytest.raises(ValueError, match="not drawn from a TimeGym scenario"):
        report.replay_case(0)
    drawn = [Scenario("ar1", context_length=40, horizon=4).generate(3)]
    recorded = ForecastRun.from_forecasts(drawn, [repeat_last(drawn[0])]).report
    with pytest.raises(ValueError, match="no pipeline configuration"):
        recorded.replay_case(0)
