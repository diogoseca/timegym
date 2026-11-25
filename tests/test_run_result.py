"""One call, ``pipeline.run``: its scorecard, what it keeps, and the calls it replaces."""
import json
import pickle
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from timegym import (Case, ForecastRun, ForecastTask, LastValueForecaster, MeanForecaster, Pipeline,
                     Reference, Scenario, Scorecard, TestReport, cases_from_series)
from timegym.diagnostics import calibration
from timegym.scenarios import default_scenarios

TABLE = ["pattern", "noise", "score", "score_low", "score_high", "verdict", "settled", "attainable",
         "mistakes", "unavoidable_mse", "avoidable_mse", "mse", "instances", "failed_checks"]


def _old_scorecard(pipeline, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return pipeline.scorecard(*args, **kwargs)


def _own_data():
    values = np.sin(np.arange(200) / 5) + np.random.default_rng(0).normal(size=200)
    return cases_from_series(pd.DataFrame({"y": values}), origins=[150, 170], horizon=12)


def test_a_bare_run_draws_the_scorecards_series():
    result = Pipeline(LastValueForecaster()).run()
    assert len(result.cases) == 10 * 2 * 10
    assert result.titles[:2] == ["Pure noise · Low noise", "Pure noise · High noise"]
    card = result.scorecard
    assert isinstance(card, Scorecard) and card.run is result
    assert (card.table["pattern"].nunique(), card.instances, card.seed) == (10, 10, 0)


def test_the_runs_scorecard_is_the_old_scorecard_number_for_number():
    pipeline = Pipeline(LastValueForecaster())
    old = _old_scorecard(pipeline, ["trend", "mean_reverting"], instances=4, seed=3)
    new = pipeline.run(["trend", "mean_reverting"], instances=4, seed=3).scorecard
    pd.testing.assert_frame_equal(old.table[TABLE], new.table[TABLE])
    assert str(old) == str(new)
    assert (old.title, old.seed, old.instances) == (new.title, new.seed, new.instances)


def test_scorecard_settings_go_through_run():
    pipeline = Pipeline(LastValueForecaster())
    card = pipeline.run(["random_walk"], noise_levels=[0.2, 0.4], instances=3, history_length=40, horizon=6,
                        title="Mine", interval_level=0.99).scorecard
    assert card.title == "Mine" and card.interval_level == 0.99
    assert list(card.table["noise_sd"]) == [0.2, 0.4]
    assert card.cells[0].scenario.history_length == 40 and card.cells[0].scenario.horizon == 6
    assert pipeline.run("random_walk", instances=2).scorecard.table["pattern"].tolist() == ["random_walk"] * 2


def test_series_and_forecasts_are_dropped_by_default_and_the_card_still_draws():
    pipeline = Pipeline(LastValueForecaster())
    dropped = pipeline.run(["trend", "seasonal"], instances=3)
    kept = pipeline.run(["trend", "seasonal"], instances=3, keep_data=True)
    assert dropped.kept_data is False and kept.kept_data is True
    assert all(entry.case is None and entry.forecast is None for entry in dropped.cases)
    assert all(entry.case is not None and entry.forecast is not None for entry in kept.cases)
    assert [case.to_dict() for case in dropped.report.cases] == [case.to_dict() for case in kept.report.cases]
    pd.testing.assert_frame_equal(dropped.scorecard.table, kept.scorecard.table)
    assert str(dropped.scorecard) == str(kept.scorecard)
    figure = dropped.scorecard.plot()
    plt.close(figure)
    plt.close(dropped.scorecard.plot(compact=True))
    assert dropped.scorecard._repr_png_()[:4] == b"\x89PNG"
    plt.close(dropped.plot_errors())


def test_a_dropped_run_keeps_what_a_baseline_needs(tmp_path):
    pipeline = Pipeline(LastValueForecaster())
    dropped = pipeline.run(["trend"], instances=3).scorecard
    kept = pipeline.run(["trend"], instances=3, keep_data=True).scorecard
    dropped.save_baseline(tmp_path / "dropped.json")
    kept.save_baseline(tmp_path / "kept.json")
    assert (tmp_path / "dropped.json").read_text() == (tmp_path / "kept.json").read_text()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # few instances
        assert len(dropped.compare(tmp_path / "kept.json").worse) == 0


def test_reading_forecasts_of_a_dropped_run_names_the_switch():
    result = Pipeline(LastValueForecaster()).run(["trend"], instances=2)
    with pytest.raises(ValueError, match=r"plot_forecasts\(\).*keep_data=True"):
        result.plot_forecasts()
    with pytest.raises(ValueError, match="calibration.*keep_data=True"):
        calibration(result)
    kept = Pipeline(LastValueForecaster()).run(["trend"], instances=2, keep_data=True)
    plt.close(kept.plot_forecasts())


def test_the_result_answers_for_its_report():
    result = Pipeline(LastValueForecaster()).run(["trend"], instances=2)
    assert isinstance(result.report, TestReport)
    result.assert_passed()
    assert result.checks == result.report.checks
    assert json.loads(result.to_json()) == json.loads(result.report.to_json())

    class Broken:
        def fit(self, history):
            return self

        def predict(self, future_index):
            raise RuntimeError("no forecast")

    with pytest.raises(AssertionError):
        Pipeline(Broken()).run(["trend"], instances=1).assert_passed()


def test_to_json_writes_the_report(tmp_path):
    result = Pipeline(LastValueForecaster()).run(["trend"], instances=2)
    path = tmp_path / "report.json"
    text = result.to_json(path)
    assert path.read_text(encoding="utf-8") == text + "\n"
    assert TestReport.read(path).to_dict() == result.report.to_dict()


def test_scenarios_you_built_give_one_row_each():
    scenarios = [Scenario("trend", history_length=30, horizon=5), Scenario("ar1", history_length=30, horizon=5),
                 Scenario("trend", history_length=30, horizon=5, parameters={"slope": 0.5})]
    result = Pipeline(LastValueForecaster()).run(scenarios, instances=3)
    card = result.scorecard
    assert card.table["pattern"].tolist() == ["trend", "ar1", "trend (2)"]
    assert set(card.table["noise"]) == {"As built"} and card.table["noise_sd"].isna().all()
    assert card.table["attainable"].isna().all()
    assert result.titles == []  # the plots keep naming each scenario by its kind and parameters
    text = str(card)
    assert "3 scenarios you built" in text and "| Scenario | As built | Mistakes |" in text
    assert "As built:" not in text  # one column: its name is not repeated before each mistake
    plt.close(card.plot())
    plt.close(card.plot(compact=True))
    with pytest.raises(ValueError, match="scenarios you built"):
        card.save_baseline("never-written.json")
    assert not Path("never-written.json").exists()


def test_old_default_scenarios_get_a_card_too():
    card = Pipeline(LastValueForecaster()).run(default_scenarios(), instances=2).scorecard
    assert card.table["pattern"].tolist() == ["white_noise", "trend", "seasonal", "ar1", "random_walk"]


def test_your_own_data_has_no_scorecard():
    result = Pipeline(LastValueForecaster()).run(cases=_own_data())
    assert result.kept_data is False and len(result.cases) == 2
    with pytest.raises(ValueError, match="no scorecard.*answer key.*result.checks"):
        result.scorecard
    plt.close(result.plot_errors(metric="mse"))
    kept = Pipeline(LastValueForecaster()).run(cases=_own_data(), keep_data=True)
    assert kept.cases[0].forecast is not None
    plt.close(kept.plot_forecasts())


@pytest.mark.parametrize("setting", [{"noise_levels": 3}, {"history_length": 50}, {"horizon": 6},
                                     {"title": "Mine"}, {"interval_level": 0.99}, {"attainable": True}])
def test_scorecard_settings_are_refused_for_your_own_data(setting):
    name = next(iter(setting))
    with pytest.raises(ValueError, match=f"{name} belongs to the scorecard"):
        Pipeline().run(cases=_own_data(), **setting)


def test_cases_still_refuse_scenarios_and_seeds():
    with pytest.raises(ValueError, match="not both"):
        Pipeline().run(["trend"], cases=_own_data())
    with pytest.raises(ValueError, match="not both"):
        Pipeline().run(cases=_own_data(), instances=2)


def test_scorecard_settings_for_kinds_of_series_are_refused_for_scenarios_you_built():
    with pytest.raises(ValueError, match="horizon set the scorecard's kinds of series"):
        Pipeline().run([Scenario("trend")], horizon=6)


def test_kinds_of_series_and_scenarios_do_not_mix():
    with pytest.raises(TypeError, match="not both in one run"):
        Pipeline().run(["trend", Scenario("ar1")])


def test_keep_data_must_be_a_boolean():
    with pytest.raises(TypeError, match="keep_data must be True or False"):
        Pipeline().run(["trend"], keep_data="yes")


def test_explicit_seeds_name_the_card_seed_only_when_they_count_up():
    pipeline = Pipeline(LastValueForecaster())
    assert pipeline.run(["trend"], seeds=[5, 6, 7]).scorecard.seed == 5
    card = pipeline.run(["trend"], seeds=[5, 9]).scorecard
    assert card.seed is None and card.instances == 2
    assert "your own seeds" in str(card)
    with pytest.raises(ValueError, match="seeds of your own"):
        card.save_baseline("never-written.json")
    plt.close(card.plot())


class _NoAnswerKey:
    """A scenario whose cases carry no answer key."""

    kind = "mystery"

    def generate(self, seed):
        values = pd.DataFrame({"y": np.random.default_rng(seed).normal(size=30)})
        task = ForecastTask(values.iloc[:25], values.index[25:])
        return Case(task, values.iloc[25:], None, {"kind": "mystery", "seed": seed})


class _FlatLine:
    """A scenario of your own with an answer key: a flat line at 1 plus noise. It has no lengths."""

    kind = "flat_line"

    def generate(self, seed):
        values = pd.DataFrame({"y": 1.0 + np.random.default_rng(seed).normal(size=30)})
        task = ForecastTask(values.iloc[:25], values.index[25:])
        key = Reference.from_function(lambda index: np.ones((len(index), 1)), values.index[25:],
                                      covariance=np.eye(5))
        return Case(task, values.iloc[25:], key, {"kind": "flat_line", "seed": seed})


def test_a_scenario_of_your_own_without_lengths_gets_a_row():
    card = Pipeline(MeanForecaster()).run([_FlatLine()], instances=4).scorecard
    assert card.table["pattern"].tolist() == ["flat_line"] and np.isnan(card.table["attainable"].iloc[0])
    assert 0 <= card.table["score"].iloc[0] <= 100
    plt.close(card.plot())


def test_a_card_that_cannot_be_made_says_why_when_read():
    result = Pipeline(LastValueForecaster()).run([_NoAnswerKey()], instances=2)
    result.assert_passed()  # the run itself stands
    with pytest.raises(ValueError, match="could not be made: none of these series has an answer key"):
        result.scorecard
    kept = Pipeline(LastValueForecaster()).run([_NoAnswerKey()], instances=2, keep_data=True)
    with pytest.raises(ValueError, match="none of these series has an answer key"):
        kept.scorecard


def test_a_run_with_its_scorecard_pickles():
    for keep in (False, True):
        result = Pipeline(LastValueForecaster()).run(["trend", "seasonal"], instances=2, keep_data=keep)
        result.scorecard
        again = pickle.loads(pickle.dumps(result))
        assert str(again.scorecard) == str(result.scorecard)
        plt.close(again.scorecard.plot())
    built = Pipeline(LastValueForecaster()).run([Scenario("trend", history_length=30, horizon=5)], instances=2)
    assert str(pickle.loads(pickle.dumps(built)).scorecard) == str(built.scorecard)


def test_forecast_runs_made_elsewhere_keep_their_data_and_have_no_card():
    cases = _own_data()
    run = ForecastRun.from_forecasts(cases, [np.zeros(12), np.zeros(12)])
    assert run.kept_data is True
    with pytest.raises(ValueError, match="no scorecard"):
        run.scorecard


# --- The calls run() replaces ---------------------------------------------------------------

def _one_warning(caught):
    messages = [str(w.message) for w in caught if w.category is DeprecationWarning]
    assert len(messages) == 1, messages
    assert next(w for w in caught if w.category is DeprecationWarning).filename == __file__
    return messages[0]


def test_test_warns_and_keeps_its_old_series():
    pipeline = Pipeline(LastValueForecaster())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        report = pipeline.test()
    message = _one_warning(caught)
    assert "pipeline.test(...) is deprecated" in message and "pipeline.run(...)" in message
    assert "will be removed in a later release" in message
    assert isinstance(report, TestReport) and len(report.cases) == 5 * 3
    assert [case.scenario for case in report.cases[::3]] == ["white_noise", "trend", "seasonal", "ar1",
                                                             "random_walk"]
    expected = pipeline.run(default_scenarios(), instances=3).report
    assert [case.to_dict() for case in report.cases] == [case.to_dict() for case in expected.cases]


def test_test_still_takes_ready_made_cases():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        report = Pipeline(LastValueForecaster()).test(cases=_own_data())
        with pytest.raises(ValueError, match="not both"):
            Pipeline().test(["trend"], cases=_own_data())
    assert len(report.cases) == 2


def test_scorecard_warns_and_keeps_every_series():
    pipeline = Pipeline(LastValueForecaster())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        card = pipeline.scorecard(["trend"], instances=2)
    message = _one_warning(caught)
    assert "pipeline.scorecard(...) is deprecated" in message and "pipeline.run(...).scorecard" in message
    assert card.run.kept_data is True
    plt.close(card.run.plot_forecasts())
    pd.testing.assert_frame_equal(card.table, pipeline.run(["trend"], instances=2).scorecard.table)


def test_a_bad_interval_level_is_refused_before_any_forecast():
    fits = []

    class Counted(LastValueForecaster):
        def fit(self, history):
            fits.append(1)
            return super().fit(history)

    with pytest.raises(ValueError, match="interval_level must be a number between 0 and 1"):
        Pipeline(Counted()).run(["trend"], instances=2, interval_level=95)
    assert fits == []
