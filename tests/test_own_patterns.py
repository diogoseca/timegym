"""Your own kinds of series on the scorecard: a Pattern beside the built-in keys."""
import json
import math
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import timegym
from timegym import (
    PATTERNS,
    Case,
    ForecastTask,
    Pattern,
    Pipeline,
    Reference,
    Scenario,
    scenario_grid,
    MeanForecaster,
    SeasonalMeanForecaster,
)


class RainScenario:
    """Rain (1) or no rain (0) at each step; the chance of rain follows a 12-step cycle."""

    kind = "rain"

    def __init__(self, swing, history_length, horizon):
        self.swing, self.history_length, self.horizon = swing, history_length, horizon

    def generate(self, seed):
        steps = np.arange(self.history_length + self.horizon)
        chance = 0.5 + self.swing * np.sin(2 * np.pi * steps / 12)
        rain = (np.random.default_rng(seed).random(len(steps)) < chance).astype(float)
        series = pd.DataFrame({"rain": rain}, index=pd.RangeIndex(len(steps)))
        future = series.index[self.history_length:]
        p = chance[self.history_length:]
        answer_key = Reference(mean=p[:, None], variance=(p * (1 - p))[:, None],
                               covariance=np.diag(p * (1 - p)), index=future, target_names=("rain",))
        task = ForecastTask(series.iloc[:self.history_length], future)
        return Case(task, series.iloc[self.history_length:], answer_key, {"kind": "rain", "seed": seed})


def simulate_rain(rng, history, horizon, noise):
    """The same rain through simulate: what happened, and the chance of rain as the answer key."""
    steps = np.arange(history + horizon)
    chance = 0.5 + 0.45 / (1 + noise) * np.sin(2 * np.pi * steps / 12)
    rained = rng.random(len(steps)) < chance
    ahead = chance[history:]
    return dict(values=rained, mean=ahead, variance=ahead * (1 - ahead))


def built_rain(noise, history_length, horizon):
    return RainScenario(0.45 / (1 + noise) ** 0.5, history_length, horizon)


SEASONAL_MEAN = SeasonalMeanForecaster(period=12)


def rain_pattern(key="rain", attainable_by=SEASONAL_MEAN, simulate=None, build=None, **texts):
    """A rain row made with simulate, or with build when one is given."""
    texts = {"title": "Rain or no rain", "aka": "a yes/no outcome", "predictable": "The chance of rain.",
             "unpredictable": "Whether it rains.", **texts}
    if simulate is None and build is None:
        simulate = simulate_rain
    return Pattern(key=key, build=build, simulate=simulate, attainable_by=attainable_by, **texts)


@pytest.fixture(scope="module")
def mean_card():
    return Pipeline(MeanForecaster()).scorecard(patterns=["seasonal", rain_pattern()], instances=4)


def test_pattern_is_exported_from_timegym():
    assert timegym.Pattern is timegym.grid.Pattern
    assert "Pattern" in timegym.__all__


def test_own_pattern_is_a_row_beside_the_built_in_ones(mean_card):
    table = mean_card.table
    assert list(table["pattern"]) == ["seasonal", "seasonal", "rain", "rain"]
    rain = table[table["pattern"] == "rain"]
    assert set(rain["title"]) == {"Rain or no rain"}
    assert rain["score"].between(0, 100).all()
    assert rain["verdict"].notna().all()
    assert rain["attainable"].isna().all()  # not asked for
    assert mean_card.cells[2].definition.title == "Rain or no rain"


def test_built_in_rows_do_not_change_beside_your_own(mean_card):
    alone = Pipeline(MeanForecaster()).scorecard(patterns=["seasonal"], instances=4)
    after = Pipeline(MeanForecaster()).scorecard(patterns=[rain_pattern(), "seasonal"], instances=4)
    expected = alone.table.set_index(["pattern", "noise"])["score"]
    for card in (mean_card, after):
        got = card.table.set_index(["pattern", "noise"]).loc["seasonal", "score"]
        assert np.allclose(got.to_numpy(), expected.loc["seasonal"].to_numpy(), rtol=0, atol=1e-12)


def test_own_pattern_gets_the_brier_share_as_its_score():
    pattern = rain_pattern()
    card = Pipeline(MeanForecaster()).scorecard(patterns=[pattern], instances=3)
    # Recompute from the cases: the answer key's variance p(1 - p) against the squared distance to p.
    unavoidable, avoidable = [], []
    for entry in card.run.cases:
        if entry.scenario_index != 0:  # the low-noise column
            continue
        p = entry.reference.mean[:, 0]
        point = entry.forecast.point[:, 0]
        unavoidable.append(np.mean(p * (1 - p)))
        avoidable.append(np.mean((point - p) ** 2))
    expected = 100 * sum(unavoidable) / (sum(unavoidable) + sum(avoidable))
    assert math.isclose(card.table.loc[0, "score"], expected, rel_tol=1e-9)


def test_attainable_is_simulated_only_when_asked_and_with_a_model_of_the_row():
    pipeline = Pipeline(MeanForecaster())
    asked = pipeline.scorecard(patterns=[rain_pattern()], instances=2, simulate_attainable=True)
    assert asked.table["attainable"].between(85, 95).all()  # about 100 / 1.1 from ten cycles of history
    no_model = pipeline.scorecard(patterns=[rain_pattern(attainable_by=None)], instances=2,
                                  simulate_attainable=True)
    assert no_model.table["attainable"].isna().all()
    not_asked = pipeline.scorecard(patterns=[rain_pattern()], instances=2)
    assert not_asked.table["attainable"].isna().all()


def test_built_in_rows_keep_their_stored_attainable_score_beside_your_own():
    card = Pipeline(MeanForecaster()).scorecard(patterns=["trend", rain_pattern(attainable_by=None)],
                                                       instances=2)
    trend = card.table[card.table["pattern"] == "trend"]["attainable"]
    stored = Pipeline(MeanForecaster()).scorecard(patterns=["trend"], instances=2).table["attainable"]
    assert trend.notna().all()
    assert np.array_equal(trend.to_numpy(), stored.to_numpy())


def test_a_built_in_pattern_object_counts_as_its_key():
    by_object = Pipeline(MeanForecaster()).scorecard(patterns=[PATTERNS["trend"]], instances=2)
    by_key = Pipeline(MeanForecaster()).scorecard(patterns=["trend"], instances=2)
    pd.testing.assert_frame_equal(by_object.table, by_key.table)
    assert by_object.cells[0].definition is PATTERNS["trend"]


def test_own_pattern_can_reuse_a_built_in_recipe():
    gentle = Pattern("gentle_trend", "Gentle trend", "a slow straight line", "The slope.", "The noise.",
                     build=lambda s, n, h: Scenario("trend", n, h, {"slope": 0.01, "sigma": s}))
    card = Pipeline(MeanForecaster()).scorecard(patterns=[gentle], instances=2,
                                                       history_length=60, horizon=12)
    assert list(card.table["pattern"]) == ["gentle_trend", "gentle_trend"]
    assert card.cells[0].scenario.history_length == 60


def test_scenario_grid_keeps_the_pattern_of_each_row():
    pattern = rain_pattern()
    cells = scenario_grid(["noise", pattern], noise_levels=3)
    assert [cell.pattern for cell in cells] == ["noise"] * 3 + ["rain"] * 3
    assert all(cell.definition is pattern for cell in cells[3:])
    assert all(cell.definition is PATTERNS["noise"] for cell in cells[:3])
    assert all(cell.title == "Rain or no rain" for cell in cells[3:])
    assert all(cell.scenario.kind == "rain" for cell in cells[3:])
    assert [cell.scenario.noise for cell in cells[3:]] == [cell.noise_sd for cell in cells[3:]]
    built = scenario_grid([rain_pattern(build=built_rain)])
    assert all(isinstance(cell.scenario, RainScenario) for cell in built)


def test_the_printed_card_names_your_row_by_its_title(mean_card):
    text = mean_card.to_markdown()
    assert "| Rain or no rain |" in text
    named = mean_card.good_at + mean_card.fails_at + mean_card.noise_sensitive
    assert all(name in set(mean_card.table["title"]) for name in named)
    assert str(mean_card)


@pytest.mark.parametrize("compact", [False, True])
def test_the_picture_draws_your_row(mean_card, compact):
    figure = mean_card.plot(compact=compact)
    texts = [text.get_text() for axis in figure.axes for text in axis.texts] + [t.get_text() for t in figure.texts]
    labels = [label.get_text() for axis in figure.axes for label in axis.get_yticklabels()]
    assert any("Rain or no rain" in text for text in texts + labels)
    plt.close(figure)


def test_baseline_and_compare_know_your_row_by_its_key(tmp_path, mean_card):
    path = mean_card.save_baseline(tmp_path / "baseline.json")
    saved = json.loads(path.read_text())
    assert "rain" in json.dumps(saved)
    again = Pipeline(MeanForecaster()).scorecard(patterns=["seasonal", rain_pattern()], instances=4)
    with pytest.warns(UserWarning):  # fewer than 20 instances
        comparison = again.compare(path)
    assert (comparison.table["change"] == "unchanged").all()
    assert set(comparison.table["pattern"]) == {"seasonal", "rain"}
    other_key = Pipeline(MeanForecaster()).scorecard(patterns=["seasonal", rain_pattern("showers")],
                                                            instances=4)
    with pytest.raises(ValueError):
        other_key.compare(path)


# --- What the grid refuses, and what it says ------------------------------------------------

@pytest.mark.parametrize("entry, message", [
    ("weather", "patterns must be keys from"),
    (3, "patterns must be keys from"),
    (rain_pattern("trend"), "the key 'trend' belongs to a built-in pattern; give yours another key"),
    (rain_pattern(""), "a pattern's key must be a nonempty string"),
    (rain_pattern(title=""), "pattern 'rain' needs a nonempty title"),
    (rain_pattern(aka=None), "check aka"),
])
def test_grid_refuses_what_is_not_a_pattern(entry, message):
    with pytest.raises(ValueError, match=message):
        scenario_grid([entry])


def test_grid_refuses_two_rows_with_one_key():
    with pytest.raises(ValueError, match="patterns must be distinct"):
        scenario_grid([rain_pattern(), rain_pattern()])


def test_grid_refuses_a_pattern_without_one_way_to_make_its_series():
    neither = Pattern("rain", "Rain or no rain", "", "", "")
    with pytest.raises(TypeError, match=r"needs simulate\(rng, history_length, horizon, noise\) or "
                                        r"build\(noise, history_length, horizon\), one of the two; got neither"):
        scenario_grid([neither])
    with pytest.raises(TypeError, match="one of the two; got simulate and build"):
        scenario_grid([rain_pattern(simulate=simulate_rain, build=built_rain)])
    with pytest.raises(TypeError, match="build must be a function"):
        scenario_grid([rain_pattern(build="not a function")])
    with pytest.raises(TypeError, match="simulate must be a function"):
        scenario_grid([rain_pattern(simulate=42)])


def test_grid_refuses_an_attainable_by_that_is_not_a_model():
    with pytest.raises(TypeError, match="attainable_by must be a model with fit and predict, a class or function "
                                        "that returns one, or None; got a str"):
        scenario_grid([rain_pattern(attainable_by="seasonal mean")])


class _NoGenerate:
    kind, history_length, horizon = "rain", 120, 24


def _shifted(**changes):
    def build(noise, history_length, horizon):
        scenario = RainScenario(0.3, history_length, horizon)
        for name, value in changes.items():
            setattr(scenario, name, value)
        return scenario
    return build


class _ReturnsATable(RainScenario):
    def generate(self, seed):
        return super().generate(seed).observed


class _NoAnswerKey(RainScenario):
    def generate(self, seed):
        case = super().generate(seed)
        return Case(case.task, case.observed, None, case.manifest)


class _FlatAnswerKey(RainScenario):
    """An answer key with one value a step, where the scorecard needs one a step and series."""

    def generate(self, seed):
        case = super().generate(seed)
        key = SimpleNamespace(mean=case.reference.mean[:, 0], variance=case.reference.variance[:, 0])
        return Case(case.task, case.observed, key, case.manifest)


@pytest.mark.parametrize("build, error, message", [
    (lambda s, n, h: _NoGenerate(), TypeError, r"build returned a _NoGenerate without generate\(seed\)"),
    (lambda s, n, h: object(), TypeError, "without generate\\(seed\\), kind, history_length, horizon"),
    (_shifted(kind=""), ValueError, "kind must be a nonempty string"),
    (_shifted(history_length=100), ValueError,
     "build was asked for 120 steps of history and 24 ahead, but its scenario has 100 and 24"),
    (lambda s, n, h: _ReturnsATable(0.3, n, h), TypeError, r"generate\(seed\) must return a Case; got a DataFrame"),
    (lambda s, n, h: _NoAnswerKey(0.3, n, h), ValueError, "its cases have no answer key"),
    (lambda s, n, h: _FlatAnswerKey(0.3, n, h), ValueError,
     r"must each have the shape of what happens, \(24, 1\); got \(24,\) and \(24,\)"),
])
def test_grid_says_what_your_scenario_is_missing(build, error, message):
    with pytest.raises(error, match=message):
        scenario_grid([rain_pattern(build=build)])


def test_pipeline_scorecard_refuses_before_anything_runs():
    calls = []

    class Counting(MeanForecaster):
        def fit(self, history):
            calls.append(1)
            return super().fit(history)

    with pytest.raises(ValueError, match="no answer key"):
        Pipeline(Counting()).scorecard(
            patterns=["noise", rain_pattern(build=lambda s, n, h: _NoAnswerKey(0.3, n, h))])
    assert calls == []


# --- simulate: one series and its answer key, the case built by TimeGym -------------------------

def _case(pattern, noise=0.5, seed=0, history=120, horizon=24):
    (cell,) = [cell for cell in scenario_grid([pattern], [noise], history_length=history, horizon=horizon)]
    return cell.scenario.generate(seed)


def test_simulate_values_are_cut_into_history_and_future():
    seen = {}

    def ramp(rng, history, horizon, noise):
        seen.update(history=history, horizon=horizon, noise=noise, rng=rng)
        values = np.arange(history + horizon, dtype=float)
        return dict(values=values, mean=values[history:], variance=np.ones(horizon))

    case = _case(rain_pattern("ramp", simulate=ramp, attainable_by=None), noise=2.0, history=30, horizon=6)
    assert (seen["history"], seen["horizon"], seen["noise"]) == (30, 6, 2.0)
    assert isinstance(seen["rng"], np.random.Generator)
    assert case.task.history["y"].tolist() == list(range(30))
    assert case.observed["y"].tolist() == list(range(30, 36))
    assert list(case.task.future_index) == list(range(30, 36))
    assert case.reference.mean[:, 0].tolist() == list(range(30, 36))
    assert np.array_equal(case.reference.covariance, np.eye(6))  # independent steps without a covariance
    assert case.manifest["kind"] == "ramp" and case.manifest["seed"] == 0


def test_simulate_yes_no_values_become_zeros_and_ones_and_the_seed_repeats_them():
    first, again, other = (_case(rain_pattern(), seed=seed) for seed in (3, 3, 4))
    assert set(np.unique(first.task.history.to_numpy())) <= {0.0, 1.0}
    pd.testing.assert_frame_equal(first.task.history, again.task.history)
    assert not first.task.history.equals(other.task.history)


def test_simulate_takes_several_series_and_a_covariance():
    def pair(rng, history, horizon, noise):
        values = rng.normal(size=(history + horizon, 2))
        covariance = np.kron(np.eye(horizon), [[1.0, 0.5], [0.5, 1.0]])
        return dict(values=values, mean=np.zeros((horizon, 2)), variance=np.ones((horizon, 2)), covariance=covariance)

    case = _case(rain_pattern("pair", simulate=pair, attainable_by=None), horizon=4)
    assert list(case.task.history.columns) == ["y0", "y1"]
    assert case.reference.covariance[0, 1] == 0.5
    card = Pipeline(MeanForecaster()).scorecard(patterns=[rain_pattern("pair", simulate=pair,
                                                                              attainable_by=None)], instances=2)
    assert card.table["score"].between(0, 100).all()


def _returning(**parts):
    def simulate(rng, history, horizon, noise):
        good = simulate_rain(rng, history, horizon, noise)
        good.update(parts)
        return {name: value for name, value in good.items() if value is not None}
    return simulate


@pytest.mark.parametrize("simulate, error, message", [
    (lambda rng, n, h, s: [1, 2, 3], TypeError,
     "pattern 'rain': simulate must return a dict with values, mean and variance; got a list"),
    (_returning(variance=None), ValueError, "got no variance"),
    (_returning(mean=None, variance=None), ValueError, "got no mean, variance"),
    (_returning(answer=1), ValueError, "got unknown answer"),
    (_returning(values=np.zeros(100)), ValueError,
     r"values must have 144 steps \(120 of history and 24 to forecast\), one column per series; got shape \(100,\)"),
    (_returning(mean=np.zeros(23)), ValueError,
     r"mean must have one value per step to forecast, shape \(24, 1\); got \(23,\)"),
    (_returning(variance=-np.ones(24)), ValueError, "pattern 'rain': simulate: Reference variance must be nonnegative"),
    (_returning(covariance=np.eye(3)), ValueError,
     r"pattern 'rain': simulate: Reference covariance must have shape \(horizon \* targets, horizon \* targets\)"),
])
def test_grid_says_what_simulate_returned_wrong(simulate, error, message):
    with pytest.raises(error, match=message):
        scenario_grid([rain_pattern(simulate=simulate)])


# --- attainable_by: the row's own model ---------------------------------------------------------

def test_attainable_by_takes_a_model_and_never_fits_the_one_you_passed():
    model = SeasonalMeanForecaster(period=12)
    card = Pipeline(MeanForecaster()).scorecard(patterns=[rain_pattern(attainable_by=model)], instances=2,
                                                       simulate_attainable=True)
    assert card.table["attainable"].between(85, 95).all()
    assert not hasattr(model, "history_")  # each series fitted a copy


@pytest.mark.parametrize("given", [
    SeasonalMeanForecaster(period=12),
    lambda: SeasonalMeanForecaster(period=12),
])
def test_attainable_by_gives_the_same_score_as_a_model_or_a_function(given):
    card = Pipeline(MeanForecaster()).scorecard(patterns=[rain_pattern(attainable_by=given)], instances=2,
                                                       simulate_attainable=True)
    reference = Pipeline(MeanForecaster()).scorecard(patterns=[rain_pattern()], instances=2,
                                                     simulate_attainable=True)
    assert np.array_equal(card.table["attainable"].to_numpy(), reference.table["attainable"].to_numpy())


def test_attainable_by_takes_a_class():
    class Half:
        def fit(self, history):
            return self

        def predict(self, future_index):
            return np.full(len(future_index), 0.5)

    card = Pipeline(MeanForecaster()).scorecard(patterns=[rain_pattern(attainable_by=Half)], instances=2,
                                                       simulate_attainable=True)
    assert card.table["attainable"].between(50, 99).all()


def test_fitted_still_makes_a_pattern_with_a_warning():
    with pytest.warns(DeprecationWarning, match="attainable_by"):
        old = Pattern("rain", "Rain or no rain", "", "", "", simulate=simulate_rain,
                      fitted=lambda: SeasonalMeanForecaster(period=12))
    card = Pipeline(MeanForecaster()).scorecard(patterns=[old], instances=2, simulate_attainable=True)
    assert card.table["attainable"].between(85, 95).all()


# --- The seasonal mean, the rain row's own model -------------------------------------------------

def test_seasonal_mean_forecasts_the_mean_at_each_point_of_the_cycle():
    values = [1, 0, 0, 1, 1, 0, 0, 1, 0, 1]  # positions 0..9 on a cycle of 3
    history = pd.DataFrame({"y": values}, index=pd.RangeIndex(10))
    forecast = SeasonalMeanForecaster(period=3).fit(history).predict(pd.RangeIndex(10, 16))
    # step 10 is at point 1 of the cycle (0, 1, 1), step 11 at point 2 (0, 0, 0), step 12 at point 0 (1, 1, 0, 1)
    assert np.allclose(forecast.point[:, 0], [2 / 3, 0, 3 / 4, 2 / 3, 0, 3 / 4])


def test_seasonal_mean_works_on_several_series_and_needs_one_full_cycle():
    history = pd.DataFrame({"a": np.arange(8.0), "b": np.arange(8.0) % 4}, index=pd.RangeIndex(8))
    forecast = SeasonalMeanForecaster(period=4).fit(history).predict(pd.RangeIndex(8, 12))
    assert np.allclose(forecast.point, [[2, 0], [3, 1], [4, 2], [5, 3]])
    with pytest.raises(ValueError, match="Insufficient history"):
        SeasonalMeanForecaster(period=12).fit(history)
    copy = SeasonalMeanForecaster(period=4).clone_unfitted()
    assert type(copy) is SeasonalMeanForecaster and copy.params() == {"period": 4}


def test_seasonal_mean_scores_on_the_seasonal_row():
    card = Pipeline(SeasonalMeanForecaster(period=12)).scorecard(["seasonal"], instances=4)
    assert (card.table["score"] > 80).all()
