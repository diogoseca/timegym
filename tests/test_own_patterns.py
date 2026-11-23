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
    NaiveForecaster,
    Pattern,
    Pipeline,
    Reference,
    Scenario,
    scenario_grid,
)


class RainScenario:
    """Rain (1) or no rain (0) at each step; the chance of rain follows a 12-step cycle."""

    kind = "rain"

    def __init__(self, swing, context_length, horizon):
        self.swing, self.context_length, self.horizon = swing, context_length, horizon

    def generate(self, seed):
        steps = np.arange(self.context_length + self.horizon)
        chance = 0.5 + self.swing * np.sin(2 * np.pi * steps / 12)
        rain = (np.random.default_rng(seed).random(len(steps)) < chance).astype(float)
        series = pd.DataFrame({"rain": rain}, index=pd.RangeIndex(len(steps)))
        future = series.index[self.context_length:]
        p = chance[self.context_length:]
        answer_key = Reference(mean=p[:, None], variance=(p * (1 - p))[:, None],
                               covariance=np.diag(p * (1 - p)), index=future, target_names=("rain",))
        task = ForecastTask(series.iloc[:self.context_length], future)
        return Case(task, series.iloc[self.context_length:], answer_key, {"kind": "rain", "seed": seed})


class ShareAtThisPointOfTheCycle:
    """The share of rainy steps so far at the same point of the 12-step cycle."""

    def fit(self, context):
        values = context.to_numpy()[:, 0]
        self.length_ = len(values)
        self.share_ = np.array([values[phase::12].mean() for phase in range(12)])
        return self

    def predict(self, future_index):
        return self.share_[np.arange(self.length_, self.length_ + len(future_index)) % 12]


def rain_pattern(key="rain", fitted=ShareAtThisPointOfTheCycle, build=None, **texts):
    texts = {"title": "Rain or no rain", "aka": "a yes/no outcome", "predictable": "The chance of rain.",
             "unpredictable": "Whether it rains.", **texts}
    if build is None:
        def build(noise, context_length, horizon):
            return RainScenario(0.45 / (1 + noise) ** 0.5, context_length, horizon)
    return Pattern(key, texts["title"], texts["aka"], texts["predictable"], texts["unpredictable"],
                   build=build, fitted=fitted)


@pytest.fixture(scope="module")
def mean_card():
    return Pipeline(NaiveForecaster("mean")).scorecard(patterns=["seasonal", rain_pattern()], instances=4)


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
    alone = Pipeline(NaiveForecaster("mean")).scorecard(patterns=["seasonal"], instances=4)
    after = Pipeline(NaiveForecaster("mean")).scorecard(patterns=[rain_pattern(), "seasonal"], instances=4)
    expected = alone.table.set_index(["pattern", "noise"])["score"]
    for card in (mean_card, after):
        got = card.table.set_index(["pattern", "noise"]).loc["seasonal", "score"]
        assert np.allclose(got.to_numpy(), expected.loc["seasonal"].to_numpy(), rtol=0, atol=1e-12)


def test_own_pattern_gets_the_brier_share_as_its_score():
    pattern = rain_pattern()
    card = Pipeline(NaiveForecaster("mean")).scorecard(patterns=[pattern], instances=3)
    # Recompute from the cases: the answer key's variance p(1 - p) against the squared distance to p.
    unavoidable, avoidable = [], []
    for entry in card.run.cases:
        if entry.scenario_index != 0:  # the low-noise column
            continue
        p = entry.case.reference.mean[:, 0]
        point = entry.forecast.point[:, 0]
        unavoidable.append(np.mean(p * (1 - p)))
        avoidable.append(np.mean((point - p) ** 2))
    expected = 100 * sum(unavoidable) / (sum(unavoidable) + sum(avoidable))
    assert math.isclose(card.table.loc[0, "score"], expected, rel_tol=1e-9)


def test_attainable_is_simulated_only_when_asked_and_with_fitted():
    pipeline = Pipeline(NaiveForecaster("mean"))
    asked = pipeline.scorecard(patterns=[rain_pattern()], instances=2, attainable=True)
    assert asked.table["attainable"].between(85, 95).all()  # about 100 / 1.1 from ten cycles of history
    no_fitted = pipeline.scorecard(patterns=[rain_pattern(fitted=None)], instances=2, attainable=True)
    assert no_fitted.table["attainable"].isna().all()
    not_asked = pipeline.scorecard(patterns=[rain_pattern()], instances=2)
    assert not_asked.table["attainable"].isna().all()


def test_built_in_rows_keep_their_stored_attainable_score_beside_your_own():
    card = Pipeline(NaiveForecaster("mean")).scorecard(patterns=["trend", rain_pattern(fitted=None)], instances=2)
    trend = card.table[card.table["pattern"] == "trend"]["attainable"]
    stored = Pipeline(NaiveForecaster("mean")).scorecard(patterns=["trend"], instances=2).table["attainable"]
    assert trend.notna().all()
    assert np.array_equal(trend.to_numpy(), stored.to_numpy())


def test_a_built_in_pattern_object_counts_as_its_key():
    by_object = Pipeline(NaiveForecaster("mean")).scorecard(patterns=[PATTERNS["trend"]], instances=2)
    by_key = Pipeline(NaiveForecaster("mean")).scorecard(patterns=["trend"], instances=2)
    pd.testing.assert_frame_equal(by_object.table, by_key.table)
    assert by_object.cells[0].definition is PATTERNS["trend"]


def test_own_pattern_can_reuse_a_built_in_recipe():
    gentle = Pattern("gentle_trend", "Gentle trend", "a slow straight line", "The slope.", "The noise.",
                     build=lambda s, n, h: Scenario("trend", n, h, {"slope": 0.01, "sigma": s}))
    card = Pipeline(NaiveForecaster("mean")).scorecard(patterns=[gentle], instances=2,
                                                       context_length=60, horizon=12)
    assert list(card.table["pattern"]) == ["gentle_trend", "gentle_trend"]
    assert card.cells[0].scenario.context_length == 60


def test_scenario_grid_keeps_the_pattern_of_each_row():
    pattern = rain_pattern()
    cells = scenario_grid(["noise", pattern], noise_levels=3)
    assert [cell.pattern for cell in cells] == ["noise"] * 3 + ["rain"] * 3
    assert all(cell.definition is pattern for cell in cells[3:])
    assert all(cell.definition is PATTERNS["noise"] for cell in cells[:3])
    assert all(cell.title == "Rain or no rain" for cell in cells[3:])
    assert all(isinstance(cell.scenario, RainScenario) for cell in cells[3:])


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
    again = Pipeline(NaiveForecaster("mean")).scorecard(patterns=["seasonal", rain_pattern()], instances=4)
    with pytest.warns(UserWarning):  # fewer than 20 instances
        comparison = again.compare(path)
    assert (comparison.table["change"] == "unchanged").all()
    assert set(comparison.table["pattern"]) == {"seasonal", "rain"}
    other_key = Pipeline(NaiveForecaster("mean")).scorecard(patterns=["seasonal", rain_pattern("showers")],
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


def test_grid_refuses_build_or_fitted_that_cannot_be_called():
    with pytest.raises(TypeError, match=r"needs build\(noise, context_length, horizon\)"):
        scenario_grid([rain_pattern(build="not a function")])
    with pytest.raises(TypeError, match="fitted must be a function"):
        scenario_grid([rain_pattern(fitted=ShareAtThisPointOfTheCycle())])


class _NoGenerate:
    kind, context_length, horizon = "rain", 120, 24


def _shifted(**changes):
    def build(noise, context_length, horizon):
        scenario = RainScenario(0.3, context_length, horizon)
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
    (lambda s, n, h: object(), TypeError, "without generate\\(seed\\), kind, context_length, horizon"),
    (_shifted(kind=""), ValueError, "kind must be a nonempty string"),
    (_shifted(context_length=100), ValueError,
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

    class Counting(NaiveForecaster):
        def fit(self, context):
            calls.append(1)
            return super().fit(context)

    with pytest.raises(ValueError, match="no answer key"):
        Pipeline(Counting("mean")).scorecard(
            patterns=["noise", rain_pattern(build=lambda s, n, h: _NoAnswerKey(0.3, n, h))])
    assert calls == []
