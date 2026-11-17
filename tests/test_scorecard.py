"""The scorecard grid, its exact references, seeding, scores, mistakes and checks."""
import math
import re
import tomllib
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from timegym import PATTERNS, Forecast, NaiveForecaster, Pipeline, Scenario, scenario_grid
from timegym.contracts import _index_shift
from timegym.grid import NOISE_HIGH, NOISE_LOW, noise_levels
from timegym.scenarios import scenario_label
from timegym.scorecard import verdict

ROOT = Path(__file__).resolve().parents[1]


# --- Exact references for hidden states -------------------------------------------------

def _dense_reference(kind, p, length, horizon):
    """Condition the full joint Gaussian directly, independently of the Kalman filter."""
    n = length + horizon
    columns = 2 * n + 2  # shocks, two initial-state draws, measurement errors
    sigma, noise = p["sigma"], p.get("noise", 0.0)
    rows, mean = [np.zeros(columns)], np.zeros(n)
    if kind == "ar2":
        phi1, phi2 = p["phi1"], p["phi2"]
        g0 = (1 - phi2) / ((1 + phi2) * ((1 - phi2) ** 2 - phi1 ** 2))
        g1 = phi1 * g0 / (1 - phi2)
        values, vectors = np.linalg.eigh(np.array([[g0, g1], [g1, g0]]) * sigma ** 2)
        factor = vectors * np.sqrt(values)
        rows[0][n:n + 2], previous = factor[0], np.zeros(columns)
        previous[n:n + 2] = factor[1]
        for t in range(1, n):
            row = phi1 * rows[t - 1] + phi2 * previous
            row[t] += sigma
            previous = rows[t - 1]
            rows.append(row)
        mean[:] = p.get("mean", 0.0)
    elif kind == "ar1":
        rows[0][0] = sigma / np.sqrt(1 - p["phi"] ** 2)
        for t in range(1, n):
            row = p["phi"] * rows[t - 1]
            row[t] += sigma
            rows.append(row)
        mean[:] = p.get("mean", 0.0)
    else:
        for t in range(1, n):
            row = rows[t - 1].copy()
            row[t] += sigma
            rows.append(row)
        mean[:] = p.get("drift", 0.0) * np.arange(n)
    loading = np.array(rows)
    loading[np.arange(n), n + 2 + np.arange(n)] = noise
    return mean, loading @ loading.T


@pytest.mark.parametrize("kind, parameters", [
    ("ar1", {"phi": 0.8, "sigma": 1.0, "mean": 2.0, "noise": 1.5}),
    ("ar2", {"phi1": 1.7, "phi2": -0.85, "sigma": 0.5}),
    ("ar2", {"phi1": 1.7, "phi2": -0.85, "sigma": 0.5, "mean": 1.0, "noise": 2.0}),
    ("random_walk", {"drift": 0.1, "sigma": 0.5, "noise": 2.0}),
])
def test_hidden_state_references_equal_brute_force_gaussian_conditioning(kind, parameters):
    length, horizon = 50, 8
    case = Scenario(kind, length, horizon, parameters).generate(seed=3)
    series = np.concatenate([case.task.context["y"], case.observed["y"]])
    mean, joint = _dense_reference(kind, parameters, length, horizon)
    past, future = slice(0, length), slice(length, length + horizon)
    gain = joint[future, past] @ np.linalg.inv(joint[past, past])
    np.testing.assert_allclose(case.reference.mean[:, 0], mean[future] + gain @ (series[past] - mean[past]),
                               atol=1e-9)
    np.testing.assert_allclose(case.reference.covariance, joint[future, future] - gain @ joint[past, future],
                               atol=1e-9)


def test_measurement_noise_hides_the_same_path_at_every_noise_level():
    quiet = Scenario("random_walk", 40, 5, {"sigma": 0.5, "noise": 0.5}).generate(seed=9)
    loud = Scenario("random_walk", 40, 5, {"sigma": 0.5, "noise": 2.0}).generate(seed=9)
    hidden = Scenario("random_walk", 40, 5, {"sigma": 0.5}).generate(seed=9)
    # The measurement errors are the same draws, scaled: noise columns are paired.
    np.testing.assert_allclose(loud.task.context - hidden.task.context,
                               4 * (quiet.task.context - hidden.task.context))
    # Without measurement noise, the long-standing analytic reference is kept.
    assert "noise" not in hidden.manifest["parameters"]
    assert "last_observed" in hidden.manifest["conditioning"]
    assert loud.manifest["conditioning"]["reference_method"] == "kalman_filter"
    # Noise makes the best forecast less certain.
    assert np.all(loud.reference.variance > quiet.reference.variance)


def test_ar2_requires_stationary_weights_and_labels_measurement_noise():
    with pytest.raises(ValueError, match="Stationary ar2"):
        Scenario("ar2", parameters={"phi1": 1.2, "phi2": 0.1})
    with pytest.raises(ValueError, match="Unsupported parameters"):
        Scenario("trend", parameters={"noise": 1.0})
    assert scenario_label("ar2", {"noise": 0.5}) == "Stationary AR(2) + measurement noise"
    assert scenario_label("random_walk", {"drift": 0.1}) == "Random walk with drift"


# --- The grid -----------------------------------------------------------------------------

def test_default_grid_has_ten_patterns_at_low_and_high_noise():
    cells = scenario_grid()
    assert [cell.pattern for cell in cells[::2]] == list(PATTERNS) and len(PATTERNS) == 10
    assert {cell.noise for cell in cells} == {"Low noise", "High noise"}
    assert [cell.noise_sd for cell in cells[:2]] == [NOISE_LOW, NOISE_HIGH]
    for pattern in PATTERNS.values():
        assert pattern.title and pattern.predictable and pattern.unpredictable
    for cell in cells:
        case = cell.scenario.generate(seed=1)
        assert case.task.context.shape == (120, 1) and len(case.task.future_index) == 24


def test_more_noise_levels_fill_the_range_between_the_same_two_ends():
    assert [name for name, _ in noise_levels(3)] == ["Low noise", "Medium noise", "High noise"]
    assert [sd for _, sd in noise_levels(3)] == pytest.approx([0.5, 1.0, 2.0])
    five = noise_levels(5)
    assert five[0] == ("Low noise", NOISE_LOW) and five[-1] == ("High noise", NOISE_HIGH)
    with pytest.raises(ValueError, match="at least two"):
        noise_levels(1)
    with pytest.raises(ValueError, match="patterns"):
        scenario_grid(["trend", "unknown"])


def test_level_shift_happens_inside_the_history():
    cell = scenario_grid(["level_shift"], context_length=120)[0]
    change = cell.scenario.parameters["change_point"]
    assert 0 < change < 120
    case = cell.scenario.generate(seed=0)
    np.testing.assert_allclose(case.reference.mean, 6.0)


# --- Seeds ----------------------------------------------------------------------------------

def test_one_seed_gives_instance_seeds_seed_plus_index():
    run = Pipeline().run([Scenario("trend", 20, 3)], seed=5, instances=3)
    assert [entry.result.seed for entry in run.cases] == [5, 6, 7]
    assert (run.metadata["seed"], run.metadata["instances"]) == (5, 3)
    default = Pipeline().test([Scenario("trend", 20, 3)])
    explicit = Pipeline().test([Scenario("trend", 20, 3)], seeds=[0, 1, 2])
    assert [case.to_dict() for case in default.cases] == [case.to_dict() for case in explicit.cases]
    with pytest.raises(ValueError, match="not both"):
        Pipeline().run([Scenario("trend", 20, 3)], seeds=[1], instances=2)


def test_seed_none_draws_a_seed_and_records_it_for_a_rerun():
    first = Pipeline().run([Scenario("ar1", 20, 3)], seed=None, instances=2)
    again = Pipeline().run([Scenario("ar1", 20, 3)], seed=first.metadata["seed"], instances=2)
    assert first.report.cases[0].to_dict() == again.report.cases[0].to_dict()


# --- Custom models --------------------------------------------------------------------------

class LastValue:
    """The simplest custom model: a plain class returning a DataFrame."""

    def fit(self, context):
        self.last = context.iloc[-1]

    def predict(self, future_index):
        return pd.DataFrame([self.last] * len(future_index), index=future_index)


def test_a_plain_class_returning_a_dataframe_is_a_model():
    report = Pipeline(LastValue()).test([Scenario("random_walk", 30, 4)], instances=2)
    report.assert_passed()
    assert all(row["failed"] == 0 for row in report.checks)


def test_a_series_output_and_unusable_outputs():
    class SeriesModel(LastValue):
        def predict(self, future_index):
            return pd.Series(float(self.last.iloc[0]), index=future_index)

    class ArrayModel(LastValue):
        def predict(self, future_index):
            return np.zeros((len(future_index), 1))

    class DictModel(LastValue):
        def predict(self, future_index):
            return {"y": [0.0] * len(future_index)}

    class Renamed(LastValue):
        def predict(self, future_index):
            return super().predict(future_index).rename(columns={"y": "sales"})

    Pipeline(SeriesModel()).test([Scenario("trend", 20, 3)], instances=1).assert_passed()
    # An array has no dates of its own: its rows are the requested steps, in order.
    Pipeline(ArrayModel()).test([Scenario("trend", 20, 3)], instances=1).assert_passed()
    checks = {row["check"]: row for row in Pipeline(DictModel()).test([Scenario("trend", 20, 3)],
                                                                    instances=1).checks}
    assert checks["runs"]["failed"] == 1
    checks = {row["check"]: row for row in Pipeline(Renamed()).test([Scenario("trend", 20, 3)],
                                                                  instances=1).checks}
    assert checks["series"]["failed"] == 1 and checks["dates"]["passed"] == 1


# --- Answer-key forecasters: perfect, late and on the wrong dates -------------------------

def _answer_keys(cells, seed, instances):
    keys = {}
    for cell in cells:
        for i in range(instances):
            case = cell.scenario.generate(seed=seed + i)
            keys[case.task.context.to_numpy().tobytes()] = case.reference.mean
    return keys


class AnswerKey:
    """Looks up the true conditional mean of the context it is fitted on (tests only)."""

    def __init__(self, keys, lag=0, date_shift=0):
        self.keys, self.lag, self.date_shift = keys, lag, date_shift

    def fit(self, context):
        self.columns = context.columns
        self.mean = self.keys[context.to_numpy().tobytes()]

    def predict(self, future_index):
        values = np.concatenate([self.mean[:1]] * self.lag + [self.mean[:len(self.mean) - self.lag]])
        return pd.DataFrame(values, index=future_index + self.date_shift, columns=self.columns)


def test_the_answer_key_scores_100_everywhere_with_no_mistakes():
    cells = scenario_grid()
    card = Pipeline(AnswerKey(_answer_keys(cells, 0, 3))).scorecard(instances=3)
    assert card.table["score"].min() == pytest.approx(100)
    assert (card.table["mistakes"] == "").all()
    assert len(card.good_at) == 10 and card.fails_at == []


def test_values_one_step_late_are_named_on_the_seasonal_cycle():
    cells = scenario_grid(["seasonal"])
    card = Pipeline(AnswerKey(_answer_keys(cells, 0, 4), lag=1)).scorecard(["seasonal"], instances=4)
    low = card.table.set_index("noise").loc["Low noise"]
    assert low["score"] < 40 and "Values one step late" in low["mistakes"]


def test_wrong_dates_fail_their_check_and_say_how_far_they_moved():
    cells = scenario_grid(["trend"])
    card = Pipeline(AnswerKey(_answer_keys(cells, 0, 2), date_shift=-1)).scorecard(["trend"], instances=2)
    assert (card.table["score"] == 0).all()
    assert card.table["mistakes"].str.contains("Wrong dates: shifted 1 step early").all()
    dates = next(row for row in card.checks if row["check"] == "dates")
    assert (dates["passed"], dates["failed"]) == (0, 4)
    with pytest.raises(AssertionError, match="Forecasts the requested dates: failed in 4 of 4 cases"):
        card.report.assert_passed()


@pytest.mark.parametrize("index, future, expected", [
    (pd.RangeIndex(9, 12), pd.RangeIndex(10, 13), -1),
    (pd.RangeIndex(12, 15), pd.RangeIndex(10, 13), 2),
    (pd.date_range("2024-01-31", periods=3, freq="D"), pd.date_range("2024-02-01", periods=3, freq="D"), -1),
    (pd.period_range("2024-03", periods=3, freq="M"), pd.period_range("2024-02", periods=3, freq="M"), 1),
    (pd.RangeIndex(10, 12), pd.RangeIndex(10, 13), None),
    (pd.Index([10, 11, 15]), pd.RangeIndex(10, 13), None),
])
def test_index_shift_counts_whole_steps(index, future, expected):
    assert _index_shift(index, future) == expected


# --- Simple models: scores agree with what is predictable ---------------------------------

def test_last_value_nails_the_random_walk_and_misses_the_cycle():
    card = Pipeline(NaiveForecaster()).scorecard(["seasonal", "random_walk"], instances=10)
    table = card.table.set_index(["pattern", "noise"])
    assert table.loc[("random_walk", "Low noise"), "score"] >= 90
    assert table.loc[("seasonal", "Low noise"), "score"] < 40
    assert "Misses the cycle" in table.loc[("seasonal", "Low noise"), "mistakes"]
    # Bands describe skill; they are not checks. A "Fails" cell with valid output passes.
    assert table.loc[("seasonal", "Low noise"), "verdict"] == "Fails"
    report = card.report
    assert report.assert_passed() is report


def test_the_historical_mean_forecasts_a_rising_line_too_low():
    card = Pipeline(NaiveForecaster("mean")).scorecard(["trend", "level"], instances=5)
    table = card.table.set_index(["pattern", "noise"])
    assert "Forecasts too low" in table.loc[("trend", "Low noise"), "mistakes"]
    assert table.loc[("level", "High noise"), "score"] >= 90


def test_verdict_bands_follow_the_printed_number():
    assert [verdict(s) for s in (99.6, 89.6, 69.7, 39.4, 0)] == ["Nails it", "Nails it", "Good",
                                                                   "Fails", "Fails"]
    assert verdict(float("nan")) is None


# --- Outputs: image, Markdown, JSON -----------------------------------------------------------

def test_scorecard_draws_saves_and_prints(tmp_path):
    card = Pipeline(NaiveForecaster()).scorecard(["noise", "trend"], noise_levels=3, instances=2, seed=4)
    figure = card.plot()
    assert len(figure.axes) == 3 * (3 + 2)  # header row + two patterns, by text + 3 levels + mistakes
    plt.close(figure)
    assert card.save(tmp_path / "card.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    card.save(tmp_path / "compact.svg", compact=True)
    assert card._repr_png_()[:4] == b"\x89PNG"
    text = str(card)
    assert "Straight-line trend" in text and "Medium noise" in text and "seed 4" in text
    assert "✓ Forecasts the requested dates (12/12)" in text
    assert card.report.to_dict()["metadata"]["seed"] == 4


def test_zooming_into_a_scorecard_row_keeps_its_plain_titles():
    card = Pipeline(NaiveForecaster()).scorecard(["trend"], instances=2)
    forecasts = card.run.plot_forecasts()
    title = forecasts.axes[0].get_title(loc="left")
    assert title == "Straight-line trend · Low noise · instance 1"
    header = " ".join(text.get_text() for text in forecasts.axes[0].texts)
    assert "error vs answer key" in header
    assert "seed" not in title + header and "MSE" not in title + header
    assert forecasts.axes[-1].get_title(loc="left") == "Straight-line trend · High noise · instance 2"
    errors = card.run.plot_errors()
    assert errors.axes[0].get_title(loc="left") == "Straight-line trend · Low noise"
    assert errors.axes[0].get_title(loc="right").endswith("2 scored / 2 evaluated cases")
    assert errors.axes[-1].get_xlabel() == "Error against the answer key (MSE), one score per case"
    plt.close("all")


def test_plain_install_brings_plots_and_the_sklearn_adapter():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    names = {requirement.split(">")[0].split("=")[0] for requirement in project["dependencies"]}
    assert {"matplotlib", "scikit-learn"} <= names
    # The empty extras that 1.2.0 kept for older install commands are gone.
    assert not {"plots", "sklearn"} & project["optional-dependencies"].keys()


def test_scorecard_text_counts_one_of_a_kind_and_compact_headers_fit():
    card = Pipeline(NaiveForecaster()).scorecard(["trend"], noise_levels=3, instances=1)
    assert "1 kind of series x 3 noise levels, 1 instance each" in card.to_markdown()
    figure = card.plot(compact=True)
    texts = [text.get_text() for ax in figure.axes for text in ax.texts]
    texts += [text.get_text() for text in figure.texts]
    assert "Medium noise\n(sd 1)" in texts
    full = [text.get_text() for text in card.plot().texts]
    assert any("1 kind of series × 3 noise levels · 1 instance each" in text for text in full)
    plt.close("all")


def test_the_built_in_last_value_forecaster_is_scored_by_name():
    from timegym import LastValueForecaster

    patterns = ["seasonal", "random_walk"]
    named = Pipeline(LastValueForecaster()).scorecard(patterns, instances=3)
    naive = Pipeline(NaiveForecaster()).scorecard(patterns, instances=3)
    assert str(named).splitlines()[0] == f"## LastValueForecaster: {named.score:.0f}/100"
    assert repr(named).startswith("Scorecard('LastValueForecaster', ")
    assert named.report.metadata["model"] == "LastValueForecaster"
    # One forecaster, two names: the scores are the same.
    pd.testing.assert_frame_equal(named.table, naive.table)


# --- How sure a score is -----------------------------------------------------------------------

@pytest.fixture(scope="module")
def last_value_card():
    return Pipeline(NaiveForecaster()).scorecard(["noise", "trend", "random_walk"], instances=10, seed=3)


def test_the_table_has_an_interval_and_a_settled_flag_after_the_score():
    table = Pipeline(NaiveForecaster()).scorecard(["trend"], instances=3).table
    assert list(table.columns[:9]) == ["pattern", "title", "noise", "noise_sd", "score", "score_low",
                                      "score_high", "verdict", "settled"]
    assert table["settled"].dtype == bool


def test_the_interval_holds_the_score_and_stays_between_0_and_100(last_value_card):
    table = last_value_card.table
    assert (table["score_low"] <= table["score"] + 1e-9).all()
    assert (table["score"] <= table["score_high"] + 1e-9).all()
    assert (table["score_low"] >= 0).all() and (table["score_high"] <= 100).all()


def test_the_same_seed_gives_the_same_interval_and_another_seed_can_differ(last_value_card):
    again = Pipeline(NaiveForecaster()).scorecard(["noise", "trend", "random_walk"], instances=10, seed=3)
    other = Pipeline(NaiveForecaster()).scorecard(["noise", "trend", "random_walk"], instances=10, seed=4)
    columns = ["score_low", "score_high"]
    pd.testing.assert_frame_equal(again.table[columns], last_value_card.table[columns])
    assert not np.allclose(other.table[columns], last_value_card.table[columns])


def test_a_cell_has_the_same_interval_whichever_other_patterns_are_scored(last_value_card):
    alone = Pipeline(NaiveForecaster()).scorecard(["random_walk"], instances=10, seed=3).table
    shared = last_value_card.table[last_value_card.table["pattern"] == "random_walk"]
    np.testing.assert_allclose(alone[["score_low", "score_high"]], shared[["score_low", "score_high"]])


def _interval_by_definition(pairs, valid, level=0.95):
    """The cell score and its interval, written out from the definition one value at a time."""
    from scipy.stats import t as student_t

    pairs, valid, n = np.asarray(pairs, dtype=float), np.asarray(valid, dtype=bool), len(pairs)

    def cell_score(keep):
        best, total = pairs[keep, 0].sum(), pairs[keep].sum()
        return 100 * (1.0 if total == 0 else best / total) * valid[keep].sum() / len(keep)

    def spread(values):
        mean = sum(values) / n
        return math.sqrt((n - 1) / n * sum((value - mean) ** 2 for value in values))

    def logit(x):
        return math.log(x / (100 - x))

    score = cell_score(list(range(n)))
    left_out = [cell_score([j for j in range(n) if j != i]) for i in range(n)]
    above = student_t.ppf(0.5 + level / 2, n - 1)
    below = above * (1 + 1 / math.sqrt(n))  # the low end reaches further than the high end
    if 0 < score < 100 and all(0 < s < 100 for s in left_out):
        centre, wide = logit(score), spread([logit(s) for s in left_out])
        low = 100 / (1 + math.exp(-(centre - below * wide)))
        high = 100 / (1 + math.exp(-(centre + above * wide)))
        scale = "log-odds"
    else:
        wide = spread(left_out)
        low, high = max(score - below * wide, 0.0), min(score + above * wide, 100.0)
        scale = "score"
    return score, min(low, score), max(high, score), scale


# Five series with different avoidable errors, all scored: the log-odds case.
SPREAD_OUT = [(1.0, 0.2), (0.8, 1.5), (1.2, 0.1), (0.9, 3.0), (1.1, 0.7)]


@pytest.mark.parametrize("level", [0.8, 0.95, 0.99])
def test_the_interval_of_a_hand_made_cell_follows_its_definition(level):
    from timegym.scorecard import _interval

    score, low, high, scale = _interval_by_definition(SPREAD_OUT, [True] * 5, level)
    assert scale == "log-odds" and 0 < low < score < high < 100
    assert _interval(SPREAD_OUT, [True] * 5, score, level) == pytest.approx((low, high), rel=1e-9)


@pytest.mark.parametrize("level", [0.5, 0.95, 0.99])
def test_the_interval_reaches_further_below_the_score_than_above_it(level):
    from timegym.scorecard import _interval, _low_end_stretch

    def logit(x):
        return math.log(x / (100 - x))

    assert [_low_end_stretch(n) for n in (4, 25, 100)] == pytest.approx([1.5, 1.2, 1.1])
    score = _interval_by_definition(SPREAD_OUT, [True] * 5)[0]
    low, high = _interval(SPREAD_OUT, [True] * 5, score, level)
    assert (logit(score) - logit(low)) / (logit(high) - logit(score)) == pytest.approx(1 + 1 / math.sqrt(5))
    # On the score scale too: leaving the last series out scores exactly 100.
    pairs = [(1.0, 0.0)] * 3 + [(1.0, 2.0)]
    score, _, _, scale = _interval_by_definition(pairs, [True] * 4)
    low, high = _interval(pairs, [True] * 4, score, 0.2)
    assert scale == "score" and 0 < low < score < high < 100
    assert (score - low) / (high - score) == pytest.approx(1.5)


def test_the_interval_keeps_its_level_when_each_error_is_a_squared_normal_draw():
    from timegym.scorecard import _interval

    # Unavoidable error 1 and mean avoidable error 1: the long-run score is 50.
    errors = np.random.default_rng(20261116).standard_normal((3000, 10)) ** 2
    above = below = 0
    for row in errors:
        low, high = _interval([(1.0, error) for error in row], [True] * 10, 100 / (1 + row.mean()))
        above += low > 50
        below += high < 50
    # Without the longer low end the interval lies above 50 in about 6% of cards, not 2.5%.
    assert above / len(errors) < 0.035
    assert 1 - (above + below) / len(errors) > 0.95


def test_the_interval_of_a_drawn_cell_follows_its_definition(last_value_card):
    from timegym.scorecard import _instance_pairs

    card = last_value_card
    for index, row in card.table.iterrows():
        pairs, valid = _instance_pairs([entry for entry in card.run.cases if entry.scenario_index == index])
        score, low, high, _ = _interval_by_definition(pairs, valid)
        assert score == pytest.approx(row["score"], rel=1e-9)
        assert (row["score_low"], row["score_high"]) == pytest.approx((low, high), rel=1e-9)


@pytest.mark.parametrize("pairs, valid, expected", [
    ([(0.0, 0.0)] * 4, [False] * 4, (0.0, 0.0)),            # every instance failed its output checks
    ([(1.0, 0.0)] * 4, [True] * 4, (100.0, 100.0)),         # a perfect forecaster
    ([(1.0, 0.0)] * 3 + [(1.0, 2.0)], [True] * 4, None),     # leaving the last one out scores exactly 100
    ([(1.0, 0.5)] + [(0.0, 0.0)] * 3, [True] + [False] * 3, None),  # one valid instance among failed ones
    ([(1.0, 0.0), (0.0, 0.0)], [True, False], None),        # two instances: one perfect, one failed
])
def test_a_score_of_0_or_100_in_a_cell_falls_back_to_the_score_scale(pairs, valid, expected):
    from timegym.scorecard import _interval

    score, low, high, scale = _interval_by_definition(pairs, valid)
    assert scale == "score"
    got = _interval(pairs, valid, score)
    assert all(np.isfinite(got)) and 0 <= got[0] <= score <= got[1] <= 100
    assert got == pytest.approx((low, high), rel=1e-9, abs=1e-12)
    if expected is not None:
        assert got == pytest.approx(expected)


def test_fewer_than_two_instances_or_no_score_give_no_interval():
    from timegym.scorecard import _interval

    assert all(np.isnan(_interval([(1.0, 0.5)], [True], 66.7)))
    assert all(np.isnan(_interval([], [], np.nan)))
    assert all(np.isnan(_interval(SPREAD_OUT, [True] * 5, np.nan)))


def test_a_higher_level_gives_an_interval_that_holds_the_lower_one(last_value_card):
    from timegym.scorecard import _instance_pairs, _interval

    card = last_value_card
    cells = [_instance_pairs([entry for entry in card.run.cases if entry.scenario_index == index])
             for index in range(len(card.table))] + [(SPREAD_OUT, [True] * 5)]
    scores = list(card.table["score"]) + [_interval_by_definition(SPREAD_OUT, [True] * 5)[0]]
    for (pairs, valid), score in zip(cells, scores):
        ends = [_interval(pairs, valid, score, level) for level in (0.5, 0.8, 0.9, 0.95, 0.99, 0.999)]
        for (low, high), (wider_low, wider_high) in zip(ends, ends[1:]):
            assert wider_low <= low <= score <= high <= wider_high
        assert ends[-1][1] - ends[-1][0] > ends[0][1] - ends[0][0]
    wide = Pipeline(NaiveForecaster()).scorecard(["noise", "trend", "random_walk"], instances=10, seed=3,
                                                   interval_level=0.99).table
    assert (wide["score_low"] <= card.table["score_low"]).all()
    assert (wide["score_high"] >= card.table["score_high"]).all()
    pd.testing.assert_series_equal(wide["score"], card.table["score"])


def test_the_interval_does_not_depend_on_the_seed_argument_given_the_same_series(monkeypatch,
                                                                               last_value_card):
    import inspect

    from timegym.scorecard import _interval

    assert "seed" not in inspect.signature(_interval).parameters
    original = Pipeline.run

    def same_series(self, scenarios=None, seeds=None, *, seed=0, instances=None, **options):
        run = original(self, scenarios, seeds, seed=3, instances=instances, **options)
        run.metadata["seed"] = seed  # the card records the seed it was given; the series are those of seed 3
        return run

    monkeypatch.setattr(Pipeline, "run", same_series)
    relabelled = Pipeline(NaiveForecaster()).scorecard(["noise", "trend", "random_walk"], instances=10,
                                                        seed=99)
    assert relabelled.seed == 99
    columns = ["score", "score_low", "score_high", "settled"]
    pd.testing.assert_frame_equal(relabelled.table[columns], last_value_card.table[columns])


def test_the_answer_key_is_settled_at_100_everywhere():
    cells = scenario_grid()
    card = Pipeline(AnswerKey(_answer_keys(cells, 0, 3))).scorecard(instances=3)
    assert card.table["score_low"].min() == pytest.approx(100)
    assert card.table["settled"].all() and card.unsettled.empty
    assert "?" not in str(card) and "Run scorecard(" not in str(card)  # no mark, no closing line


def test_settled_means_both_ends_have_the_verdict_and_unsettled_lists_the_rest(last_value_card):
    table = last_value_card.table
    both = [verdict(low) == verdict(high) for low, high in zip(table["score_low"], table["score_high"])]
    assert table["settled"].tolist() == both
    assert last_value_card.unsettled.index.tolist() == table.index[~table["settled"]].tolist()
    assert 0 < len(last_value_card.unsettled) < len(table)  # this card has both kinds


class SometimesBroken(LastValue):
    """Repeats the last value, but returns a missing value when the last value is above zero."""

    def predict(self, future_index):
        forecast = super().predict(future_index)
        if float(self.last.iloc[0]) > 0:
            forecast.iloc[0, 0] = np.nan
        return forecast


def test_failed_output_checks_pull_the_interval_down_and_always_failing_settles_at_0():
    healthy = Pipeline(LastValue()).scorecard(["noise"], instances=10).table
    card = Pipeline(SometimesBroken()).scorecard(["noise"], instances=10)
    broken = card.table
    assert (0 < broken["failed_checks"]).all() and (broken["failed_checks"] < 10).all()
    assert (broken["score_low"] < healthy["score_low"]).all()
    # Scoring the series exactly as drawn gives the score, broken instances included.
    from timegym.scorecard import _instance_pairs, _subset_scores

    pairs, valid = _instance_pairs([entry for entry in card.run.cases if entry.scenario_index == 0])
    as_drawn = _subset_scores(*np.array(pairs).T, np.array(valid), np.arange(10)[None, :])
    assert as_drawn[0] == pytest.approx(card.table.loc[0, "score"])
    cells = scenario_grid(["trend"])
    dates = Pipeline(AnswerKey(_answer_keys(cells, 0, 2), date_shift=-1)).scorecard(["trend"], instances=2)
    assert dates.table[["score_low", "score_high"]].to_numpy().ravel() == pytest.approx([0] * 4)
    assert dates.table["settled"].all()


def test_one_instance_has_no_interval_and_no_settled_verdict():
    card = Pipeline(NaiveForecaster()).scorecard(["trend"], instances=1)
    assert card.table[["score_low", "score_high"]].isna().all().all()
    assert not card.table["settled"].any() and len(card.unsettled) == 2
    assert str(card).endswith("2 of 2 verdicts are not settled at 1 instance. "
                              "Run scorecard(instances=30) to narrow them.")


def test_the_printed_card_marks_unsettled_verdicts_and_says_how_many(last_value_card):
    from timegym.scorecard import _unsettled_line

    text, table = str(last_value_card), last_value_card.table
    row = last_value_card.unsettled.iloc[0]
    assert f"{row['score']:.0f} {row['verdict']}?" in text
    assert "A question mark means the verdict is not settled: the score's 95% interval reaches into another band." in text
    assert text.splitlines()[-2:] == ["", f"{len(last_value_card.unsettled)} of {len(table)} verdicts are not "
                                          "settled at 10 instances. Run scorecard(instances=30) to narrow them."]
    assert _unsettled_line(1, 20, 10) == ("1 of 20 verdicts is not settled at 10 instances. "
                                          "Run scorecard(instances=30) to narrow it.")
    assert _unsettled_line(2, 4, 40).endswith("at 40 instances. Run scorecard(instances=120) to narrow them.")
    assert _unsettled_line(0, 20, 10) == ""


def test_the_picture_marks_unsettled_verdicts_in_both_modes(last_value_card):
    row = last_value_card.unsettled.iloc[0]
    label = f"{row['score']:.0f}  {row['verdict']}?"
    settled = last_value_card.table[last_value_card.table["settled"]].iloc[0]
    for compact in (False, True):
        figure = last_value_card.plot(compact=compact)
        texts = [text.get_text().strip() for ax in figure.axes for text in ax.texts]
        assert label in texts and f"{settled['score']:.0f}  {settled['verdict']}" in texts
        plt.close(figure)


@pytest.mark.parametrize("bad", [True, False, "0.95", float("nan"), float("inf"), 0, 1, 0.0, 1.0, 95, -0.5,
                                 None])
def test_interval_level_must_be_a_number_between_0_and_1(monkeypatch, bad):
    from timegym.scorecard import _scorecard
    from timegym.scorecard import scorecard as deprecated_scorecard

    def never(*args, **kwargs):
        raise AssertionError("the level is checked before any series is drawn")

    monkeypatch.setattr(Pipeline, "run", never)
    message = f"interval_level must be a number between 0 and 1, such as 0.95; got {bad!r}"
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        Pipeline(NaiveForecaster()).scorecard(["trend"], instances=3, interval_level=bad)
    with pytest.raises(ValueError, match="interval_level must be a number between 0 and 1"):
        _scorecard(Pipeline(NaiveForecaster()), ["trend"], instances=3, interval_level=bad)
    with pytest.warns(DeprecationWarning), pytest.raises(ValueError, match="interval_level must be"):
        deprecated_scorecard(Pipeline(NaiveForecaster()), ["trend"], instances=3, interval_level=bad)


def test_the_card_keeps_its_interval_level_and_older_constructions_still_work(last_value_card):
    from timegym.scorecard import Scorecard
    from timegym.scorecard import scorecard as deprecated_scorecard

    assert last_value_card.interval_level == 0.95
    card = Pipeline(NaiveForecaster()).scorecard(["trend"], instances=3, interval_level=np.float32(0.9))
    assert type(card.interval_level) is float and card.interval_level == pytest.approx(0.9)
    with pytest.warns(DeprecationWarning):
        old = deprecated_scorecard(Pipeline(NaiveForecaster()), ["trend"], instances=3, interval_level=0.99)
    assert old.interval_level == 0.99
    fields = last_value_card
    rebuilt = Scorecard(fields.title, fields.table, fields.run, fields.cells, fields.seed, fields.instances)
    assert rebuilt.interval_level == 0.95 and str(rebuilt) == str(last_value_card)


def _caption(figure):
    return " ".join(text.get_text() for text in figure.texts).replace("\n", " ")


@pytest.mark.parametrize("level, words", [(0.9, "90%"), (0.95, "95%"), (0.99, "99%"), (0.995, "99.5%")])
def test_the_level_is_written_as_a_percent(level, words):
    from timegym.scorecard import _percent

    assert _percent(level) == words


def test_the_printed_card_and_the_image_name_the_card_s_level(last_value_card):
    default = "the score's 95% interval reaches into another band."
    figure = last_value_card.plot()
    assert "The line beside each score is its 95% interval on a 0 to 100 scale, ticked at 40, 70 and 90" in (
        _caption(figure))
    plt.close(figure)
    assert default in str(last_value_card)
    wide = Pipeline(NaiveForecaster()).scorecard(["trend"], instances=3, interval_level=0.99)
    text = str(wide)
    assert ("A question mark means the verdict is not settled: the score's 99% interval reaches into "
            "another band.") in text and "95%" not in text
    figure = wide.plot()
    caption = _caption(figure)
    assert "The line beside each score is its 99% interval on a 0 to 100 scale" in caption
    assert "95%" not in caption
    plt.close(figure)


def test_scoring_an_ordinary_card_raises_no_warning():
    from timegym.scorecard import _interval

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        card = Pipeline(NaiveForecaster()).scorecard(["noise", "trend", "seasonal", "random_walk"],
                                                     instances=5, interval_level=0.99)
        str(card)
        plt.close(card.plot())
        for pairs, valid in [([(0.0, 0.0)] * 3, [False] * 3), ([(1.0, 0.0)] * 3, [True] * 3),
                             ([(1.0, 0.0), (0.0, 0.0)], [True, False]), (SPREAD_OUT, [True] * 5),
                             ([(1.0, 1e-300), (1.0, 1e300)], [True, True])]:
            score = _interval_by_definition(pairs, valid)[0]
            _interval(pairs, valid, score, 0.999)
    assert card.table["score_low"].notna().all()


# --- What a fitted model could score ----------------------------------------------------------

def test_every_pattern_has_a_fitted_estimator_and_six_arguments_still_build_a_pattern():
    from timegym.grid import Pattern

    assert all(callable(pattern.fitted) for pattern in PATTERNS.values())
    plain = Pattern("x", "X", "x", "y", "z", lambda s, n, h: Scenario("white_noise", n, h, {"sigma": s}))
    assert plain.fitted is None


@pytest.mark.parametrize("key, instances, floor", [
    ("noise", 10, 95), ("trend", 10, 90), ("seasonal", 10, 90), ("trend_seasonal", 10, 85),
    ("level_shift", 10, 85), ("mean_reverting", 5, 80), ("cycles", 5, 80), ("random_walk", 5, 80),
    ("random_walk_drift", 5, 80),
])
def test_each_fitted_estimator_scores_well_on_its_own_row(key, instances, floor):
    card = Pipeline(PATTERNS[key].fitted()).scorecard([key], instances=instances)
    assert (card.table["score"] >= floor).all() and card.table["failed_checks"].sum() == 0


def test_the_sample_mean_matches_its_expected_score():
    from timegym.scorecard import _ATTAINABLE

    # Forecasting with the mean of 120 values adds 1/120 of the noise variance.
    assert _ATTAINABLE["noise"][0.5] == pytest.approx(100 / (1 + 1 / 120), abs=0.3)


def test_the_stored_table_has_the_20_default_cells_between_80_and_100():
    from timegym.scorecard import _ATTAINABLE

    cells = {(cell.pattern, cell.noise_sd) for cell in scenario_grid()}
    stored = {(key, sd): value for key, values in _ATTAINABLE.items() for sd, value in values.items()}
    assert set(stored) == cells
    assert all(80 <= value <= 100 for value in stored.values())


def test_the_default_card_shows_the_stored_attainable_score_after_settled():
    from timegym.scorecard import _ATTAINABLE

    table = Pipeline(NaiveForecaster()).scorecard(instances=1).table
    assert list(table.columns[8:11]) == ["settled", "attainable", "mistakes"]
    expected = [_ATTAINABLE[row.pattern][row.noise_sd] for row in table.itertuples()]
    assert table["attainable"].to_numpy() == pytest.approx(expected)


def test_a_custom_cell_is_simulated_once_and_agrees_with_a_pipeline_scorecard():
    from timegym.scorecard import _ATTAINABLE_SEED, _attainable_score, _simulated_attainable

    _simulated_attainable.cache_clear()
    first = Pipeline(NaiveForecaster()).scorecard(["trend"], noise_levels=[1.0], instances=1, horizon=6,
                                                  attainable=True)
    again = Pipeline(NaiveForecaster()).scorecard(["trend"], noise_levels=[1.0], instances=1, horizon=6,
                                                  attainable=True)
    info = _simulated_attainable.cache_info()
    assert (info.misses, info.hits) == (1, 1)
    assert again.table["attainable"].iloc[0] == first.table["attainable"].iloc[0]
    cell = scenario_grid(["trend"], [1.0], horizon=6)[0]
    direct = _attainable_score(PATTERNS["trend"], cell.scenario, series=20)
    piped = Pipeline(PATTERNS["trend"].fitted()).scorecard(["trend"], noise_levels=[1.0], horizon=6,
                                                           instances=20, seed=_ATTAINABLE_SEED)
    assert direct == pytest.approx(piped.table["score"].iloc[0], abs=1e-9)


def test_a_custom_cell_is_nan_and_not_simulated_unless_asked():
    from timegym.scorecard import _simulated_attainable

    before = _simulated_attainable.cache_info()
    card = Pipeline(NaiveForecaster()).scorecard(["random_walk"], noise_levels=[0.5, 1.0], instances=1)
    assert card.table["attainable"].iloc[0] == pytest.approx(99.8)  # stored: low noise on the default grid
    assert np.isnan(card.table["attainable"].iloc[1])
    assert _simulated_attainable.cache_info() == before
    figure = card.plot(compact=True)
    assert len(_attainable_marks(figure)) == 1
    plt.close(figure)


def test_card_titles_drop_leading_underscores_from_class_names():
    card = Pipeline(PATTERNS["trend"].fitted()).scorecard(["trend"], instances=1)
    assert card.title == "LeastSquaresLine"
    assert Pipeline(PATTERNS["seasonal"].fitted()).scorecard(["noise"], instances=1).title == \
        "Sinusoid(period=12.0, trend=False)"


def _attainable_marks(figure):
    return [line for ax in figure.axes for line in ax.get_lines() if line.get_gid() == "attainable"]


def test_a_pattern_without_a_fitted_estimator_has_no_attainable_score_or_mark(monkeypatch):
    import dataclasses

    monkeypatch.setitem(PATTERNS, "noise", dataclasses.replace(PATTERNS["noise"], fitted=None))
    card = Pipeline(NaiveForecaster()).scorecard(["noise"], instances=2)
    assert card.table["attainable"].isna().all()
    for compact in (False, True):
        figure = card.plot(compact=compact)
        assert not _attainable_marks(figure)
        plt.close(figure)


def test_the_picture_marks_the_attainable_score_in_both_modes():
    card = Pipeline(NaiveForecaster()).scorecard(["noise", "trend"], instances=1)
    for compact in (False, True):
        figure = card.plot(compact=compact)
        marks = _attainable_marks(figure)
        assert len(marks) == 4 and all(mark.get_marker() == "v" for mark in marks)
        plt.close(figure)


# --- Compare with a saved baseline -----------------------------------------------------------

_COMPARED = ["noise", "trend", "random_walk"]


@pytest.fixture(scope="module")
def last_value_baseline():
    return Pipeline(NaiveForecaster()).scorecard(_COMPARED, instances=20, seed=4)


class CrashesAboveThree(LastValue):
    """The last value, except that it raises when the history ends above 3."""

    def fit(self, context):
        if context.iloc[-1, 0] > 3:
            raise ValueError("intentional crash")
        super().fit(context)


def _with_differences(path, cells, differences):
    """Rewrite a saved baseline so that ``new - baseline`` is ``differences``, scaled, in ``cells``."""
    import json

    record = json.loads(path.read_text(encoding="utf-8"))
    for index in cells:
        avoidable = record["cells"][index]["avoidable"]
        size = 1e-3 * min(avoidable)
        record["cells"][index]["avoidable"] = [value - d * size for value, d in zip(avoidable, differences)]
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


def test_a_saved_baseline_holds_each_series_avoidable_error_and_reads_back_exactly(tmp_path,
                                                                                 last_value_baseline):
    import json

    from timegym.scorecard import _case_numbers

    card = last_value_baseline
    path = tmp_path / "scorecard-baseline.json"
    assert card.save_baseline(path) == path
    record = json.loads(path.read_text(encoding="utf-8"))
    assert (record["format"], record["format_version"]) == ("timegym-scorecard-baseline", 1)
    assert record["timegym_version"] == card.run.metadata["timegym_version"]
    assert record["title"] == card.title
    assert record["grid"] == {"patterns": _COMPARED, "noise_sds": [0.5, 2.0], "context_length": 120,
                              "horizon": 24}
    assert (record["seed"], record["instances"]) == (4, 20)
    assert len(record["cells"]) == 6
    for index, cell in enumerate(record["cells"]):
        row = card.table.loc[index]
        assert (cell["pattern"], cell["noise_sd"], cell["score"]) == (row["pattern"], row["noise_sd"],
                                                                       row["score"])
        entries = sorted((entry for entry in card.run.cases if entry.scenario_index == index),
                         key=lambda entry: entry.seed_index)
        assert cell["avoidable"] == [_case_numbers(entry)[1] for entry in entries]  # exactly
    again = tmp_path / "again.json"
    card.save_baseline(str(again))
    assert again.read_bytes() == path.read_bytes()
    assert list(record) == sorted(record)


def test_a_series_that_crashed_is_stored_as_null(tmp_path):
    card = Pipeline(CrashesAboveThree()).scorecard(["random_walk"], instances=20, seed=4)
    record = __import__("json").loads(card.save_baseline(tmp_path / "b.json").read_text(encoding="utf-8"))
    crashed = [entry.seed_index for entry in card.run.cases
               if entry.scenario_index == 0
               and any(d.name == "model_execution" for d in entry.result.diagnostics)]
    assert crashed and len(crashed) < 20
    assert [i for i, value in enumerate(record["cells"][0]["avoidable"]) if value is None] == crashed


def test_the_same_model_on_the_same_series_is_unchanged_everywhere(tmp_path, last_value_baseline):
    path = last_value_baseline.save_baseline(tmp_path / "scorecard-baseline.json")
    card = Pipeline(NaiveForecaster()).scorecard(_COMPARED, instances=20, seed=4)
    diff = card.compare(path)
    assert (diff.table["change"] == "unchanged").all()
    assert diff.table["p_value"].isna().all() and diff.worse.empty
    assert list(diff.table.columns) == ["pattern", "title", "noise", "noise_sd", "change", "baseline_score",
                                        "score", "baseline_avoidable", "avoidable", "p_value",
                                        "newly_failing", "newly_passing"]
    diff.assert_no_regression()
    assert str(diff).splitlines() == [
        "Compared with scorecard-baseline.json (20 series per cell)",
        "  6 cells  unchanged",
    ]
    assert diff.baseline_version == diff.card_version == card.run.metadata["timegym_version"]


def test_the_printed_comparison_names_a_baseline_saved_with_another_version(tmp_path, last_value_baseline):
    import json

    path = last_value_baseline.save_baseline(tmp_path / "scorecard-baseline.json")
    record = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps({**record, "timegym_version": "0.0.1"}), encoding="utf-8")
    diff = last_value_baseline.compare(path)
    assert diff.baseline_version == "0.0.1"
    assert str(diff).splitlines()[0] == (
        "Compared with scorecard-baseline.json (20 series per cell; saved with timegym 0.0.1)")


def test_computed_noise_levels_that_differ_in_the_last_digit_are_the_same_grid(tmp_path):
    import json

    card = Pipeline(NaiveForecaster()).scorecard(["noise"], noise_levels=3, instances=20, seed=4)
    path = card.save_baseline(tmp_path / "scorecard-baseline.json")
    record = json.loads(path.read_text(encoding="utf-8"))
    nudged = [sd * (1 + 2e-16) for sd in record["grid"]["noise_sds"]]
    record["grid"]["noise_sds"] = nudged
    for cell, sd in zip(record["cells"], nudged):
        cell["noise_sd"] = sd
    path.write_text(json.dumps(record), encoding="utf-8")
    assert (card.compare(path).table["change"] == "unchanged").all()


def test_a_path_and_a_scorecard_give_the_same_comparison(tmp_path, last_value_baseline):
    path = last_value_baseline.save_baseline(tmp_path / "scorecard-baseline.json")
    card = Pipeline(NaiveForecaster("mean")).scorecard(_COMPARED, instances=20, seed=4)
    from_file, from_card = card.compare(path), card.compare(last_value_baseline)
    pd.testing.assert_frame_equal(from_file.table, from_card.table)
    assert from_file.baseline == "scorecard-baseline.json"  # the file's name, not its folder
    assert from_card.baseline == f"the baseline scorecard {last_value_baseline.title!r}"


def test_each_cell_is_unchanged_changed_no_worse_or_changed_and_worse(last_value_baseline):
    # The mean of the history is better than the last value on pure noise and worse on a random walk.
    card = Pipeline(NaiveForecaster("mean")).scorecard(_COMPARED, instances=20, seed=4)
    diff = card.compare(last_value_baseline)
    change = diff.table.set_index(["pattern", "noise"])["change"]
    assert change["noise", "Low noise"] == change["noise", "High noise"] == "changed, no worse"
    assert change["random_walk", "Low noise"] == change["random_walk", "High noise"] == "changed and worse"
    assert diff.table["p_value"].notna().all()
    worse = diff.worse
    assert set(worse["pattern"]) >= {"random_walk"}
    assert (worse["avoidable"] > worse["baseline_avoidable"]).all()
    lines = str(diff).splitlines()
    assert any(line.strip().startswith("Random walk: shocks never fade, low noise") and "p = " in line
               for line in lines)


def test_a_series_that_newly_crashes_is_worse_and_one_that_newly_runs_is_not(last_value_baseline):
    crashing = Pipeline(CrashesAboveThree()).scorecard(_COMPARED, instances=20, seed=4)
    diff = crashing.compare(last_value_baseline)
    row = diff.table.set_index(["pattern", "noise"]).loc["random_walk", "Low noise"]
    assert row["newly_failing"] > 0 and row["newly_passing"] == 0
    assert np.isnan(row["p_value"])  # the series scored in both did not move
    assert row["change"] == "changed and worse"
    failing = int(row["newly_failing"])
    says = f"{failing} series now {'crashes or fails' if failing == 1 else 'crash or fail'} an output check"
    assert says in str(diff)
    with pytest.raises(AssertionError, match=says):
        diff.assert_no_regression()
    back = last_value_baseline.compare(crashing)
    row = back.table.set_index(["pattern", "noise"]).loc["random_walk", "Low noise"]
    assert row["newly_passing"] > 0 and row["newly_failing"] == 0
    assert row["change"] == "changed, no worse" and np.isnan(row["p_value"])
    back.assert_no_regression()


def test_holm_counts_only_the_cells_that_changed(tmp_path, last_value_baseline):
    # Distinct sizes, the eight smallest negative: a p-value between 0.05 / 3 and 0.05 (about 0.03).
    differences = np.sqrt(np.arange(2, 22)) * np.where(np.arange(20) < 8, -1.0, 1.0)
    card = Pipeline(NaiveForecaster()).scorecard(_COMPARED, instances=20, seed=4)
    one = _with_differences(last_value_baseline.save_baseline(tmp_path / "one.json"), [2], differences)
    diff = card.compare(one)
    p = diff.table.loc[2, "p_value"]
    assert 0.05 / 3 < p <= 0.05
    # One changed cell is a family of one, so p <= 0.05 rejects; 0.05 / 6 cells would not.
    assert list(diff.table["change"]) == ["unchanged"] * 2 + ["changed and worse"] + ["unchanged"] * 3
    three = _with_differences(last_value_baseline.save_baseline(tmp_path / "three.json"), [0, 2, 4],
                              differences)
    diff = card.compare(three)
    # Three changed cells: Holm needs the smallest p-value at most 0.05 / 3, and none is.
    assert diff.table.loc[[0, 2, 4], "p_value"].tolist() == pytest.approx([p] * 3)
    assert diff.table.loc[[0, 2, 4], "change"].eq("changed, no worse").all()
    assert diff.table.loc[[1, 3, 5], "change"].eq("unchanged").all()
    diff.assert_no_regression()


def test_comparison_results_stay_as_they_were_before_sharing_the_sign_flips(last_value_baseline):
    # Recorded before the sign flips and Holm's correction moved to timegym.diagnostics, which
    # compare_reports shares: the move must not change a single p-value or verdict.
    diff = Pipeline(NaiveForecaster("mean")).scorecard(_COMPARED, instances=20, seed=4).compare(
        last_value_baseline)
    # Within two flips of 9,999: roundoff on another platform may move a flipped mean past a tie.
    assert diff.table["p_value"].tolist() == pytest.approx(
        [0.9999, 0.9999, 0.0001, 0.0001, 0.0002, 0.0056], abs=2e-4)
    assert diff.table["change"].tolist() == ["changed, no worse"] * 2 + ["changed and worse"] * 4


def test_holm_steps_down_and_keeps_the_given_order():
    from timegym.scorecard import _holm

    np.testing.assert_array_equal(_holm([0.04, 0.01, 0.20, 0.02], 0.05), [False, True, False, False])
    np.testing.assert_array_equal(_holm([0.05, 0.001, 0.01, 0.016], 0.05), [True, True, True, True])


def test_a_p_value_recomputed_by_hand_from_the_stored_numbers(tmp_path, last_value_baseline):
    import json

    path = last_value_baseline.save_baseline(tmp_path / "scorecard-baseline.json")
    card = Pipeline(NaiveForecaster("mean")).scorecard(_COMPARED, instances=20, seed=4)
    diff = card.compare(path)
    index = 4  # random walk, low noise
    old = json.loads(path.read_text(encoding="utf-8"))["cells"][index]["avoidable"]
    new = [entry for entry in card.run.cases if entry.scenario_index == index]
    new = [n for _, n in sorted((entry.seed_index, entry.result.metrics["mean_reference_mse"]) for entry in new)]
    d = np.array(new) - np.array(old)
    flips = np.random.default_rng([4, len(d)]).choice([-1.0, 1.0], size=(9999, len(d)))
    p = (1 + np.sum((flips * d).mean(axis=1) >= d.mean())) / 10000
    assert diff.table.loc[index, "p_value"] == pytest.approx(p)
    assert diff.table.loc[index, "baseline_avoidable"] == pytest.approx(np.mean(old))
    assert diff.table.loc[index, "avoidable"] == pytest.approx(np.mean(new))


def test_p_values_repeat_and_do_not_depend_on_the_other_patterns(last_value_baseline):
    card = Pipeline(NaiveForecaster("mean")).scorecard(_COMPARED, instances=20, seed=4)
    first, second = card.compare(last_value_baseline), card.compare(last_value_baseline)
    pd.testing.assert_frame_equal(first.table, second.table)
    alone = Pipeline(NaiveForecaster("mean")).scorecard(["random_walk"], instances=20, seed=4).compare(
        Pipeline(NaiveForecaster()).scorecard(["random_walk"], instances=20, seed=4))
    shared = first.table[first.table["pattern"] == "random_walk"]
    np.testing.assert_array_equal(alone.table["p_value"], shared["p_value"])


def test_assert_no_regression_names_the_worse_cells_with_their_numbers(last_value_baseline):
    diff = Pipeline(NaiveForecaster("mean")).scorecard(_COMPARED, instances=20, seed=4).compare(
        last_value_baseline)
    with pytest.raises(AssertionError) as raised:
        diff.assert_no_regression()
    message = str(raised.value)
    assert message.startswith(f"Worse than the baseline scorecard {last_value_baseline.title!r} in ")
    from timegym.scorecard import _error

    for row in diff.worse.itertuples():
        assert f"{row.title}, {row.noise.lower()}: avoidable error {_error(row.baseline_avoidable)} -> " in message
    assert (_error(2.0), _error(0.0455), _error(100.0), _error(np.nan)) == ("2.00", "0.0455", "100", "none scored")
    assert "Pure noise" not in message


@pytest.mark.parametrize("settings, says", [
    ({"patterns": ["noise", "trend"]}, "patterns"),
    ({"noise_levels": 3}, "noise standard deviations"),
    ({"context_length": 60}, "history length"),
    ({"horizon": 12}, "horizon"),
    ({"seed": 5}, "seed 5 here, 4 in the baseline"),
    ({"instances": 25}, "instances 25 here, 20 in the baseline"),
])
def test_a_baseline_on_other_series_is_refused_with_what_differs(tmp_path, last_value_baseline, settings,
                                                                 says):
    path = last_value_baseline.save_baseline(tmp_path / "scorecard-baseline.json")
    arguments = {"patterns": _COMPARED, "instances": 20, "seed": 4, **settings}
    card = Pipeline(NaiveForecaster()).scorecard(arguments.pop("patterns"), **arguments)
    for baseline in (path, last_value_baseline):
        with pytest.raises(ValueError, match=f"other series.*{says}"):
            card.compare(baseline)


def test_a_file_that_is_not_a_baseline_or_is_newer_is_refused(tmp_path, last_value_baseline):
    import json

    card = last_value_baseline
    other = tmp_path / "report.json"
    other.write_text(json.dumps({"schema_version": 2}), encoding="utf-8")
    with pytest.raises(ValueError, match="is not a TimeGym scorecard baseline"):
        card.compare(other)
    broken = tmp_path / "broken.json"
    broken.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="is not a TimeGym scorecard baseline"):
        card.compare(broken)
    newer = card.save_baseline(tmp_path / "newer.json")
    record = json.loads(newer.read_text(encoding="utf-8"))
    newer.write_text(json.dumps({**record, "format_version": 2}), encoding="utf-8")
    with pytest.raises(ValueError, match="format version 2.*reads version 1"):
        card.compare(newer)
    short = card.save_baseline(tmp_path / "short.json")
    record["cells"][0]["avoidable"] = record["cells"][0]["avoidable"][:-1]
    short.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="inconsistent"):
        card.compare(short)
    reordered = card.save_baseline(tmp_path / "reordered.json")
    record = json.loads(reordered.read_text(encoding="utf-8"))
    reordered.write_text(json.dumps({**record, "cells": record["cells"][::-1]}), encoding="utf-8")
    with pytest.raises(ValueError, match="inconsistent"):
        card.compare(reordered)
    with pytest.raises(TypeError, match="baseline"):
        card.compare(card.table)
    with pytest.raises(ValueError, match="alpha"):
        card.compare(card, alpha=1.5)


def test_fewer_than_20_series_warn_that_only_large_regressions_are_found():
    small = Pipeline(NaiveForecaster()).scorecard(["trend"], instances=10)
    with pytest.warns(UserWarning, match="10 series per cell finds only large regressions"):
        diff = small.compare(small)
    assert (diff.table["change"] == "unchanged").all()
