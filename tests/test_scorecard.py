"""The scorecard grid, its exact references, seeding, scores, mistakes and checks."""
import tomllib
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
    # Old install commands keep working.
    assert project["optional-dependencies"]["plots"] == project["optional-dependencies"]["sklearn"] == []


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
