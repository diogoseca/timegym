"""Output checks of your own: the range check `within`, your own functions, and how reports count them."""

import numpy as np
import pandas as pd
import pytest

from timegym import (
    EvaluatedCase,
    Forecast,
    ForecastRun,
    GaussianAR1Forecaster,
    LastValueForecaster,
    Pipeline,
    Scenario,
    TestReport,
    cases_from_series,
)
from timegym.checks import within
from timegym.diagnostics import _describe_checks, evaluate_forecast
from timegym.reporting import CHECKS


def rainfall(periods=120, seed=1):
    """Daily rainfall in mm: dry on most days, a gamma amount on the rest."""
    rng = np.random.default_rng(seed)
    days = pd.date_range("2024-01-01", periods=periods, freq="D")
    wet = rng.random(periods) < 0.4
    return pd.Series(np.where(wet, rng.gamma(2.0, 3.0, periods), 0.0), index=days, name="rain_mm")


def store_sales(periods=120, seed=3):
    """Coats sold by two branches, with their total."""
    rng = np.random.default_rng(seed)
    days = pd.date_range("2024-01-01", periods=periods, freq="D")
    north, south = rng.poisson(3, periods), rng.poisson(2, periods)
    return pd.DataFrame({"north": north, "south": south, "total": north + south}, index=days).astype(float)


def rain_cases():
    return cases_from_series(rainfall(), origins=range(60, 113, 7), horizon=7, name="rain")


def one_case():
    return rain_cases()[0]


def forecast_for(case, point=None, quantiles=None, samples=None, levels=(0.1, 0.5, 0.9)):
    """A forecast on the case's dates and series, with whichever outputs are given."""
    return Forecast(index=case.task.future_index, target_names=case.task.target_names,
                    point=point, quantiles=quantiles, samples=samples,
                    levels=levels if quantiles is not None else ())


def totals_add_up(predicted, target_names):
    """North and south add up to the total."""
    north, south, total = (predicted[:, target_names.index(name)] for name in ("north", "south", "total"))
    gap = float(np.max(np.abs(north + south - total)))
    return gap <= 1e-9 or f"the total is off by up to {gap:.2f} coats"


def findings(result, name):
    return [d for d in result.diagnostics if d.name == name]


# ---------------------------------------------------------------- within

def test_within_titles_say_the_range():
    assert within(0, 1).__doc__ == "Values between 0 and 1"
    assert within(0, None).__doc__ == "Values at least 0"
    assert within(None, 100).__doc__ == "Values at most 100"
    assert within(0.5, 2.25).__doc__ == "Values between 0.5 and 2.25"
    assert within(lower={"rain_mm": 0}).__doc__ == "Values in each series' allowed range"
    assert within(0, 1).__name__ == "within"
    assert repr(within(0, None)) == "within(0.0, None)"


@pytest.mark.parametrize("lower, upper, message", [
    (None, None, "needs a lower or an upper bound"),
    ({"rain_mm": None}, None, "needs a lower or an upper bound"),
    (1, 0, "the lower bound 1 is above the upper bound 0"),
    ({"humidity": 50}, {"humidity": 10}, "for 'humidity'"),
    (float("nan"), None, "finite number"),
    (None, float("inf"), "finite number"),
    (True, None, "finite number"),
    ("0", None, "finite number"),
    ({}, None, "empty dictionary"),
    ({1: 0}, None, "series names"),
])
def test_within_refuses_ranges_that_cannot_work(lower, upper, message):
    with pytest.raises(ValueError, match=message):
        within(lower, upper)


def test_within_passes_values_on_the_bounds_and_names_what_falls_outside():
    case = one_case()
    inside = forecast_for(case, point=np.zeros((7, 1)))
    assert evaluate_forecast(case, inside, checks=[within(0, None)]).diagnostics[-1].status == "PASS"
    point = np.array([[0.0], [-0.5], [-2.0], [1.0], [3.0], [12.0], [0.0]])
    quantiles = np.stack([point - 1.0, point, point + 1.0], axis=-1)
    samples = np.stack([point, point + 20.0])
    result = evaluate_forecast(case, forecast_for(case, point, quantiles, samples), checks=[within(0, 10)])
    (finding,) = findings(result, "check_within")
    assert finding.status == "FAIL" and finding.category == "correctness"
    assert finding.evidence == {"title": "Values between 0 and 10"}
    assert finding.message == (
        "point forecast: 2 values below 0 (lowest -2); point forecast: 1 value above 10 (highest 12); "
        "quantiles: 7 values below 0 (lowest -3); quantiles: 3 values above 10 (highest 13); and 2 more")


def test_within_gives_each_series_its_own_range():
    sales = store_sales()
    case = cases_from_series(sales, origins=[60], horizon=7, name="coats")[0]
    point = np.tile([[1.0, -1.0, 0.0]], (7, 1))  # the south branch sells -1 coats a day
    forecast = forecast_for(case, point=point)
    per_series = within(lower={"north": 0, "south": 0}, upper={"north": 5})
    (finding,) = findings(evaluate_forecast(case, forecast, checks=[per_series]), "check_within")
    assert finding.status == "FAIL"
    assert finding.message == "south, point forecast: 7 values below 0 (lowest -1)"
    north_only = within(lower={"north": 0})  # south and the total are open
    assert findings(evaluate_forecast(case, forecast, checks=[north_only]), "check_within")[0].status == "PASS"


def test_within_naming_a_series_the_forecast_lacks_is_an_error():
    case = one_case()
    result = evaluate_forecast(case, forecast_for(case, point=np.ones((7, 1))),
                               checks=[within(lower={"rain_m": 0})])
    (finding,) = findings(result, "check_within")
    assert finding.status == "ERROR"
    assert "within names series the forecast doesn't have: 'rain_m'; it has 'rain_mm'" in finding.message
    assert finding.evidence["title"] == "Values in each series' allowed range"
    assert finding.evidence["exception_type"] == "ValueError"


# ---------------------------------------------------------------- your own checks

def test_own_check_outcomes_become_findings():
    case = one_case()
    forecast = forecast_for(case, point=np.ones((7, 1)))

    def passes(predicted):
        """Always fine."""
        return np.bool_(True)

    def says_why(predicted):
        """Never fine."""
        return "  too wet  "

    def returns_false(predicted):
        """A plain no."""
        return False

    def forgets_to_return(predicted):
        """Forgot the return."""

    def raises(predicted):
        """Breaks."""
        raise RuntimeError("boom")

    checks = [passes, says_why, returns_false, forgets_to_return, raises]
    result = evaluate_forecast(case, forecast, checks=checks)
    found = {d.name: (d.status, d.message) for d in result.diagnostics if d.name.startswith("check_")}
    assert found == {
        "check_passes": ("PASS", "Always fine"),
        "check_says_why": ("FAIL", "too wet"),
        "check_returns_false": ("FAIL", "A plain no: the check returned False."),
        "check_forgets_to_return": ("ERROR", "A check must return True, or a message saying what is wrong; got None."),
        "check_raises": ("ERROR", "The check raised RuntimeError: boom"),
    }
    assert [d.name for d in result.diagnostics if d.name.startswith("check_")] == [f"check_{f.__name__}" for f in checks]


def test_a_check_gets_what_its_argument_names_ask_for():
    case = one_case()
    seen = {}

    def everything(observed, predicted, history, target_names, quantiles=None, samples=None, levels=None):
        seen.update(observed=observed.shape, predicted=predicted.shape, history=history.shape,
                    target_names=target_names, quantiles=quantiles, samples=samples, levels=levels)
        return True

    evaluate_forecast(case, forecast_for(case, point=np.ones((7, 1))), checks=[everything])
    assert seen == {"observed": (7, 1), "predicted": (7, 1), "history": (60, 1), "target_names": ("rain_mm",),
                    "quantiles": None, "samples": None, "levels": None}


def test_a_check_needing_an_output_the_forecast_lacks_is_unsupported_and_not_counted():
    def bands_above_zero(quantiles):
        """Bands stay above zero."""
        return bool((quantiles >= 0).all()) or "a band dips below zero"

    report = Pipeline(LastValueForecaster()).test(cases=rain_cases(), checks=[bands_above_zero])
    statuses = {d.status for case in report.cases for d in case.diagnostics if d.name == "check_bands_above_zero"}
    assert statuses == {"UNSUPPORTED"}
    finding = findings(report.cases[0], "check_bands_above_zero")[0]
    assert finding.evidence == {"title": "Bands stay above zero", "needs": ["quantiles"]}
    assert all(row["check"] != "bands_above_zero" for row in report.checks)
    report.assert_passed()


def test_titles_come_from_the_first_line_of_the_docstring():
    def no_doc(predicted):
        return True

    def ends_with_dots(predicted):
        """Wait for it..."""
        return True

    records = _describe_checks({"rain_range": within(0, None), "plain": no_doc, "dots": ends_with_dots,
                                "sum": totals_add_up})
    assert [record["title"] for record in records] == [
        "Values at least 0", "plain", "Wait for it...", "North and south add up to the total"]


@pytest.mark.parametrize("checks, error, message", [
    (within(0, None), TypeError, "list or a dictionary"),
    ("within", TypeError, "list or a dictionary"),
    (["within"], TypeError, "must be a function"),
    ([lambda predicted: True], ValueError, "no name of its own"),
    ([within(0, None), within(None, 1)], ValueError, "asked for twice"),
    ({"runs": within(0, None)}, ValueError, "built-in output check"),
    ({"": within(0, None)}, ValueError, "nonempty string"),
])
def test_requests_that_cannot_work_are_refused_before_anything_runs(checks, error, message):
    with pytest.raises(error, match=message):
        Pipeline(LastValueForecaster()).test(cases=rain_cases(), checks=checks)


@pytest.mark.parametrize("name", list(CHECKS))
def test_the_names_of_built_in_checks_are_kept_for_them(name):
    with pytest.raises(ValueError, match="built-in output check"):
        _describe_checks({name: within(0, None)})


def test_a_check_asking_for_an_input_timegym_cannot_give_is_refused():
    def needs_weather(predicted, weather):
        return True

    with pytest.raises(TypeError, match="check 'needs_weather' needs 'weather'.*target_names"):
        _describe_checks([needs_weather])


def test_checks_run_only_on_forecasts_that_pass_the_built_in_checks():
    case = one_case()
    calls = []

    def counted(predicted):
        """Counted."""
        calls.append(1)
        return True

    broken = forecast_for(case, point=np.full((7, 1), np.nan))
    result = evaluate_forecast(case, broken, checks=[counted])
    assert not calls and not findings(result, "check_counted")


# ---------------------------------------------------------------- reports

def test_report_checks_and_assert_passed_include_your_checks():
    cases = cases_from_series(store_sales(), origins=range(60, 113, 7), horizon=7, name="coats")
    checks = [within(0, None), totals_add_up]
    good = Pipeline(LastValueForecaster()).test(cases=cases, checks=checks)
    bad = Pipeline(GaussianAR1Forecaster()).test(cases=cases, checks=checks)
    assert good.checks[-2:] == [
        {"check": "within", "title": "Values at least 0", "passed": 8, "failed": 0},
        {"check": "totals_add_up", "title": "North and south add up to the total", "passed": 8, "failed": 0},
    ]
    assert [row["check"] for row in good.checks[:-2]] == ["runs", "dates", "series", "shape", "finite"]  # in CHECKS order
    good.assert_passed()
    assert [row["failed"] for row in bad.checks[-2:]] == [8, 8]
    with pytest.raises(AssertionError) as raised:
        bad.assert_passed()
    lines = str(raised.value).splitlines()
    assert lines[1:3] == ["Values at least 0: failed in 8 of 8 cases",
                          "North and south add up to the total: failed in 8 of 8 cases"]
    assert "check_totals_add_up: the total is off by up to" in str(raised.value)


def test_a_saved_report_loads_and_counts_checks_without_their_functions():
    cases = cases_from_series(store_sales(), origins=range(60, 113, 7), horizon=7, name="coats")
    report = Pipeline(GaussianAR1Forecaster()).test(cases=cases, checks={"sales_range": within(0, None),
                                                                        "sum": totals_add_up})
    loaded = TestReport.from_json(report.to_json())
    assert loaded.checks == report.checks
    assert [row["title"] for row in loaded.checks[-2:]] == ["Values at least 0", "North and south add up to the total"]
    with pytest.raises(AssertionError, match="North and south add up to the total: failed in 8 of 8"):
        loaded.assert_passed()
    first, second = loaded.metadata["checks"]
    assert first == {"name": "sales_range", "title": "Values at least 0", "function": "timegym.checks.within",
                     "settings": {"lower": 0.0, "upper": None}, "replayable": True}
    assert second["function"].endswith(".totals_add_up")
    assert {key: second[key] for key in ("name", "title", "settings", "replayable")} == {
        "name": "sum", "title": "North and south add up to the total", "settings": None, "replayable": False}


def test_no_checks_leave_reports_as_they_were():
    cases = rain_cases()
    plain = Pipeline(LastValueForecaster()).test(cases=cases)
    assert "checks" not in plain.metadata
    assert not any(d.name.startswith("check_") for case in plain.cases for d in case.diagnostics)
    empty = Pipeline(LastValueForecaster()).test(cases=cases, checks=[])
    assert empty.metadata["checks"] == []
    assert [case.to_dict() for case in empty.cases] == [case.to_dict() for case in plain.cases]


def test_forecasts_made_earlier_get_the_checks_too():
    cases = rain_cases()
    made = [np.full(7, -1.0) if position == 2 else np.ones(7) for position in range(len(cases))]
    run = ForecastRun.from_forecasts(cases, made, checks=[within(0, None)])
    assert run.report.checks[-1] == {"check": "within", "title": "Values at least 0", "passed": 7, "failed": 1}
    assert run.metadata["checks"][0]["function"] == "timegym.checks.within"


def test_a_case_failing_your_check_counts_as_newly_failing_in_a_comparison():
    cases = rain_cases()
    baseline = Pipeline(LastValueForecaster()).test(cases=cases, checks=[within(0, None)])
    candidate = Pipeline(GaussianAR1Forecaster()).test(cases=cases, checks=[within(0, None)])
    table = candidate.compare(baseline, metric="mae").table
    assert table.loc[0, "newly_failing"] == len(cases)


def test_scorecard_names_a_failed_check_of_your_own():
    from timegym.scorecard import _check_problems

    case = one_case()
    forecast = forecast_for(case, point=np.full((7, 1), -1.0))
    result = evaluate_forecast(case, forecast, checks=[within(0, None)])

    entry = EvaluatedCase(case, forecast, result, 0, 0)
    assert _check_problems([entry]) == ["Fails check: values at least 0 (1 of 1)"]


# ---------------------------------------------------------------- replay

def test_replay_runs_within_again_and_lists_your_own_checks():
    scenario = Scenario("ar1", history_length=60, horizon=6)

    def positive(predicted):
        """Every value above zero."""
        return bool((predicted > 0).all()) or "some values at or below zero"

    report = Pipeline(LastValueForecaster()).test([scenario], seeds=[0, 1],
                                                     checks={"range": within(-100, 100), "positive": positive})
    replay = report.replay_case(1).metadata["replay"]
    assert replay["reproduced"] and replay["diagnostics_match"]
    assert replay["checks_not_rerun"] == ["positive"]
    assert "not_recomputed" not in replay  # no scores were asked for
    replayed = report.replay_case(1)
    assert [d.name for d in replayed.cases[0].diagnostics if d.name.startswith("check_")] == ["check_range"]

    plain = Pipeline(LastValueForecaster()).test([scenario], seeds=[0])
    assert "checks_not_rerun" not in plain.replay_case(0).metadata["replay"]


def test_replay_runs_within_from_its_saved_bounds():
    scenario = Scenario("ar1", history_length=60, horizon=6)
    report = Pipeline(LastValueForecaster()).test([scenario], seeds=[0], checks=[within(-100, 100)])
    assert report.replay_case(0).metadata["replay"]["diagnostics_match"]
    payload = report.to_dict()
    payload["metadata"]["checks"][0]["settings"] = {"lower": 100.0, "upper": 200.0}
    edited = TestReport.from_dict(payload)
    replayed = edited.replay_case(0)
    assert findings(replayed.cases[0], "check_within")[0].status == "FAIL"
    assert not replayed.metadata["replay"]["diagnostics_match"]
