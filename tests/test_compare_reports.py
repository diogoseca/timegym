"""Comparing forecasters on the same cases: paired sign flips, clusters, groups and many candidates."""

import numpy as np
import pandas as pd
import pytest

from timegym import (
    CaseResult,
    DiagnosticResult,
    LastValueForecaster,
    MeanForecaster,
    Pipeline,
    SeasonalLastValueForecaster,
    TestReport,
    cases_from_series,
    compare_reports,
)
from timegym.diagnostics import ReportComparison, _holm, _holm_adjusted, _paired_flip_test


def report(values, manifests=None, *, metric="mae", model=None, crashed=()):
    """A report with one metric value per case; ``None`` leaves a case without it, ``crashed`` makes it crash."""
    cases = []
    for index, value in enumerate(values):
        diagnostics, metrics = [], {}
        if index in crashed:
            diagnostics = [DiagnosticResult("model_execution", "ERROR", "execution", "boom", {})]
        elif value is not None:
            metrics = {metric: float(value)}
        manifest = {"case": index} if manifests is None else manifests[index]
        cases.append(CaseResult("case", None, diagnostics, metrics, manifest))
    return TestReport(cases, {} if model is None else {"model": model})


def noisy(n, seed=0, shift=0.0):
    return 5.0 + shift + np.random.default_rng(seed).normal(0.0, 1.0, n)


def test_better_worse_and_no_clear_difference():
    base = noisy(40)
    gain = np.random.default_rng(1).uniform(0.2, 0.6, 40)  # every case improves
    better = report(base - gain).compare(report(base), metric="mae")
    assert better.table["verdict"].tolist() == ["better"] and len(better.better) == 1
    assert better.table.loc[0, "p_value"] == pytest.approx(1 / 10_000)  # only the observed signs
    worse = report(base + gain).compare(report(base), metric="mae")
    assert worse.table["verdict"].tolist() == ["worse"] and len(worse.worse) == 1
    mixed = np.random.default_rng(2).normal(0.0, 0.3, 40)
    unclear = report(base + mixed).compare(report(base), metric="mae")
    assert unclear.table["verdict"].tolist() == ["no clear difference"]
    assert unclear.table.loc[0, "p_value"] > 0.05
    row = better.table.iloc[0]
    assert row["cases"] == row["clusters"] == 40
    assert row["mean_difference"] == pytest.approx(-gain.mean())
    assert row["baseline_mean"] == pytest.approx(base.mean())


def test_higher_is_better_reads_the_same_numbers_the_other_way():
    base = noisy(40)
    up = base + np.random.default_rng(1).uniform(0.2, 0.6, 40)
    assert report(up).compare(report(base), metric="mae").table["verdict"][0] == "worse"
    assert report(up).compare(report(base), metric="mae",
                              higher_is_better=True).table["verdict"][0] == "better"


def test_a_p_value_recomputed_by_hand():
    base = noisy(30)
    d = np.random.default_rng(5).normal(0.15, 0.5, 30)
    diff = report(base + d).compare(report(base), metric="mae", seed=7)
    d = (base + d) - base
    flips = np.random.default_rng([7, 30]).choice([-1.0, 1.0], size=(9_999, 30))
    p = (1 + np.sum(np.abs(flips @ d) >= abs(d.sum()) - 1e-9)) / 10_000
    assert diff.table.loc[0, "p_value"] == pytest.approx(p)


def test_cases_of_one_cluster_flip_together():
    # Ten cases from each of four stores; every case of a store moves the same way.
    stores = np.repeat(np.arange(4), 10)
    manifests = [{"store": f"store {s}", "origin": i} for i, s in enumerate(stores)]
    base = noisy(40)
    shift = np.where(stores % 2 == 0, -1.0, -0.2)  # all better, by store-sized amounts
    candidate = report(base + shift + np.random.default_rng(3).normal(0, 0.01, 40), manifests)
    alone = candidate.compare(report(base, manifests), metric="mae")
    assert alone.table["verdict"][0] == "better"  # 40 cases treated as independent
    together = candidate.compare(report(base, manifests), metric="mae", cluster="store")
    row = together.table.iloc[0]
    assert row["verdict"] == "too few to tell" and np.isnan(row["p_value"])
    assert (row["cases"], row["clusters"]) == (40, 4)
    assert together.units_needed == 6


def test_six_clusters_can_reach_five_percent_and_five_cannot():
    for clusters, verdict in [(5, "too few to tell"), (6, "worse")]:
        manifests = [{"site": c} for c in range(clusters) for _ in range(3)]
        base = noisy(3 * clusters)
        diff = report(base + 1.0, manifests).compare(report(base, manifests),
                                                     metric="mae", cluster="site")
        assert diff.table["verdict"][0] == verdict
    # Six clusters all worse: the observed signs and their mirror image are 2 of 64 patterns.
    assert diff.table["p_value"][0] == pytest.approx(2 / 64, abs=0.006)
    loose = report(base + 1.0, manifests).compare(report(base, manifests),
                                                  metric="mae", cluster="site", alpha=0.2)
    assert loose.units_needed == 4


def test_groups_get_holms_correction():
    groups = ["north"] * 30 + ["south"] * 30 + ["east"] * 30
    manifests = [{"region": g, "i": i} for i, g in enumerate(groups)]
    base = noisy(90)
    shift = np.r_[np.full(30, -0.5), np.random.default_rng(4).normal(0, 0.3, 60)]
    diff = report(base + shift, manifests).compare(report(base, manifests),
                                                   metric="mae", by="region")
    assert diff.table["group"].tolist() == ["north", "south", "east"]
    assert diff.table["verdict"].tolist()[0] == "better"
    # The strong group: only the observed signs reach it, so 1 / 10,000, times three groups.
    assert diff.table["p_value"][0] == pytest.approx(3 / 10_000)
    assert diff.table["cases"].tolist() == [30, 30, 30]


def test_holm_adjusted_p_values_agree_with_holms_rejections():
    rng = np.random.default_rng(0)
    for _ in range(200):
        p = rng.uniform(0, 0.1, rng.integers(1, 8))
        np.testing.assert_array_equal(_holm_adjusted(p) <= 0.05, _holm(p, 0.05))
    assert _holm_adjusted([0.01, 0.04, 0.03]).tolist() == pytest.approx([0.03, 0.06, 0.06])


def test_the_best_of_many_candidates_does_not_beat_the_baseline_by_luck():
    # Twenty candidates with no real gain, alike as variants of one model. Tested alone, the best
    # often looks better; against the most extreme of all flipped candidates, rarely.
    rng = np.random.default_rng(11)
    n, k, reps = 60, 20, 100
    alone = adjusted = 0
    shared = rng.normal(0, 1, (reps, n, 1))
    for rep in range(reps):
        d = 0.7 * shared[rep] + 0.7 * rng.normal(0, 1, (n, k))
        base = noisy(n, seed=rep)
        named = {f"variant {j}": report(base + d[:, j]) for j in range(k)}
        diff = compare_reports(named, report(base), metric="mae", seed=rep)
        single, _ = _paired_flip_test(d, rep)
        alone += np.any((single <= 0.05) & (d.mean(axis=0) < 0))
        adjusted += len(diff.better) > 0
    # Here 1 in 100 against 17 in 100; a "better" verdict is half of a two-sided 5% test.
    assert adjusted / reps <= 0.06
    assert alone / reps >= 0.10


def test_each_candidate_gets_a_p_value_adjusted_for_all_of_them():
    base = noisy(50)
    rng = np.random.default_rng(6)
    strong, weak = base - 0.5 - rng.uniform(0, 0.2, 50), base + rng.normal(0.02, 0.3, 50)
    together = compare_reports({"strong": report(strong), "weak": report(weak)}, report(base), metric="mae")
    assert together.table["candidate"].tolist() == ["strong", "weak"]
    assert together.table["verdict"].tolist() == ["better", "no clear difference"]
    weak_alone = report(weak).compare(report(base), metric="mae")
    assert together.table["p_value"][1] >= weak_alone.table["p_value"][0]


def test_a_case_that_newly_crashes_is_worse_and_one_that_newly_runs_is_not():
    base = noisy(30)
    crashing = report(base, crashed={3}).compare(report(base), metric="mae")
    row = crashing.table.iloc[0]
    assert (row["verdict"], row["newly_failing"], row["cases"]) == ("worse", 1, 29)
    assert "1 case now crashes or fails an output check" in str(crashing)
    with pytest.raises(AssertionError, match="worse than the baseline in 1 of 1 rows"):
        crashing.assert_no_regression()
    back = report(base).compare(report(base, crashed={3}), metric="mae")
    row = back.table.iloc[0]
    assert (row["verdict"], row["newly_passing"], row["newly_failing"]) == ("no clear difference", 1, 0)
    back.assert_no_regression()
    unscored = report([None] + list(base[1:])).compare(report(base), metric="mae")
    assert unscored.table.loc[0, "newly_failing"] == 0 and unscored.table.loc[0, "cases"] == 29


def test_the_same_scores_and_roundoff_are_no_difference():
    base = noisy(30)
    same = report(base).compare(report(base), metric="mae")
    assert same.table.loc[0, "verdict"] == "no clear difference" and same.table.loc[0, "p_value"] == 1.0
    roundoff = report(base * (1 + 1e-12)).compare(report(base), metric="mae")
    assert roundoff.table.loc[0, "verdict"] == "no clear difference"
    same.assert_no_regression()


def test_too_few_to_tell_fails_the_regression_gate():
    manifests = [{"site": c} for c in range(3) for _ in range(5)]
    base = noisy(15)
    diff = report(base - 1, manifests).compare(report(base, manifests), metric="mae", cluster="site")
    assert len(diff.too_few) == 1 and diff.better.empty
    with pytest.raises(AssertionError, match="at least 6 must differ"):
        diff.assert_no_regression()


@pytest.mark.parametrize("change, message", [
    (lambda r: TestReport(r.cases[:-1], r.metadata), "has 9 cases and the baseline 10"),
    (lambda r: TestReport([CaseResult("case", None, [], c.metrics, {**c.manifest, "case": 99})
                           if i == 2 else c for i, c in enumerate(r.cases)], {}),
     "case 2 differs .* manifest key 'case' is 99"),
])
def test_reports_on_other_cases_are_refused(change, message):
    base = report(noisy(10))
    with pytest.raises(ValueError, match=message):
        change(base).compare(base, metric="mae")


def test_bad_requests_are_refused():
    base = report(noisy(10), [{"store": "a", "targets": ["y"]} for _ in range(10)])
    with pytest.raises(ValueError, match="no case has a metric 'rmse'; the reports have mae"):
        base.compare(base, metric="rmse")
    with pytest.raises(ValueError, match="case 0 has no manifest key 'region'"):
        base.compare(base, metric="mae", by="region")
    with pytest.raises(ValueError, match="one plain value per case"):
        base.compare(base, metric="mae", cluster="targets")
    with pytest.raises(ValueError, match="alpha"):
        base.compare(base, metric="mae", alpha=1.5)
    with pytest.raises(TypeError, match="candidate must be a TestReport"):
        compare_reports("report.json", base, metric="mae")
    with pytest.raises(TypeError, match="metric must be the name of a metric"):
        base.compare(base, metric=None)
    with pytest.raises(ValueError, match="at least one report"):
        compare_reports([], base, metric="mae")


def stores(n_stores=24, seed=3):
    rng = np.random.default_rng(seed)
    days = pd.date_range("2024-01-01", periods=140, freq="D")
    cases = []
    for store in range(n_stores):
        level = rng.uniform(20, 60)
        weekly = 0.3 * level * np.sin(2 * np.pi * np.arange(140) / 7 + store)
        sales = pd.Series(level + weekly + rng.normal(0, 3, 140), index=days, name="sales")
        cases += cases_from_series(sales, [84, 98, 112, 126], 7, name="sales",
                                   manifest={"store": f"store {store}", "region": ["north", "south", "east"][store % 3]})
    return cases


def test_on_your_own_data_by_region_with_stores_as_clusters():
    cases = stores()
    last = Pipeline(LastValueForecaster()).test(cases=cases)
    seasonal_run = Pipeline(SeasonalLastValueForecaster(period=7)).run(cases=cases)
    diff = seasonal_run.compare(last, metric="mae", by="region", cluster="store")
    assert isinstance(diff, ReportComparison)
    assert diff.table["verdict"].tolist() == ["better"] * 3
    assert diff.table["clusters"].tolist() == [8, 8, 8] and diff.table["cases"].tolist() == [32, 32, 32]
    text = str(diff)
    assert text.startswith("mae against the baseline (lower is better; alpha 0.05; cases in one store move together)")
    assert "region north  better" in text
    again = seasonal_run.compare(last, metric="mae", by="region", cluster="store")
    pd.testing.assert_frame_equal(diff.table, again.table)
    repr(diff)


def test_compare_on_a_run_or_a_report_is_compare_reports_with_it_as_the_candidate():
    cases = stores()
    last = Pipeline(LastValueForecaster()).test(cases=cases)
    seasonal_run = Pipeline(SeasonalLastValueForecaster(period=7)).run(cases=cases)
    settings = dict(metric="mae", by="region", cluster="store", alpha=0.1, higher_is_better=True, seed=3)
    for candidate in (seasonal_run, seasonal_run.report):
        method = candidate.compare(last, **settings)
        function = compare_reports(candidate, last, **settings)
        pd.testing.assert_frame_equal(method.table, function.table)
        assert str(method) == str(function)
    assert last.compare(seasonal_run, metric="mae").table["verdict"].tolist() == ["worse"]
    with pytest.raises(TypeError, match="metric"):
        last.compare(seasonal_run)


def test_candidates_are_named_by_their_model_or_their_position():
    cases = stores(n_stores=6)
    last = Pipeline(LastValueForecaster()).test(cases=cases)
    seasonal = Pipeline(SeasonalLastValueForecaster(period=7)).test(cases=cases)
    fortnight = Pipeline(SeasonalLastValueForecaster(period=14)).run(cases=cases)
    mean = Pipeline(MeanForecaster()).test(cases=cases)
    assert compare_reports([seasonal, fortnight], last, metric="mae").table["candidate"].tolist() == [
        "candidate 1", "candidate 2"]  # both are SeasonalLastValueForecaster
    assert compare_reports([seasonal, mean], last, metric="mae").table["candidate"].tolist() == [
        "SeasonalLastValueForecaster", "MeanForecaster"]
    assert seasonal.compare(last, metric="mae").table["candidate"].tolist() == [
        "SeasonalLastValueForecaster"]


def test_a_list_of_candidates_is_labelled_by_their_names():
    cases = stores(n_stores=6)
    last = Pipeline(LastValueForecaster()).run(cases=cases)
    weekly = Pipeline(SeasonalLastValueForecaster(period=7), name="same weekday").run(cases=cases)
    fortnight = Pipeline(SeasonalLastValueForecaster(period=14), name="a fortnight ago").run(cases=cases)
    mean = Pipeline(MeanForecaster()).run(cases=cases)
    assert compare_reports([weekly, fortnight, mean], last, metric="mae").table["candidate"].tolist() == [
        "same weekday", "a fortnight ago", "MeanForecaster"]  # unnamed: its model
    assert weekly.compare(last, metric="mae").table["candidate"].tolist() == ["same weekday"]
    twice = Pipeline(MeanForecaster(), name="same weekday").run(cases=cases)
    assert compare_reports([weekly, twice], last, metric="mae").table["candidate"].tolist() == [
        "candidate 1", "candidate 2"]
    by_hand = compare_reports({"weekly": weekly, "mean": mean}, last, metric="mae")  # a dictionary names them
    assert by_hand.table["candidate"].tolist() == ["weekly", "mean"]
