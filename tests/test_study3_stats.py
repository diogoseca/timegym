"""Study 3 statistics on hand-made counts with known answers, including the numbers the protocol states."""
import numpy as np
import pytest
from scipy.stats import binom

from experiments import study3_stats as stats
from experiments.run_addendum import sign_flip_pvalue


def test_the_alarm_counts_the_protocol_states():
    assert stats.calibration_rank(239, 0.05) == 228
    assert [stats.alarms_needed(93, size) for size in (1, 4, 16, 400)] == [10, 12, 14, 18]
    assert stats.beta_binomial_pvalue(10, 93) == pytest.approx(0.0388, abs=5e-5)
    assert stats.binomial_pvalue(10, 93, 0.05) == pytest.approx(0.0179, abs=5e-5)
    assert stats.beta_binomial_pvalue(0, 93) == 1.0 and stats.beta_binomial_pvalue(0, 0) is None


def test_beta_binomial_tail_by_direct_sum():
    from math import comb, lgamma, exp

    def pmf(x, m, a, b):
        log_beta = lambda p, q: lgamma(p) + lgamma(q) - lgamma(p + q)
        return comb(m, x) * exp(log_beta(x + a, m - x + b) - log_beta(a, b))

    for x in (1, 5, 12):
        assert stats.beta_binomial_pvalue(x, 20) == pytest.approx(sum(pmf(k, 20, 12, 228) for k in range(x, 21)),
                                                                  rel=1e-10)
    # Nineteen calibration cases at 0.05: the threshold is the largest of them, so a = 1 and b = 19.
    assert stats.beta_binomial_pvalue(1, 1, n_calibration=19) == pytest.approx(1 / 20)


def test_the_classification_rule():
    assert stats.binomial_pvalue(9, 96) > 0.05 >= stats.binomial_pvalue(10, 96)
    assert stats.binomial_pvalue(10, 96) == pytest.approx(0.022, abs=5e-4)
    assert stats.shift_class(["PASS"] * 96)["class"] == "exact"
    assert stats.shift_class(["PASS"] * 95 + ["FAIL"])["class"] is None
    assert stats.shift_class(["PASS"] * 95 + ["ERROR"], 9, 96)["class"] == "in distribution"
    row = stats.shift_class(["FAIL"] * 96, 10, 96)
    assert row["class"] == "not applicable" and row["public_pass"] == 0
    assert row["binomial_p_value"] == pytest.approx(binom.sf(9, 96, 0.05))


def test_paired_sign_flip_uses_the_registered_generator():
    d = np.array([0.3, -0.1, 0.2, 0.5, 0.05, -0.02, 0.4])
    expected = sign_flip_pvalue(d, 9999, np.random.default_rng([7_600_000, 7]))
    assert stats.paired_pvalue(d) == expected
    assert stats.paired_pvalue(np.zeros(5)) == 1.0
    assert stats.paired_pvalue([]) is None
    # All twelve differences positive: the exact sign-flip p-value is 1/4096; the Monte Carlo one is near it.
    assert stats.paired_pvalue(np.full(12, 1.0)) == pytest.approx(1 / 4096, abs=5e-4)


def test_holm_and_unadjusted_decisions():
    decisions = stats.holm_decisions({"a": 0.01, "b": 0.04, "c": 0.02, "d": None, "e": 0.5})
    assert {name: decision["holm"] for name, decision in decisions.items()} == {
        "a": True, "b": False, "c": False, "d": None, "e": False}
    assert {name: decision["unadjusted"] for name, decision in decisions.items()} == {
        "a": True, "b": True, "c": True, "d": None, "e": False}
    assert decisions["a"]["family_size"] == 4
    four = stats.holm_decisions({"w": 3 / 240, "x": 4 / 240, "y": 6 / 240, "z": 12 / 240})
    assert all(decision["holm"] for decision in four.values())
    assert stats.holm_decisions({"only": None}) == {"only": {"p_value": None, "family_size": 0,
                                                             "unadjusted": None, "holm": None}}


def test_deterministic_rule_and_rates():
    assert stats.deterministic_decision(1, 93, 0)["caught"] is True
    assert stats.deterministic_decision(0, 93, 0)["caught"] is False
    decision = stats.deterministic_decision(5, 93, 2)
    assert decision["deterministic"] is False and decision["caught"] is None and decision["clean_alarms"] == 2
    assert stats.deterministic_decision(0, 0, 0)["caught"] is None
    assert stats.rate(0, 0) is None
    interval = stats.rate(46, 93)
    assert interval["count"] == 46 and interval["n"] == 93 and interval["ci95"][1] - interval["ci95"][0] < 0.2


def test_ranks_and_curves():
    calibration = [1.0, 2.0, 3.0, 4.0]
    assert stats.rank_counts([0.5, 2.0, 3.5, 9.0], calibration) == [5, 4, 2, 1]
    curve = stats.detection_curve([[1, 5], [2, 2, 2, 2]], levels=5)
    assert curve == [0.25, 0.75, 0.75, 0.75, 1.0]
    assert stats.detection_curve([[], []], levels=3) == [None, None, None]
    intervals = stats.curve_intervals([[1, 5], [2, 2, 2, 30]], levels=(1, 2, 24))
    assert [intervals[k]["count"] for k in ("1", "2", "24")] == [1, 4, 5]
    assert all(intervals[k]["n"] == 6 for k in intervals)
