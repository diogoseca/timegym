"""Runners of protocol 3.0, Parts 3, 4 and 5, on hand-made inputs. No test here computes a study outcome.

Part 3 is tested on hand-made p-values and on a smoke run of two suites with 19 calibration draws, 20
sample paths and seeds below 1,000,000, for structure and determinism only. Part 4 is tested against
brute force, for unbiasedness on a small known Gaussian, and on hand-typed estimates; its simulation
loop runs twice on a toy two-dimensional reference, for structure and determinism only. Part 5's
estimators run on NumPy arrays whose answer is known, and its runner on stub cells and stub estimators.
"""
import copy
import itertools
import json
import math
import sys

import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2

from timegym import Reference
from timegym.statistics import wilson_interval

from experiments import study3_common as common
from experiments import study3_margin as margin
from experiments import study3_predictability as predictability
from experiments import run_study3_margin as margin_runner
from experiments import run_study3_predictability as predictability_runner
from experiments import run_study3_suites as suites
from experiments.run_addendum import fisher_combination
from experiments.run_supplement import holm_rejections


@pytest.fixture(scope="module")
def protocol():
    return common.load_protocol()


def _draft(protocol):
    """A copy of the protocol as it is before registration, whatever the status of the file."""
    draft = copy.deepcopy(protocol)
    draft.update(status="draft")
    draft["registration"].update(merge_commit="", registered_on="")
    return draft


@pytest.mark.parametrize("runner", [suites, margin_runner, predictability_runner])
def test_a_draft_protocol_stops_every_runner_and_the_plan_computes_nothing(runner, monkeypatch, capsys, protocol):
    monkeypatch.setattr(common, "load_protocol", lambda: _draft(protocol))
    monkeypatch.setattr(sys, "argv", ["runner"])
    with pytest.raises(RuntimeError, match="may not start"):
        runner.main()
    monkeypatch.setattr(sys, "argv", ["runner", "--plan"])
    runner.main()
    shown = json.loads(capsys.readouterr().out)
    assert shown["part"] == runner.PART and "cpu_hours_estimate" in shown


# Part 3


def test_fisher_combination_on_hand_made_pvalues():
    assert fisher_combination([1.0, 1.0, 1.0, 1.0]) == (0.0, 1.0)
    statistic, probability = fisher_combination([0.01] * 4)
    assert statistic == pytest.approx(-8 * math.log(0.01))
    assert probability == pytest.approx(chi2.sf(statistic, 8), rel=1e-12)
    statistic, probability = fisher_combination([0.2, 0.5, 1 / 240, 0.9])
    assert probability == pytest.approx(chi2.sf(-2 * math.log(0.2 * 0.5 / 240 * 0.9), 8), rel=1e-12)


def test_holm_attains_its_thresholds_at_multiples_of_one_over_240():
    assert holm_rejections([3 / 240, 0.5, 0.5, 0.5], 0.05).tolist() == [True, False, False, False]
    assert holm_rejections([0.5, 4 / 240, 0.5, 0.5], 0.05).tolist() == [False, False, False, False]
    assert holm_rejections([12 / 240, 6 / 240, 4 / 240, 3 / 240], 0.05).tolist() == [True] * 4
    assert holm_rejections([12 / 240, 6 / 240, 5 / 240, 3 / 240], 0.05).tolist() == [False, False, False, True]


def _records(pvalues, nulls):
    """Hand-made records: pvalues[variant][suite] lists the four p-values in condition order."""
    rows = []
    for variant, by_suite in pvalues.items():
        for suite, values in enumerate(by_suite):
            for condition, (p, null) in enumerate(zip(values, nulls[variant])):
                rows.append({"variant": variant, "suite": suite, "condition": f"c{condition}", "p_value": p,
                             "null_true": null})
    return rows


def test_analyse_counts_hand_made_suites():
    conditions = ["c0", "c1", "c2", "c3"]
    pvalues = {"clean": [[0.5, 0.5, 0.5, 0.5], [0.04, 0.5, 0.5, 0.5], [3 / 240, 0.9, 0.9, 0.9]],
               "shuffle": [[0.02, 0.02, 0.02, 0.02], [0.5, 3 / 240, 0.5, 0.5], [0.04, 0.9, 0.9, 0.9]]}
    nulls = {"clean": [True] * 4, "shuffle": [True, False, False, False]}
    found = suites.analyse(_records(pvalues, nulls), conditions, ["clean", "shuffle"], 0.05)
    assert found["suites"] == 3 and len(found["suite_rows"]) == 6
    fisher = {(row["variant"], row["suite"]): row["fisher"] for row in found["suite_rows"]}
    expected = {key: fisher_combination(values)[1] <= 0.05
                for key, values in (((v, s), p) for v in pvalues for s, p in enumerate(pvalues[v]))}
    assert fisher == expected
    assert fisher[("shuffle", 0)] and not fisher[("clean", 0)]
    clean, shuffle = found["variant_summaries"]
    assert clean["suites_flagged"]["raw"] == wilson_interval([False, True, True])
    assert clean["suites_flagged"]["holm"] == wilson_interval([False, False, True])
    assert clean["suites_flagged"]["fisher"] == wilson_interval([expected["clean", s] for s in range(3)])
    assert clean["suites_rejecting_a_true_null"]["holm"] == wilson_interval([False, False, True])
    assert clean["suites_rejecting_a_true_null"]["fisher"] == clean["suites_flagged"]["fisher"]
    assert shuffle["true_null_conditions"] == ["c0"]
    assert shuffle["suites_rejecting_a_true_null"]["raw"] == wilson_interval([True, False, True])
    assert shuffle["suites_rejecting_a_true_null"]["holm"] == wilson_interval([False, False, False])
    assert shuffle["suites_rejecting_a_true_null"]["fisher"] is None
    per_condition = {(row["variant"], row["condition"]): row for row in found["condition_summaries"]}
    assert per_condition["shuffle", "c1"]["raw"] == wilson_interval([True, True, False])
    assert per_condition["shuffle", "c1"]["holm"] == wilson_interval([False, True, False])


def test_analyse_refuses_a_record_whose_stored_decision_differs():
    rows = _records({"clean": [[0.5, 0.5, 0.5, 0.5]]}, {"clean": [True] * 4})
    rows[0]["raw_alarm"] = True
    with pytest.raises(RuntimeError, match="differs"):
        suites.analyse(rows, ["c0", "c1", "c2", "c3"], ["clean"], 0.05)


def test_registered_spec_takes_every_seed_from_the_seed_table(protocol):
    spec = suites.registered_spec(protocol)
    assert (spec["independent_suites"], spec["n_calibration"], spec["n_samples"]) == (93, 239, 64)
    assert spec["seed_base"] == 8_000_000 and spec["variants"] == ["clean", "partial_shuffle", "reduced_spread"]
    assert [c["id"] for c in spec["conditions"]] == ["factorized_white_noise", "correlated_white_noise",
                                                     "negative_ar1", "correlated_random_walk"]
    broken = copy.deepcopy(protocol)
    broken["seeds"]["case_seed_blocks"]["covariance_calibration"] = [10_000_001, 10, 93, 4]
    with pytest.raises(ValueError, match="covariance_calibration"):
        suites.registered_spec(broken)


def test_smoke_run_of_two_small_suites_is_deterministic(protocol):
    spec = suites.registered_spec(protocol)
    spec.update(independent_suites=2, n_calibration=19, n_samples=20, seed_base=500,
                reference_sample_seed_offset=1000, calibration_seed_offset=2000, mutation_seed_offset=3000)
    first, second = suites.run(spec), suites.run(copy.deepcopy(spec))
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    records = first["covariance"]["records"]
    assert len(records) == 2 * 4 * 3 and max(row["calibration_seed"] for row in records) < 1_000_000
    analysis = first["analysis"]
    assert analysis["suites"] == 2 and len(analysis["suite_rows"]) == 6 and len(analysis["condition_summaries"]) == 12
    assert {row["variant"] for row in analysis["variant_summaries"]} == set(spec["variants"])
    for row in analysis["suite_rows"]:
        assert len(row["p_values"]) == 4 and all(p * 20 == round(p * 20) for p in row["p_values"])


# Part 4


def _brute_force(paths, sigma0):
    paths = paths.reshape(len(paths), -1)
    n, gram = len(paths), paths @ paths.T
    y1 = np.mean([gram[i, j] ** 2 for i, j in itertools.permutations(range(n), 2)])
    y2 = np.mean([gram[i, j] * gram[j, k] for i, j, k in itertools.permutations(range(n), 3)])
    y3 = np.mean([gram[i, j] * gram[k, l] for i, j, k, l in itertools.permutations(range(n), 4)])
    s = np.cov(paths.T, ddof=1)
    return (y1 - 2 * y2 + y3 - 2 * np.trace(s @ sigma0) + np.trace(sigma0 @ sigma0)) / np.sum(sigma0 ** 2)


SIGMA0 = np.array([[2.0, 0.3, 0.0], [0.3, 1.0, 0.2], [0.0, 0.2, 0.5]])


def test_estimator_matches_brute_force_and_ignores_a_common_translation():
    paths = np.random.default_rng(11).normal(size=(7, 3)) + np.array([5.0, -2.0, 1.0])
    found = margin.estimate(paths, SIGMA0)
    assert found["T"] == pytest.approx(_brute_force(paths, SIGMA0), abs=1e-12)
    assert margin.estimator(paths + 40.0, SIGMA0) == pytest.approx(found["T"], abs=1e-9)
    assert margin.estimator(paths.reshape(7, 3, 1), SIGMA0) == found["T"]


def test_leave_one_out_values_and_standard_error_match_a_direct_computation():
    paths = np.random.default_rng(12).normal(size=(9, 3)) * [1.0, 2.0, 0.5]
    found = margin.estimate(paths, SIGMA0)
    direct = np.array([_brute_force(np.delete(paths, i, axis=0), SIGMA0) for i in range(9)])
    np.testing.assert_allclose(found["leave_one_out"], direct, rtol=0, atol=1e-12)
    expected = math.sqrt(8 / 9 * np.sum((direct - direct.mean()) ** 2))
    assert found["standard_error"] == pytest.approx(expected, rel=1e-10)


def test_estimator_is_unbiased_on_a_small_known_gaussian():
    sigma = np.array([[1.0, 0.5, 0.0], [0.5, 2.0, -0.4], [0.0, -0.4, 1.5]])
    mean = np.array([3.0, -1.0, 0.5])
    truth = np.sum((sigma - SIGMA0) ** 2) / np.sum(SIGMA0 ** 2)
    rng = np.random.default_rng(13)
    draws = rng.multivariate_normal(mean, sigma, size=(20_000, 8))
    values = np.array([margin.estimator(sample, SIGMA0) for sample in draws])
    error = values.std(ddof=1) / math.sqrt(len(values))
    assert abs(values.mean() - truth) < 4 * error


@pytest.mark.parametrize("t, se, expected", [
    (0.30, 0.01, "FAIL"), (0.10, 0.05, "WITHIN_TOLERANCE"), (0.26, 0.05, "INCONCLUSIVE"),
    (0.25, 0.0, "INCONCLUSIVE"), (-0.02, 0.03, "WITHIN_TOLERANCE"),
])
def test_verdict_on_hand_typed_estimates(t, se, expected):
    found = margin.decide(t, se, 0.5)
    assert found["verdict"] == expected
    lower, upper = found["interval_D_squared"]
    assert (lower, upper) == (t - margin.Z * se, t + margin.Z * se)
    assert found["interval_D"] == [math.sqrt(max(lower, 0)), math.sqrt(max(upper, 0))]
    assert found["D_hat"] == math.sqrt(max(t, 0))


def test_decision_constants_match_the_protocol(protocol):
    part = protocol["part_4_joint_spread_margin"]
    assert str(margin.Z) in part["decision"]["interval"]
    assert margin.decide(0.1, 0.0, 0.25)["verdict"] == "FAIL"
    with pytest.raises(ValueError):
        margin.decide(0.1, -1.0, 0.25)


def test_registered_cells_and_seeds(protocol):
    found, cells = margin_runner.settings(protocol)
    assert len(cells) == 240 and [cell["j"] for cell in cells] == list(range(240))
    assert found["case_seeds"] == [12_100_000, 12_100_001, 12_100_002, 12_100_003] and found["tag"] == 12_000_000
    assert found["repetitions"] == 381
    first = [(cell["direction"], cell["multiple"], cell["n_paths"]) for cell in cells[:6]]
    assert first == [("none", 0.0, 64), ("none", 0.0, 256), ("shrink", 0.5, 64), ("shrink", 0.5, 256),
                     ("inflate", 0.5, 64), ("inflate", 0.5, 256)]
    assert cells[30]["margin"] == 0.4375 and cells[60]["condition_index"] == 1
    for cell in cells:
        c = margin_runner.spread_multiplier(cell["d"], cell["direction"])
        assert abs(c * c - 1) == pytest.approx(cell["d"], abs=1e-15)
    # The largest shrink of the grid: sqrt(1 - 0.65625) = 0.58630.
    assert margin_runner.spread_multiplier(1.5 * 0.4375, "shrink") == pytest.approx(0.5863, abs=5e-5)
    assert margin_runner.ensemble_seed(12_000_000, 3, 7) == int(
        np.random.SeedSequence([12_000_000, 3, 7]).generate_state(1)[0])


def _toy_reference():
    covariance = np.array([[1.0, 0.4], [0.4, 2.0]])
    return Reference(mean=np.array([[1.0, -2.0]]), variance=np.diag(covariance).reshape(1, 2), covariance=covariance,
                     index=pd.RangeIndex(10, 11), target_names=("a", "b"))


def test_ensemble_scales_the_draws_about_the_mean():
    reference = _toy_reference()
    paths = margin_runner.ensemble(reference, 0.36, "shrink", 6, 99)
    draws = reference.sample(6, 99)
    np.testing.assert_allclose(paths, reference.mean + 0.8 * (draws - reference.mean), rtol=0, atol=1e-15)


def test_simulation_smoke_run_is_deterministic_and_independent_of_workers():
    cells = margin_runner.build_cells(["toy"], [0.5], [0.0, 1.0], [8, 10])
    assert len(cells) == 6
    references = [_toy_reference()]
    one = margin_runner.run_simulation(references, cells, 2, 4321, workers=1)
    two = margin_runner.run_simulation(references, cells, 2, 4321, workers=2)
    assert json.dumps(one, sort_keys=True) == json.dumps(two, sort_keys=True)
    assert [row["j"] for row in one] == list(range(6))
    for row in one:
        assert set(row["verdicts"]) == set(margin.VERDICTS) and len(row["T"]) == 2
        assert sum(row["verdicts"][name]["count"] for name in margin.VERDICTS) == 2


# Part 5


def test_welch_windows_and_hann_convention():
    assert predictability.welch_windows(24) == (6, [0, 3, 6, 9, 12, 15, 18])
    from scipy.signal.windows import hann

    np.testing.assert_allclose(hann(6, sym=False), [0, 0.25, 0.75, 1, 0.75, 0.25], atol=1e-15)


def test_spectra_match_a_written_out_transform():
    rng = np.random.default_rng(21)
    x, y = rng.normal(size=24), rng.normal(size=24)
    frequencies, s_xx, s_yy, s_xy = predictability.spectra(x, y)
    np.testing.assert_allclose(frequencies, [0, 1 / 6, 1 / 3, 1 / 2])
    w = np.array([0, 0.25, 0.75, 1, 0.75, 0.25])
    basis = np.exp(-2j * np.pi * np.outer(np.arange(4), np.arange(6)) / 6)
    xs = np.array([basis @ (w * x[s:s + 6]) for s in range(0, 19, 3)])
    ys = np.array([basis @ (w * y[s:s + 6]) for s in range(0, 19, 3)])
    u = np.sum(w ** 2)
    np.testing.assert_allclose(s_xx, np.mean(np.abs(xs) ** 2, axis=0) / u, atol=1e-12)
    np.testing.assert_allclose(s_yy, np.mean(np.abs(ys) ** 2, axis=0) / u, atol=1e-12)
    np.testing.assert_allclose(s_xy, np.mean(xs * np.conj(ys), axis=0) / u, atol=1e-12)


def test_identical_segments_have_coherence_one_and_bound_zero():
    x = np.random.default_rng(22).normal(size=24)
    found = predictability.spectral_bound(x, x.copy())
    assert found["defined"] and found["delta_squared"] == 0
    assert found["weighted_coherence"] == pytest.approx(1, abs=1e-7)
    assert found["bound"] == pytest.approx(0, abs=1e-7)
    shifted = predictability.spectral_bound(x, x + 3.0)
    assert shifted["delta_squared"] == pytest.approx(9) and shifted["bound"] == pytest.approx(9, abs=1e-6)


def test_bound_lies_between_the_mean_term_and_the_mean_term_plus_the_variance():
    rng = np.random.default_rng(23)
    for _ in range(20):
        x, y = rng.normal(size=24), rng.normal(size=24) * 2 + 1
        found = predictability.spectral_bound(x, y)
        assert found["delta_squared"] - 1e-12 <= found["bound"] <= found["delta_squared"] + np.var(y) + 1e-12


def test_constant_future_leaves_the_bound_undefined():
    found = predictability.spectral_bound(np.arange(24.0), np.full(24, 5.0))
    assert not found["defined"] and math.isnan(found["bound"])


def test_permutation_entropy_on_hand_computed_patterns():
    assert predictability.weighted_permutation_entropy(np.arange(120.0)) == 0
    assert predictability.weighted_permutation_entropy(np.arange(120.0)[::-1]) == 0
    assert math.isnan(predictability.weighted_permutation_entropy(np.full(10, 2.0)))
    _, patterns = predictability.ordinal_patterns([1.0, 3.0, 2.0, 5.0, 4.0])
    assert patterns.tolist() == [[0, 2, 1], [1, 0, 2], [0, 2, 1]]
    _, tied = predictability.ordinal_patterns([2.0, 2.0, 1.0])
    assert tied.tolist() == [[2, 0, 1]]
    # Weights 2/3, 14/9 and 14/9: pattern (0, 2, 1) carries 20/34 of the weight and (1, 0, 2) 14/34.
    p = np.array([20 / 34, 14 / 34])
    expected = -np.sum(p * np.log2(p)) / math.log2(6)
    values = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
    assert predictability.weighted_permutation_entropy(values) == pytest.approx(expected, rel=1e-12)
    assert predictability.weighted_permutation_entropy(7 * values - 3) == pytest.approx(expected, rel=1e-12)


class _Scenario:
    def __init__(self, level, floor):
        self.level, self.floor = level, floor

    def generate(self, seed):
        task = type("Task", (), {"history": pd.DataFrame({"y": np.arange(120.0) + self.level + seed})})()
        reference = type("Ref", (), {"variance": np.full((24, 1), self.floor)})()
        return type("Case", (), {"task": task, "observed": pd.DataFrame({"y": np.full(24, -1.0)}),
                                 "reference": reference})()


class _Cell:
    def __init__(self, pattern, level, floor):
        self.pattern, self.noise_sd, self.scenario = pattern, 0.5, _Scenario(level, floor)


def test_predictability_runner_plumbing_with_stub_estimators():
    seen = []

    def bound(x, y):
        seen.append((x.copy(), y.copy()))
        return {"bound": 4.0, "defined": True} if y[0] >= 0 else {"bound": 3.0, "defined": True}

    def entropy(values):
        return float(values[0] % 7) / 10

    cells = [_Cell("noise", 0.0, 2.0), _Cell("trend", 1.0, 1.0), _Cell("random_walk", 2.0, 4.0)]
    found = predictability_runner.run(cells, [0, 1], bound=bound, entropy=entropy)
    first_x, first_y = seen[0]
    np.testing.assert_array_equal(first_x, np.arange(96.0, 120.0))
    np.testing.assert_array_equal(first_y, np.full(24, -1.0))
    np.testing.assert_array_equal(seen[1][0], np.arange(72.0, 96.0))
    np.testing.assert_array_equal(seen[1][1], np.arange(96.0, 120.0))
    assert len(found["series"]) == 6 and len(found["table"]) == 3
    noise, trend, walk = found["table"]
    assert noise["primary_ratio"] == {"median": 1.5, "p05": 1.5, "p95": 1.5}
    assert noise["history_ratio"]["median"] == 2.0 and noise["primary_undefined"] == 0
    assert trend["primary_ratio"]["median"] == 3.0 and walk["label"] == "no stationary spectrum"
    assert noise["median_wpe"] == pytest.approx(0.05) and walk["median_wpe"] == pytest.approx(0.25)
    assert noise["median_scale_free_floor"] == pytest.approx(2.0 / np.var(np.arange(120.0)))
    assert found["correlations"]["wpe_vs_floor"]["rho"] == pytest.approx(0.5)
    assert found["correlations"]["wpe_vs_scale_free_floor"]["rows"] == 3


def test_predictability_runner_counts_undefined_bounds():
    def bound(x, y):
        return {"bound": math.nan, "defined": False}

    found = predictability_runner.run([_Cell("level", 0.0, 1.0)], [0, 1, 2], bound=bound, entropy=lambda v: math.nan)
    row = found["table"][0]
    assert row["primary_undefined"] == row["history_undefined"] == 3 and row["wpe_undefined"] == 3
    assert row["primary_ratio"]["median"] is None and row["median_wpe"] is None
    json.dumps(found, allow_nan=False)


def test_registered_predictability_inputs(protocol):
    cells, seeds = predictability_runner.registered_inputs(protocol)
    assert len(cells) == 20 and seeds == list(range(13_000_000, 13_000_200))
