import numpy as np
import pytest
import warnings

from timegym.metrics import (UndefinedMetricError, crps, energy_score, mae, mase,
                             mean_reference_mse, mse, pinball, variogram_score)


def test_point_scores_match_manual_values():
    observed = np.array([[1.0], [3.0]])
    predicted = np.array([[2.0], [1.0]])
    assert mse(observed, predicted) == 2.5
    assert mae(observed, predicted) == 1.5
    assert mean_reference_mse(observed, predicted) == 2.5


def test_mase_scales_each_target_before_averaging():
    context = [[0, 0], [2, 1], [4, 2]]
    assert mase([[1, 2], [3, 4]], np.zeros((2, 2)), context) == 2.0
    with pytest.raises(UndefinedMetricError, match="zero"):
        mase([[1, 2]], [[0, 0]], [[0, 0], [1, 0]])
    with pytest.raises(UndefinedMetricError, match="more context"):
        mase([[1]], [[0]], [[0]], seasonality=1)


def test_pinball_matches_manual_value():
    assert pinball([[1]], [[[0, 2]]], (0.25, 0.75)) == 0.25


def test_crps_matches_empirical_pairwise_formula_and_single_sample():
    assert crps([[1]], [[[0]], [[2]]]) == 0.5
    rng = np.random.default_rng(6)
    observed = rng.normal(size=(4, 2))
    samples = rng.normal(size=(7, 4, 2))
    expected = (np.abs(samples - observed).mean()
                - 0.5 * np.abs(samples[:, None] - samples[None, :]).mean())
    assert crps(observed, samples) == pytest.approx(expected)
    assert crps(observed, samples[:1]) == pytest.approx(mae(observed, samples[0]))


def test_energy_score_matches_manual_joint_norm_and_chunking():
    observed, samples = [[0, 0]], [[[0, 0]], [[3, 4]]]
    assert energy_score(observed, samples, chunk_size=1) == 1.25
    assert energy_score(observed, samples, chunk_size=5) == 1.25


def test_variogram_score_manual_pair_normalization():
    assert variogram_score([[0, 4]], [[[0, 0]], [[0, 4]]]) == 1.0
    with pytest.raises(UndefinedMetricError, match="two coordinates"):
        variogram_score([[0]], [[[1]], [[2]]])


def test_marginal_crps_cannot_detect_changed_joint_coupling():
    observed = np.zeros((1, 2))
    coupled = np.array([[[0, 0]], [[1, 1]], [[2, 2]], [[3, 3]]])
    permuted = coupled.copy()
    permuted[:, 0, 1] = permuted[::-1, 0, 1]
    assert crps(observed, coupled) == crps(observed, permuted)
    assert energy_score(observed, coupled) != energy_score(observed, permuted)
    assert variogram_score(observed, coupled) != variogram_score(observed, permuted)


@pytest.mark.parametrize("function", [mse, mae, mean_reference_mse])
def test_point_metrics_reject_broadcasting_and_nonfinite_values(function):
    with pytest.raises(ValueError, match="identical shapes"):
        function(np.ones((2, 1)), np.ones((2,)))
    with pytest.raises(ValueError, match="finite"):
        function([[1]], [[np.nan]])
    with pytest.raises(ValueError, match="nonempty"):
        function([], [])


@pytest.mark.parametrize("function", [crps, energy_score, variogram_score])
def test_ensemble_metrics_reject_wrong_axes(function):
    with pytest.raises(ValueError, match="shape"):
        function(np.zeros((2, 1)), np.zeros((2, 1, 3)))
    with pytest.raises(ValueError, match="nonempty"):
        function(np.zeros((2, 1)), np.zeros((0, 2, 1)))


def test_quantile_metrics_reject_crossing_and_mismatched_levels():
    with pytest.raises(ValueError, match="cross"):
        pinball([[1]], [[[2, 0]]], (0.25, 0.75))
    with pytest.raises(ValueError, match="strictly increasing"):
        pinball([[1]], [[[0, 2]]], (0.75, 0.25))
    with pytest.raises(ValueError, match="shape"):
        pinball([[1]], [[[0, 1, 2]]], (0.25, 0.75))


@pytest.mark.parametrize("function", [mse, mae, mean_reference_mse])
def test_point_scores_reject_complex_values_without_lossy_conversion(function):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="real-valued"):
            function(np.array([[1 + 2j]]), [[0]])
        with pytest.raises(ValueError, match="real-valued"):
            function([[0]], np.array([[1 + 0j]]))
        with pytest.raises(ValueError, match="real-valued"):
            function([[0]], np.array([[np.complex128(1 + 2j)]], dtype=object))


@pytest.mark.parametrize("function", [crps, energy_score, variogram_score])
def test_ensemble_scores_reject_complex_values_without_lossy_conversion(function):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="real-valued"):
            function([[0, 1]], np.ones((2, 1, 2), dtype=complex))


def test_quantile_and_context_scores_reject_complex_values():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="real-valued"):
            pinball([[1]], np.array([[[0 + 1j, 2 + 1j]]]), (0.25, 0.75))
        with pytest.raises(ValueError, match="real-valued"):
            pinball([[1]], [[[0, 2]]], np.array([0.25, 0.75], dtype=complex))
        with pytest.raises(ValueError, match="real-valued"):
            mase([[1]], [[0]], np.array([[0], [1]], dtype=complex))

