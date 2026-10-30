import numpy as np
import pytest

from timegym import GaussianAR1Forecaster, NaiveForecaster, Pipeline, Scenario
from timegym._internal import nonnegative_int, positive_int
from timegym.cli import alignment_demo
from timegym.metrics import energy_score, mase
from timegym.provenance import diagnostic_settings


@pytest.mark.parametrize("value", [True, False, np.bool_(True), np.bool_(False), 1.5, "1", None, -1])
@pytest.mark.parametrize("guard", [positive_int, nonnegative_int])
def test_integer_guards_reject_coercions(value, guard):
    with pytest.raises(ValueError, match="count must be a .* integer"):
        guard(value, "count")


@pytest.mark.parametrize("value", [1, np.int32(1), np.int64(1), np.uint64(1)])
@pytest.mark.parametrize("guard", [positive_int, nonnegative_int])
def test_integer_guards_normalize_python_and_numpy_integers(value, guard):
    result = guard(value, "count")
    assert result == 1 and type(result) is int


def test_zero_is_only_nonnegative():
    assert nonnegative_int(np.int64(0), "seed") == 0
    with pytest.raises(ValueError, match="positive integer"):
        positive_int(0, "count")


@pytest.mark.parametrize("value", [True, np.bool_(True)])
def test_metric_counts_reject_both_boolean_types(value):
    with pytest.raises(ValueError, match="seasonality.*positive integer"):
        mase([1, 2], [1, 2], [1, 2, 3], seasonality=value)
    with pytest.raises(ValueError, match="chunk_size.*positive integer"):
        energy_score(np.ones((2, 1)), np.ones((3, 2, 1)), chunk_size=value)


def test_numpy_integer_settings_are_normalized_across_entry_points():
    summary, *_ = alignment_demo(seed=np.int64(2))
    assert summary["expected_behavior_verified"]
    assert type(summary["seed"]) is int
    assert type(NaiveForecaster(period=np.int64(3)).period) is int
    assert type(GaussianAR1Forecaster(n_samples=np.int64(3)).n_samples) is int
    assert type(Pipeline(window_length=np.int64(3)).window_length) is int
    settings = diagnostic_settings(n_calibration=np.int64(9), dependence_seed_offset=np.int64(0))
    assert type(settings["n_calibration"]) is int
    assert type(settings["dependence_seed_offset"]) is int
    assert Scenario("trend", context_length=np.int64(8)).generate(seed=np.int64(1)).manifest["seed"] == 1


@pytest.mark.parametrize("value", [np.array([1 + 2j]), np.array([1 + 0j], dtype=object)])
def test_shared_real_coercion_rejects_complex_values_including_object_arrays(value):
    from timegym._internal import _real_array
    from timegym.metrics import mse

    with pytest.raises(ValueError, match="real-valued"):
        _real_array(value, "value")
    with pytest.raises(ValueError, match="real-valued"):
        mse(value, np.ones(1))
