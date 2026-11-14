import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from timegym.features import (
    add_lagged_target,
    add_time_features,
    add_tsfel_features,
    cyclical_encoding,
)
from timegym.smoothing import exponential_moving_average, simple_moving_average
from timegym.target import (
    differencing,
    fractional_differencing,
    multioutput_target,
    onestep_target,
    step_ahead_target,
)

# These helpers are deprecated until TimeGym 2.0; tests/test_deprecations.py checks their warnings.
pytestmark = pytest.mark.filterwarnings(r"ignore:.* is deprecated and will be removed in TimeGym 2\.0:DeprecationWarning")


@pytest.fixture
def frame():
    return pd.DataFrame(
        {"a": [0.0, 2.0, 4.0, 8.0], "b": [8.0, 4.0, 2.0, 0.0]},
        index=pd.date_range("2024-01-01", periods=4, freq="2D"),
    )


@pytest.mark.parametrize("as_series", [False, True])
@pytest.mark.parametrize("kind", ["sma", "ema"])
def test_smoothing_preserves_type_labels_all_columns_and_input(frame, as_series, kind):
    data = frame.a if as_series else frame
    original = data.copy(deep=True)
    if kind == "sma":
        result = simple_moving_average(data, window_length=2)
        expected = pd.DataFrame(
            {"a": [0.0, 1.0, 3.0, 6.0], "b": [8.0, 6.0, 3.0, 1.0]}, index=frame.index
        )
    else:
        result = exponential_moving_average(data, exponential_alpha=0.5)
        expected = pd.DataFrame(
            {"a": [0.0, 4 / 3, 20 / 7, 28 / 5], "b": [8.0, 16 / 3, 24 / 7, 8 / 5]},
            index=frame.index,
        )
    if as_series:
        pd.testing.assert_series_equal(result, expected.a)
        pd.testing.assert_series_equal(data, original)
    else:
        pd.testing.assert_frame_equal(result, expected)
        pd.testing.assert_frame_equal(data, original)
    result.iloc[0] = 999
    if as_series:
        pd.testing.assert_series_equal(data, original)
    else:
        pd.testing.assert_frame_equal(data, original)


@pytest.mark.parametrize(
    "function, kwargs",
    [
        (simple_moving_average, {"window_length": 2}),
        (exponential_moving_average, {"exponential_alpha": 0.5}),
    ],
)
def test_smoothing_is_causal_and_respects_per_column_missing_history(
    frame, function, kwargs
):
    frame.loc[frame.index[1], "a"] = np.nan
    frame.loc[frame.index[0], "b"] = np.nan
    result = function(frame, min_periods=2, **kwargs)
    pd.testing.assert_frame_equal(
        result.iloc[:3], function(frame.iloc[:3], min_periods=2, **kwargs)
    )
    assert result.iloc[:2].isna().all().all()
    assert result.b.iloc[2] == pytest.approx(
        3 if function is simple_moving_average else 8 / 3
    )
    assert result.a.iloc[3] == pytest.approx(
        6 if function is simple_moving_average else 80 / 13
    )


@pytest.mark.parametrize("value", [0, -1, True, 1.5, np.nan])
def test_sma_requires_positive_integer_window(frame, value):
    with pytest.raises(ValueError, match="window_length"):
        simple_moving_average(frame, value)


@pytest.mark.parametrize("value", [-1, True, 1.5])
def test_smoothing_requires_nonnegative_integer_min_periods(frame, value):
    with pytest.raises(ValueError, match="min_periods"):
        simple_moving_average(frame, min_periods=value)
    with pytest.raises(ValueError, match="min_periods"):
        exponential_moving_average(frame, 0.5, min_periods=value)


def test_sma_rejects_min_periods_larger_than_window(frame):
    with pytest.raises(ValueError, match="must not exceed"):
        simple_moving_average(frame, window_length=2, min_periods=3)


@pytest.mark.parametrize(
    "alpha", [None, 0, -0.1, 1.1, np.nan, np.inf, True, np.bool_(True), "0.5"]
)
def test_ema_requires_explicit_finite_alpha(frame, alpha):
    with pytest.raises(ValueError, match="exponential_alpha"):
        exponential_moving_average(frame, alpha)


def test_smoothing_boundaries(frame):
    pd.testing.assert_frame_equal(
        simple_moving_average(frame, np.int64(1), min_periods=0), frame
    )
    pd.testing.assert_frame_equal(
        exponential_moving_average(frame, np.float64(1), min_periods=0), frame
    )


def test_lag_features_preserve_every_target_and_align_missing_warmup(frame):
    original = frame.copy(deep=True)
    result = add_lagged_target(frame, lags=2, dropna=False)
    assert result.columns.tolist() == ["a", "b", "a_t-1", "a_t-2", "b_t-1", "b_t-2"]
    assert result.index.equals(frame.index)
    assert result.iloc[:2][["a_t-2", "b_t-2"]].isna().all().all()
    assert result.iloc[2][["a_t-1", "a_t-2", "b_t-1", "b_t-2"]].tolist() == [2, 0, 4, 8]
    pd.testing.assert_frame_equal(add_lagged_target(frame, lags=2), result.iloc[2:])
    pd.testing.assert_frame_equal(add_lagged_target(frame, lags=0), frame)
    pd.testing.assert_frame_equal(frame, original)


def test_future_labels_preserve_target_order_and_incomplete_tail(frame):
    original = frame.copy(deep=True)
    result = multioutput_target(frame, horizon=2, dropna=False)
    assert result.columns.tolist() == ["a", "b", "a_t+1", "a_t+2", "b_t+1", "b_t+2"]
    assert result.index.equals(frame.index)
    assert result.iloc[-2:][["a_t+2", "b_t+2"]].isna().all().all()
    assert result.iloc[0].tolist() == [0, 8, 2, 4, 4, 2]
    pd.testing.assert_frame_equal(multioutput_target(frame, 2), result.iloc[:2])
    pd.testing.assert_frame_equal(onestep_target(frame), multioutput_target(frame, 1))
    pd.testing.assert_frame_equal(
        step_ahead_target(frame, 2), result.iloc[:2][["a", "b", "a_t+2", "b_t+2"]]
    )
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize(
    "function, name",
    [
        (add_lagged_target, "a_t-1"),
        (multioutput_target, "a_t+1"),
        (onestep_target, "a_t+1"),
        (step_ahead_target, "a_t+5"),
        (add_time_features, "month_cos"),
    ],
)
def test_preparation_rejects_overwriting_original_columns(frame, function, name):
    frame[name] = -99
    original = frame.copy(deep=True)
    with pytest.raises(ValueError, match="already exist"):
        function(frame)
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize(
    "function",
    [add_lagged_target, multioutput_target, step_ahead_target, add_time_features],
)
def test_preparation_rejects_duplicate_column_labels(frame, function):
    frame.columns = ["y", "y"]
    with pytest.raises(ValueError, match="columns must be unique"):
        function(frame)


def test_frame_differences_have_explicit_warmup_and_preserve_columns(frame):
    original = frame.copy(deep=True)
    integer = differencing(frame, order=2)
    pd.testing.assert_frame_equal(
        integer, pd.DataFrame({"a": [0.0, 2.0], "b": [2.0, 0.0]}, index=frame.index[2:])
    )
    fractional = fractional_differencing(frame, 0.5, 2, lag=2)
    pd.testing.assert_frame_equal(
        fractional,
        pd.DataFrame({"a": [4.0, 7.0], "b": [-2.0, -2.0]}, index=frame.index[2:]),
    )
    pd.testing.assert_frame_equal(differencing(frame, 0.5, 2, lag=2), fractional)
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize("order", [-1, np.nan, np.inf, True, np.bool_(False)])
def test_differences_reject_invalid_orders(frame, order):
    with pytest.raises(ValueError, match="order"):
        differencing(frame, order)
    with pytest.raises(ValueError, match="order"):
        fractional_differencing(frame, order, 3)


def test_tsfel_subset_warmup_and_input_are_preserved(frame, monkeypatch):
    original = frame.copy(deep=True)

    def extract(config, window, **kwargs):
        assert window.columns.tolist() == ["b"]
        return pd.DataFrame({"mean_b": [window.b.mean()]})

    monkeypatch.setitem(
        sys.modules,
        "tsfel",
        SimpleNamespace(
            get_features_by_domain=lambda **kwargs: {},
            time_series_features_extractor=extract,
        ),
    )
    result = add_tsfel_features(frame, use_cols=("b",), window_size=2, dropna=False)
    assert result.columns.tolist() == ["b", "mean_b"]
    assert result.index.equals(frame.index)
    assert result.mean_b.iloc[:2].isna().all()
    assert result.mean_b.iloc[2:].tolist() == [6.0, 3.0]
    pd.testing.assert_frame_equal(frame, original)


def test_cyclical_features_reject_complex_values():
    with pytest.raises(ValueError, match="real-valued"):
        cyclical_encoding([1 + 2j], period=12)
