import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from timegym.data.synthetic import default_scenarios
from timegym.features import add_lagged_target, add_time_features, add_tsfel_features, cyclical_encoding
from timegym.smoothing import exponential_moving_average
from timegym.target import differencing, fractional_differencing, fractional_weights, multioutput_target

# These helpers are deprecated until TimeGym 2.0; tests/test_deprecations.py checks their warnings.
pytestmark = pytest.mark.filterwarnings(r"ignore:.* is deprecated and will be removed in TimeGym 2\.0:DeprecationWarning")


def test_cycles_do_not_collapse_last_label_onto_first():
    cosine, sine = cyclical_encoding([0, 6, 7], period=7, min_value=0)
    np.testing.assert_allclose([cosine[0], sine[0]], [cosine[2], sine[2]], atol=1e-14)
    assert not np.allclose([cosine[0], sine[0]], [cosine[1], sine[1]])
    np.testing.assert_allclose(cosine[1], np.cos(12 * np.pi / 7))
    hours = cyclical_encoding([0, 23], period=24)
    assert not np.allclose([hours[0][0], hours[1][0]], [hours[0][1], hours[1][1]])


def test_explicit_zero_bounds_and_unobserved_origin_are_honoured():
    cosine, sine = cyclical_encoding([-2, -1, 0], min_value=-2, max_value=0)
    np.testing.assert_allclose(cosine, [1, -0.5, -0.5], atol=1e-14)
    np.testing.assert_allclose(sine, [0, np.sqrt(3) / 2, -np.sqrt(3) / 2], atol=1e-14)
    subset = cyclical_encoding([3, 4], min_value=0, period=7)
    whole = cyclical_encoding(np.arange(7), min_value=0, max_value=6)
    np.testing.assert_array_equal(subset[0], whole[0][3:5])
    np.testing.assert_array_equal(subset[1], whole[1][3:5])
    with pytest.raises(ValueError, match="explicit"):
        cyclical_encoding([2, 3])


def test_calendar_features_have_stable_schema_and_subset_values():
    index = pd.date_range("2024-02-28 22:00:00.000001", periods=5, freq="h", tz="Europe/Lisbon")
    frame = pd.DataFrame({"y": np.arange(5)}, index=index)
    combined = add_time_features(frame)
    independently = pd.concat([add_time_features(frame.iloc[:3]), add_time_features(frame.iloc[3:])])
    pd.testing.assert_frame_equal(combined, independently)
    assert np.isfinite(combined.to_numpy()).all()
    np.testing.assert_allclose(combined.microsecond_sin, np.sin(2 * np.pi / 1_000_000))
    assert "weekday_cos" in combined and "year" in combined
    assert combined.iloc[1].hour_cos != combined.iloc[2].hour_cos
    singleton = add_time_features(frame.iloc[:1])
    assert np.isfinite(singleton.to_numpy()).all()
    assert list(frame.columns) == ["y"]


def test_calendar_features_require_datetime_index():
    with pytest.raises(ValueError, match="DatetimeIndex"):
        add_time_features(pd.DataFrame({"y": [1, 2]}))


@pytest.mark.parametrize("order", [2, 2.0, np.int64(2)])
def test_differencing_order_is_repeated_difference_not_lag(order):
    series = pd.Series(np.arange(6, dtype=float) ** 2, name="quadratic")
    result = differencing(series, order=order)
    np.testing.assert_array_equal(result.to_numpy(), [2, 2, 2, 2])
    np.testing.assert_array_equal(result.index, [2, 3, 4, 5])
    lagged = differencing(series, order=1, lag=2)
    np.testing.assert_array_equal(lagged.to_numpy(), [4, 8, 12, 16])


def test_zero_order_is_an_unchanged_independent_copy():
    series = pd.Series([1.0, np.nan, 3.0])
    result = differencing(series, order=0)
    pd.testing.assert_series_equal(result, series)
    result.iloc[0] = 100
    assert series.iloc[0] == 1
    pd.testing.assert_series_equal(fractional_differencing(series, 0, 3), series)


def test_fractional_differencing_warmup_and_missing_values():
    np.testing.assert_allclose(fractional_weights(0.5, 3).ravel(), [1, -0.5, -0.125])
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    result = fractional_differencing(series, 0.5, 3)
    np.testing.assert_array_equal(result.index, [2, 3, 4])
    np.testing.assert_allclose(result, [1.875, 2.25, 2.625])
    series.iloc[1] = np.nan
    missing = fractional_differencing(series, 0.5, 3)
    assert missing.iloc[:2].isna().all()
    assert missing.iloc[2] == 2.625


def test_ewm_rejects_unsupported_median_and_missing_alpha():
    series = pd.Series([0.0, 10.0])
    with pytest.raises(ValueError, match="median"):
        exponential_moving_average(series, exponential_alpha=0.5, statistic="median")
    with pytest.raises(ValueError, match="exponential_alpha"):
        exponential_moving_average(series)
    np.testing.assert_allclose(exponential_moving_average(series, exponential_alpha=0.5), [0.0, 20 / 3])


def test_tsfel_is_optional_and_missing_dependency_has_actionable_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "tsfel", None)
    with pytest.raises(ImportError, match=r"timegym\[features\]"):
        add_tsfel_features(pd.DataFrame({"y": [0, 1, 2]}), window_size=2)


def test_tsfel_windows_are_aligned_to_declared_lag(monkeypatch):
    windows = []

    def extract(config, window, **kwargs):
        windows.append(window.y.tolist())
        return pd.DataFrame({"last_window_value": [window.y.iloc[-1]]})

    mock = SimpleNamespace(get_features_by_domain=lambda **kwargs: {}, time_series_features_extractor=extract)
    monkeypatch.setitem(sys.modules, "tsfel", mock)
    frame = pd.DataFrame({"y": [0, 1, 2, 3, 4]})
    result = add_tsfel_features(frame, window_size=2, lag=1)
    assert windows == [[0, 1], [1, 2], [2, 3]]
    assert result.index.tolist() == [2, 3, 4]
    assert result.last_window_value.tolist() == [1, 2, 3]
    current = add_tsfel_features(frame, window_size=2, lag=0)
    assert current.index.tolist() == [1, 2, 3, 4]
    assert current.last_window_value.tolist() == [1, 2, 3, 4]
    assert frame.columns.tolist() == ["y"]


def test_target_and_feature_helpers_preserve_direction():
    frame = pd.DataFrame({"y": [10, 20, 30, 40]})
    past = add_lagged_target(frame, lags=1)
    assert past["y_t-1"].tolist() == [10, 20, 30]
    future = multioutput_target(frame, horizon=1)
    assert future["y_t+1"].tolist() == [20, 30, 40]


def test_synthetic_defaults_are_fresh_explicit_scenario_specifications():
    first = default_scenarios(context_length=8, horizon=3, n_targets=2, seed=3)
    second = default_scenarios(context_length=8, horizon=3, n_targets=2, seed=3)
    assert [scenario.kind for scenario in first] == ["white_noise", "trend", "seasonal", "ar1", "random_walk"]
    assert all(a is not b for a, b in zip(first, second))
    assert all(scenario.generate().reference.mean.shape == (3, 2) for scenario in first)


def test_all_runtime_modules_import_without_optional_libraries():
    script = """
import importlib
import importlib.abc
import pkgutil
import sys

blocked = {'tsfel', 'tqdm', 'matplotlib', 'sklearn', 'gluonts', 'sktime', 'yfinance'}
class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in blocked:
            raise ImportError('Optional dependency was imported: ' + fullname)
sys.meta_path.insert(0, BlockOptional())
import timegym
for module in pkgutil.walk_packages(timegym.__path__, 'timegym.'):
    importlib.import_module(module.name)
assert not blocked.intersection(sys.modules)
"""
    completed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
