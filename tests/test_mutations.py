from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from timegym import Scenario
from timegym.mutations import MUTATIONS, mutate_forecast


CONTRACT_CODES = {
    "index_shift": ["index_mismatch"],
    "target_swap": ["target_mismatch"],
    "horizon_truncation": ["index_mismatch"],
    "nonfinite_point": ["nonfinite_point"],
    "quantile_crossing": ["quantile_crossing"],
    "sample_axis_swap": ["sample_shape"],
    "variance_shrink": [],
    "dependence_shuffle": [],
}


def assert_forecast_equal(actual, expected):
    assert actual.index.equals(expected.index)
    assert actual.target_names == expected.target_names
    assert actual.levels == expected.levels
    for attr in ("point", "quantiles", "samples"):
        np.testing.assert_array_equal(getattr(actual, attr), getattr(expected, attr))


@pytest.mark.parametrize("name", MUTATIONS)
def test_fixture_validator_contract_and_bypass(name):
    assert set(CONTRACT_CODES) == set(MUTATIONS)
    case = Scenario("ar1", n_targets=2, horizon=4).generate(seed=2)
    forecast = case.reference.forecast(n_samples=32, seed=5)
    original = deepcopy(forecast)
    mutated = mutate_forecast(forecast, name)
    assert [item["code"] for item in mutated.validate(case.task)] == CONTRACT_CODES[name]
    bypass = mutate_forecast(forecast, name, enabled=False)
    assert bypass.validate(case.task) == []
    assert_forecast_equal(bypass, original)
    assert_forecast_equal(forecast, original)
    for attr in ("point", "quantiles", "samples"):
        assert not np.shares_memory(getattr(bypass, attr), getattr(forecast, attr))


@pytest.mark.parametrize("name, change", [
    ("index_shift", lambda f: setattr(f, "index", pd.date_range("2026-01-01", periods=3))),
    ("target_swap", lambda f: setattr(f, "target_names", ("y",))),
    ("nonfinite_point", lambda f: setattr(f, "point", None)),
    ("quantile_crossing", lambda f: setattr(f, "quantiles", None)),
    ("sample_axis_swap", lambda f: setattr(f, "samples", None)),
    ("variance_shrink", lambda f: setattr(f, "point", None)),
    ("dependence_shuffle", lambda f: setattr(f, "samples", None)),
])
@pytest.mark.parametrize("enabled", [True, False])
def test_fixture_and_bypass_reject_the_same_missing_precondition(name, change, enabled):
    forecast = Scenario("ar1", horizon=3, n_targets=2).generate().reference.forecast(24)
    change(forecast)
    with pytest.raises(ValueError, match=name):
        mutate_forecast(forecast, name, enabled=enabled)


@pytest.mark.parametrize("enabled", [True, False])
def test_fixture_and_bypass_coerce_arrays_without_changing_input(enabled):
    forecast = Scenario("trend", horizon=3).generate().reference.forecast(24)
    forecast.point = forecast.point.astype(int).tolist()
    result = mutate_forecast(forecast, "index_shift", enabled=enabled)
    assert isinstance(result.point, np.ndarray)
    assert result.point.dtype == float
    assert isinstance(forecast.point, list)


@pytest.mark.parametrize("strength", [0.0, 0.5, 1.0])
def test_variance_shrink_strength_scales_deviations(strength):
    forecast = Scenario("ar1", horizon=3).generate().reference.forecast(24)
    result = mutate_forecast(forecast, "variance_shrink", strength=strength)
    for attr, point in (("samples", forecast.point[None]), ("quantiles", forecast.point[..., None])):
        np.testing.assert_allclose(getattr(result, attr) - point,
                                   (1 - 0.75 * strength) * (getattr(forecast, attr) - point))
    if strength == 0:
        assert_forecast_equal(result, forecast)


def test_target_swap_reverses_all_target_labels():
    forecast = Scenario("ar1", n_targets=3).generate().reference.forecast(24)
    assert mutate_forecast(forecast, "target_swap").target_names == tuple(reversed(forecast.target_names))
