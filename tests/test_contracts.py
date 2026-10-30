import numpy as np
import pandas as pd
import pytest

from timegym.contracts import Forecast, ForecastTask, Reference


def task():
    return ForecastTask(
        pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]}),
        pd.RangeIndex(2, 5),
    )


def codes(forecast, forecast_task=None):
    return {item["code"] for item in forecast.validate(forecast_task)}


def test_valid_complete_forecast_and_target_order():
    forecast_task = task()
    prediction = Forecast(
        index=forecast_task.future_index,
        point=np.zeros((3, 2)),
        quantiles=np.tile([-1.0, 0.0, 1.0], (3, 2, 1)),
        levels=(0.1, 0.5, 0.9),
        samples=np.zeros((10, 3, 2)),
        target_names=("a", "b"),
    )
    assert forecast_task.target_names == ("a", "b")
    assert prediction.validate(forecast_task) == []
    prediction.target_names = ("b", "a")
    assert "target_mismatch" in codes(prediction, forecast_task)


@pytest.mark.parametrize(
    "field,value,expected",
    [
        ("point", np.zeros(3), "point_shape"),
        ("point", np.zeros((2, 3)), "point_shape"),
        ("point", np.full((3, 2), np.nan), "nonfinite_point"),
        ("samples", np.zeros((3, 2, 10)), "sample_shape"),
        ("samples", np.zeros((0, 3, 2)), "sample_shape"),
        ("samples", np.full((4, 3, 2), np.inf), "nonfinite_samples"),
        ("quantiles", np.zeros((3, 3, 2)), "quantile_shape"),
        ("quantiles", np.full((3, 2, 3), np.inf), "nonfinite_quantiles"),
    ],
)
def test_malformed_outputs_are_constructible_and_diagnosed(field, value, expected):
    prediction = Forecast(
        pd.RangeIndex(2, 5),
        target_names=("a", "b"),
        levels=(0.1, 0.5, 0.9) if field == "quantiles" else (),
        **{field: value},
    )
    findings = prediction.validate(task())
    assert expected in {item["code"] for item in findings}
    assert all(set(item) == {"code", "message"} and item["message"] for item in findings)


@pytest.mark.parametrize(
    "levels,expected",
    [
        ((0.1, 0.1, 0.9), "duplicate_levels"),
        ((0.9, 0.5, 0.1), "nonmonotonic_levels"),
        ((0.0, 0.5, 1.0), "invalid_levels"),
        ((np.nan, 0.5, 0.9), "invalid_levels"),
        (("invalid", 0.5, 0.9), "invalid_levels"),
    ],
)
def test_invalid_quantile_levels(levels, expected):
    prediction = Forecast(pd.RangeIndex(3), quantiles=np.zeros((3, 1, 3)), levels=levels)
    assert expected in codes(prediction)


def test_crossing_quantiles_and_missing_predictions():
    prediction = Forecast(
        pd.RangeIndex(3),
        quantiles=np.tile([1.0, 0.0, 2.0], (3, 1, 1)),
        levels=(0.1, 0.5, 0.9),
    )
    assert "quantile_crossing" in codes(prediction)
    assert "missing_prediction" in codes(Forecast(pd.RangeIndex(3)))
    assert "invalid_levels" in codes(Forecast(pd.RangeIndex(3), quantiles=np.zeros((3, 1, 0))))
    assert "missing_quantiles" in codes(Forecast(pd.RangeIndex(3), point=np.zeros((3, 1)), levels=(0.5,)))


def test_malformed_indexes_and_names_are_diagnosed():
    assert "index_type" in codes(Forecast([0, 1], point=np.zeros((2, 1))))
    assert "index_empty" in codes(Forecast(pd.Index([]), point=np.zeros((0, 1))))
    prediction = Forecast(pd.Index([2, 2, 1]), point=np.zeros((3, 2)), target_names=("a", "b"))
    assert {"index_duplicate", "index_order", "index_mismatch"} <= codes(prediction, task())
    assert "target_names" in codes(Forecast(pd.RangeIndex(3), point=np.zeros((3, 1)), target_names="y"))
    assert "target_names" in codes(Forecast(pd.RangeIndex(3), point=np.zeros((3, 2)), target_names=("y", "y")))


def test_mean_helper_does_not_treat_median_as_expectation():
    prediction = Forecast(pd.RangeIndex(2), samples=np.arange(8).reshape(4, 2, 1))
    np.testing.assert_array_equal(prediction.mean(), [[3.0], [4.0]])
    prediction.point = np.array([[9.0], [10.0]])
    np.testing.assert_array_equal(prediction.mean(), prediction.point)
    with pytest.raises(ValueError, match="quantiles alone"):
        Forecast(pd.RangeIndex(2), quantiles=np.zeros((2, 1, 1)), levels=(0.5,)).mean()


def test_datetime_task_and_index_alignment():
    context_index = pd.date_range("2024-01-01", periods=3, tz="UTC")
    future_index = pd.date_range("2024-01-04", periods=2, tz="UTC")
    forecast_task = ForecastTask(pd.DataFrame({"y": [1, 2, 3]}, index=context_index), future_index)
    assert Forecast(future_index, point=np.zeros((2, 1))).validate(forecast_task) == []
    wrong = Forecast(pd.date_range("2024-01-05", periods=2, tz="UTC"), point=np.zeros((2, 1)))
    assert "index_mismatch" in codes(wrong, forecast_task)


@pytest.mark.parametrize("values", [[1, np.nan], [1, np.inf], ["a", "b"]])
def test_invalid_context_rejected(values):
    with pytest.raises(ValueError):
        ForecastTask(pd.DataFrame({"y": values}), pd.RangeIndex(2, 4))


def test_context_and_future_must_be_nonempty_ordered_and_disjoint():
    with pytest.raises(ValueError):
        ForecastTask(pd.DataFrame({"y": []}), pd.RangeIndex(2, 4))
    with pytest.raises(ValueError, match="after"):
        ForecastTask(pd.DataFrame({"y": [1, 2]}), pd.RangeIndex(1, 4))
    with pytest.raises(ValueError, match="increasing"):
        ForecastTask(pd.DataFrame({"y": [1, 2]}), pd.Index([3, 2]))


def test_reference_rejects_invalid_covariance_and_variance():
    kwargs = dict(mean=np.zeros((2, 1)), variance=np.ones((2, 1)), index=pd.RangeIndex(2))
    with pytest.raises(ValueError, match="positive semidefinite"):
        Reference(covariance=np.array([[1.0, 2.0], [2.0, 1.0]]), **kwargs)
    with pytest.raises(ValueError, match="symmetric"):
        Reference(covariance=np.array([[1.0, 0.5], [0.0, 1.0]]), **kwargs)
    with pytest.raises(ValueError, match="diagonal"):
        Reference(covariance=2 * np.eye(2), **kwargs)
    with pytest.raises(ValueError, match="finite"):
        Reference(covariance=np.full((2, 2), np.nan), **kwargs)


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("representation", ["point", "quantiles", "samples"])
@pytest.mark.parametrize("dtype", [complex, object])
@pytest.mark.parametrize("imaginary", [0.0, 2.0])
def test_complex_forecasts_are_diagnosed_without_lossy_conversion(representation, dtype, imaginary):
    shapes = {"point": (2, 1), "quantiles": (2, 1, 3), "samples": (4, 2, 1)}
    values = np.full(shapes[representation], np.complex128(1 + imaginary * 1j), dtype=dtype)
    prediction = Forecast(
        pd.RangeIndex(2),
        levels=(0.1, 0.5, 0.9) if representation == "quantiles" else (),
        **{representation: values},
    )
    assert f"nonreal_{representation}" in codes(prediction)
    np.testing.assert_array_equal(getattr(prediction, representation), values)
    if representation in {"point", "samples"}:
        with pytest.raises(ValueError, match="real"):
            prediction.mean()


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("dtype", [complex, object])
def test_complex_levels_and_context_are_rejected_without_conversion(dtype):
    levels = np.array([np.complex128(0.1 + 0.2j), 0.5, 0.9], dtype=dtype)
    prediction = Forecast(pd.RangeIndex(2), quantiles=np.zeros((2, 1, 3)), levels=levels)
    assert "invalid_levels" in codes(prediction)
    context = pd.DataFrame({"y": np.array([np.complex128(1 + 2j), 3.0], dtype=dtype)})
    with pytest.raises(ValueError, match="real"):
        ForecastTask(context, pd.RangeIndex(2, 4))


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("field", ["mean", "variance", "covariance"])
@pytest.mark.parametrize("dtype", [complex, object])
def test_complex_reference_parameters_are_rejected_before_conversion(field, dtype):
    kwargs = dict(mean=np.zeros((2, 1)), variance=np.ones((2, 1)), covariance=np.eye(2), index=pd.RangeIndex(2))
    kwargs[field] = np.full(kwargs[field].shape, np.complex128(1 + 2j), dtype=dtype)
    with pytest.raises(ValueError, match="real"):
        Reference(**kwargs)


@pytest.mark.filterwarnings("error")
def test_reference_quantile_rejects_complex_levels():
    reference = Reference(mean=np.zeros((2, 1)), variance=np.ones((2, 1)), covariance=np.eye(2), index=pd.RangeIndex(2))
    with pytest.raises(ValueError, match="real"):
        reference.quantile(np.array([np.complex128(0.5 + 0.1j)], dtype=object))
