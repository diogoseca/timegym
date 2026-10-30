import numpy as np
import pandas as pd
import pytest

from timegym.contracts import Case, Forecast, ForecastTask, Reference
from timegym.diagnostics import evaluate_forecast


def test_function_reference_preserves_joint_law_and_owns_its_inputs():
    index = pd.period_range("2000-01", periods=2, freq="M")
    means = np.array([[1.0, 10.0], [2.0, 20.0]])
    factor = np.array([[1, 0, 0, 0], [0.5, 2, 0, 0],
                       [0.2, 0, 1, 0], [0, 0.3, 0, 2]])
    covariance = factor @ factor.T
    metadata = {"source": {"name": "fixed test curve"}}
    calls = []

    def truth(requested):
        calls.append(requested)
        return pd.DataFrame(means, index=requested, columns=["a", "b"])

    supplied = Reference.from_function(
        truth, index, covariance=covariance, target_names=("a", "b"),
        conditioning=metadata,
    )
    direct = Reference(means.copy(), np.diag(covariance).reshape(2, 2).copy(),
                       covariance.copy(), index, ("a", "b"))
    means[:] = -999
    covariance[:] = 0
    metadata["source"]["name"] = "changed"
    assert len(calls) == 1 and calls[0].equals(index) and calls[0] is not index
    assert supplied.kind == "user_supplied_gaussian"
    assert supplied.conditioning["source"]["name"] == "fixed test curve"
    np.testing.assert_array_equal(supplied.mean, direct.mean)
    np.testing.assert_allclose(supplied.quantile(), direct.quantile())
    np.testing.assert_allclose(supplied.sample(100, seed=9), direct.sample(100, seed=9))
    assert len(calls) == 1


@pytest.mark.parametrize("mean,match", [
    (np.zeros(2), "shape"),
    (np.zeros((1, 2)), "shape"),
    (np.array([[np.nan], [0]]), "finite"),
    (np.array([[np.inf], [0]]), "finite"),
    (np.ones((2, 1), dtype=complex), "real"),
])
def test_function_reference_rejects_invalid_means(mean, match):
    with pytest.raises(ValueError, match=match):
        Reference.from_function(lambda index: mean, pd.RangeIndex(2), covariance=np.eye(2))


@pytest.mark.parametrize("covariance,match", [
    (np.ones(2), "shape"),
    (np.eye(3), "shape"),
    (np.array([[1, 0.5], [0, 1]]), "symmetric"),
    (np.array([[1, 2], [2, 1]]), "positive semidefinite"),
    (-np.eye(2), "nonnegative"),
    (np.full((2, 2), np.nan), "finite"),
    (np.eye(2, dtype=complex), "real"),
])
def test_function_reference_rejects_invalid_covariance(covariance, match):
    with pytest.raises(ValueError, match=match):
        Reference.from_function(lambda index: np.zeros((2, 1)),
                                pd.RangeIndex(2), covariance=covariance)


@pytest.mark.parametrize("index", [pd.RangeIndex(0), [1, 2],
                                   pd.Index([2, 1]), pd.Index([1, 1]),
                                   pd.Index([1, np.nan]), pd.Index([1, np.inf])])
def test_function_reference_rejects_invalid_index(index):
    with pytest.raises(ValueError, match="index"):
        Reference.from_function(lambda requested: np.zeros((len(requested), 1)),
                                index, covariance=np.eye(len(index)))


@pytest.mark.parametrize("mismatch", ["index", "targets"])
def test_function_reference_never_silently_reorders_labelled_truth(mismatch):
    def truth(index):
        return pd.DataFrame(np.ones((2, 2)),
                            index=index[::-1] if mismatch == "index" else index,
                            columns=["b", "a"] if mismatch == "targets" else ["a", "b"])

    with pytest.raises(ValueError, match="index and target order"):
        Reference.from_function(truth, pd.RangeIndex(2), covariance=np.eye(4),
                                target_names=("a", "b"))


def test_function_reference_requires_callable_and_explicit_noise_law():
    with pytest.raises(TypeError, match="callable"):
        Reference.from_function(np.zeros((2, 1)), pd.RangeIndex(2), covariance=np.eye(2))
    with pytest.raises(TypeError, match="covariance"):
        Reference.from_function(lambda index: np.zeros((2, 1)), pd.RangeIndex(2))
    with pytest.raises(TypeError, match="conditioning"):
        Reference.from_function(lambda index: np.zeros((2, 1)), pd.RangeIndex(2),
                                covariance=np.eye(2), conditioning="not metadata")


def test_function_reference_scores_truth_without_exposing_it_to_the_model():
    task = ForecastTask(pd.DataFrame({"y": [1.0, 2.0, 3.0]}), pd.RangeIndex(3, 5))
    observed = pd.DataFrame({"y": [4.5, 4.5]}, index=task.future_index)
    supplied = Reference.from_function(lambda index: (index.to_numpy() + 1)[:, None],
                                       task.future_index, covariance=0.25 * np.eye(2))
    forecast = Forecast(task.future_index, point=np.array([[4.0], [5.0]]))
    manifest = {"kind": "user_curve", "seed": 12}
    known = evaluate_forecast(Case(task, observed, supplied, manifest), forecast)
    unknown = evaluate_forecast(Case(task, observed, None, manifest), forecast,
                                check_dependence=True)
    assert known.metrics["mean_reference_mse"] == 0
    assert known.metrics["mse"] == unknown.metrics["mse"] == 0.25
    assert "mean_reference_mse" not in unknown.metrics
    assert any(d.name == "reference_covariance" and d.status == "UNSUPPORTED"
               for d in unknown.diagnostics)
    assert not any(d.status in {"FAIL", "ERROR"} for d in known.diagnostics)
