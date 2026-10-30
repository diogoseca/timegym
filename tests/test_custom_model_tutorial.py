import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from timegym import Forecast, ForecastTask, Pipeline, Scenario
from timegym.contracts import DEFAULT_LEVELS


@pytest.fixture(scope="module")
def model_class():
    path = Path(__file__).resolve().parents[1] / "notebooks/28_custom_model.ipynb"
    notebook = json.loads(path.read_text())
    source = next("".join(c["source"]) for c in notebook["cells"]
                  if c["id"] == "custom-model-definition")
    namespace = {"np": np, "pd": pd, "Forecast": Forecast,
                 "ForecastTask": ForecastTask, "DEFAULT_LEVELS": DEFAULT_LEVELS}
    exec(compile(source, str(path), "exec"), namespace)
    return namespace["NearestAnalogues"]


def test_analogues_preserve_whole_continuations_and_never_use_a_partial_future(model_class):
    # Exact historical match [0, 1] has continuation [8, 9]. The latest
    # matching window has no observed continuation and must be excluded.
    context = pd.DataFrame({"a": [0, 1, 8, 9, 0, 1],
                            "b": [0, 10, 80, 90, 0, 10]})
    model = model_class(window_length=2, k=1).fit(context)
    forecast = model.predict(pd.RangeIndex(6, 8))
    np.testing.assert_array_equal(forecast.samples, [[[8, 80], [9, 90]]])
    np.testing.assert_array_equal(forecast.point, [[8, 80], [9, 90]])
    assert forecast.target_names == ("a", "b")
    assert forecast.validate(ForecastTask(context, pd.RangeIndex(6, 8))) == []
    assert model.neighbor_starts_.tolist() == [0]


def test_analogues_average_k_paths_and_have_stable_ties(model_class):
    context = pd.DataFrame({"y": [0, 1, 2, 0, 3, 4, 0]})
    model = model_class(window_length=1, k=2).fit(context)
    forecast = model.predict(pd.RangeIndex(7, 9))
    np.testing.assert_array_equal(forecast.samples, [[[1], [2]], [[3], [4]]])
    np.testing.assert_array_equal(forecast.point, [[2], [3]])
    np.testing.assert_allclose(forecast.quantiles, [[[1.2, 2, 2.8]], [[2.2, 3, 3.8]]])
    assert model.neighbor_starts_.tolist() == [0, 3]


def test_analogues_fit_snapshot_and_unfitted_clones(model_class):
    context = pd.DataFrame({"y": [0, 1, 8, 9, 0, 1]})
    model = model_class(window_length=2, k=1).fit(context)
    expected = model.predict(pd.RangeIndex(6, 8)).point.copy()
    context.iloc[:] = -999
    np.testing.assert_array_equal(model.predict(pd.RangeIndex(6, 8)).point, expected)
    clone = model.clone_unfitted()
    assert clone is not model and clone.params() == model.params()
    with pytest.raises(RuntimeError, match="Call fit"):
        clone.predict(pd.RangeIndex(6, 8))
    model.fit(pd.DataFrame({"y": [5, 5, 5, 5, 5, 5]}))
    np.testing.assert_array_equal(model.predict(pd.RangeIndex(6, 8)).point, [[5], [5]])


@pytest.mark.parametrize("parameter,value", [("k", 0), ("k", True), ("k", 1.5),
                                            ("window_length", -1),
                                            ("window_length", False)])
def test_analogues_reject_invalid_settings(model_class, parameter, value):
    with pytest.raises(ValueError, match=f"{parameter} must be a positive integer"):
        model_class(**{parameter: value})


def test_analogues_reject_insufficient_history_and_invalid_indexes(model_class):
    model = model_class(window_length=2, k=2).fit(pd.DataFrame({"y": [0, 1, 2, 3]}))
    with pytest.raises(ValueError, match="context rows"):
        model.predict(pd.RangeIndex(4, 6))
    with pytest.raises(ValueError, match="after"):
        model.predict(pd.RangeIndex(3, 5))
    with pytest.raises(TypeError, match="DataFrame"):
        model.fit(np.zeros(20))


def test_pipeline_refits_tutorial_model_without_fitting_template(model_class):
    template = model_class(window_length=3, k=3)
    pipeline = Pipeline(template)
    scenario = Scenario("seasonal", context_length=40, horizon=4)
    report = pipeline.test([scenario], seeds=[7, 19])
    assert not hasattr(template, "context_")
    assert all(any(d.name == "forecast_contract" and d.status == "PASS"
                   for d in result.diagnostics) for result in report.cases)
    assert all("mean_reference_mse" in result.metrics for result in report.cases)
    assert pipeline.to_config()["replayable"] is False
