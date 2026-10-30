"""Boundary inputs must not silently become unreplayable or misaligned tasks."""

from copy import deepcopy
import warnings

import numpy as np
import pandas as pd
import pytest

from timegym import Forecast, ForecastTask, Pipeline, Reference, Scenario
from timegym.pipeline import replay_case


@pytest.mark.parametrize("window", [0, -1, True, False, np.bool_(True), 1.5, "12", None])
def test_invalid_window_rejected_before_a_report_can_claim_replayability(window):
    with pytest.raises(ValueError, match="window_length must be a positive integer"):
        Pipeline(window_length=window)


def test_numpy_window_is_normalized_and_remains_replayable():
    pipeline = Pipeline(window_length=np.int64(4))
    assert type(pipeline.window_length) is int
    report = pipeline.test([Scenario("white_noise", context_length=8, horizon=2)], seeds=[1])
    assert report.metadata["pipeline_config"]["replayable"]
    assert report.replay_case().metadata["replay"]["metrics_match"]


@pytest.mark.parametrize("index", [pd.Index([np.inf]), pd.Index([np.nan]),
                                   pd.DatetimeIndex([pd.NaT]), pd.TimedeltaIndex([pd.NaT])])
def test_missing_and_infinite_forecast_labels_are_diagnostic_failures(index):
    violations = Forecast(index, point=np.ones((1, 1))).validate()
    assert {item["code"] for item in violations} & {"index_missing", "index_nonfinite"}
    with pytest.raises(ValueError, match="index must not contain"):
        Reference(mean=np.zeros((1, 1)), variance=np.ones((1, 1)), covariance=np.eye(1), index=index)


def test_task_rejects_infinite_context_and_future_indexes():
    with pytest.raises(ValueError, match="context index.*nonfinite"):
        ForecastTask(pd.DataFrame({"y": [1.0, 2.0]}, index=[-np.inf, 0]), pd.Index([1]))
    with pytest.raises(ValueError, match="future index.*nonfinite"):
        ForecastTask(pd.DataFrame({"y": [1.0]}, index=[0]), pd.Index([np.inf]))


@pytest.mark.parametrize("names", ["y", [None], [np.nan], [pd.NA], [np.inf], ["y", "y"], [["unhashable"]]])
def test_reference_and_forecast_reject_ambiguous_target_names(names):
    with pytest.raises(ValueError, match="target names"):
        Reference(mean=np.zeros((1, 1)), variance=np.ones((1, 1)), covariance=np.eye(1),
                  index=pd.RangeIndex(1), target_names=names)
    violations = Forecast(pd.RangeIndex(1), point=np.zeros((1, 1)), target_names=names).validate()
    assert "target_names" in {item["code"] for item in violations}


@pytest.mark.parametrize("name", [None, np.nan, pd.NA, np.inf])
def test_task_rejects_missing_target_labels(name):
    with pytest.raises(ValueError, match="context target names"):
        ForecastTask(pd.DataFrame([[1.0]], columns=[name]), pd.RangeIndex(1, 2))


def test_numeric_and_tuple_target_labels_remain_valid():
    label_sets = [(0,), (("building", "load"), ("building", "temperature")),
                  (("building", ("floor", 1)), ("building", ("floor", 2)))]
    columns_to_check = [pd.Index(names, tupleize_cols=False) for names in label_sets]
    columns_to_check.append(pd.MultiIndex.from_tuples(label_sets[1]))
    for columns in columns_to_check:
        names = tuple(columns)
        context = pd.DataFrame(np.ones((3, len(names))), columns=columns)
        task = ForecastTask(context, pd.RangeIndex(3, 5))
        forecast = Forecast(task.future_index, point=np.ones((2, len(names))), target_names=names)
        assert forecast.validate(task) == []


@pytest.mark.parametrize("component", [None, np.nan, pd.NA, pd.NaT, np.inf])
@pytest.mark.parametrize("representation", ["multiindex", "tuple", "nested_tuple"])
def test_missing_or_nonfinite_tuple_components_are_invalid_target_labels(component, representation):
    label = ("building", ("floor", component)) if representation == "nested_tuple" else ("building", component)
    columns = (pd.MultiIndex.from_tuples([label]) if representation == "multiindex"
               else pd.Index([label], tupleize_cols=False))
    with pytest.raises(ValueError, match="context target names"):
        ForecastTask(pd.DataFrame([[1.0]], columns=columns), pd.RangeIndex(1, 2))
    names = tuple(columns)
    violations = Forecast(pd.RangeIndex(1), point=np.ones((1, 1)), target_names=names).validate()
    assert "target_names" in {item["code"] for item in violations}
    with pytest.raises(ValueError, match="Reference target names"):
        Reference(mean=np.zeros((1, 1)), variance=np.ones((1, 1)), covariance=np.eye(1),
                  index=pd.RangeIndex(1), target_names=names)


def test_replay_schema_versions_do_not_accept_boolean_aliases_for_integers():
    config = Pipeline().to_config()
    config["schema_version"] = True
    with pytest.raises(ValueError, match="configuration schema"):
        Pipeline.from_config(config)
    report = Pipeline().test([Scenario("white_noise", context_length=8, horizon=2)], seeds=[1]).to_dict()
    changed = deepcopy(report)
    changed["cases"][0]["manifest"]["schema_version"] = True
    with pytest.raises(ValueError, match="scenario manifest schema"):
        replay_case(changed)


@pytest.mark.parametrize("values, crossing", [([-1e308, 1e308], False), ([1e308, -1e308], True)])
def test_extreme_quantile_order_is_checked_without_subtraction_overflow(values, crossing):
    forecast = Forecast(pd.RangeIndex(1), quantiles=np.array(values).reshape(1, 1, 2),
                        levels=(0.1, 0.9))
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        violations = forecast.validate()
    assert ("quantile_crossing" in {item["code"] for item in violations}) == crossing
