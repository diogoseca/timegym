"""The Part 6 zoo of protocol 3.0 on hand-made inputs. No test here scores a model or computes an error.

Context parroting runs on sequences whose continuation is known. The model wrappers run on a hand-made
DataFrame (a sine plus a ramp) for shapes and types only. The runner and the figure run on a stub
scorer that returns a fabricated scorecard. The seed test intercepts the scorecard before any cell is
scored.
"""
import copy
import json
import sys

import numpy as np
import pandas as pd
import pytest

import timegym
from timegym import Forecast, Pipeline
from timegym.models import NaiveForecaster

from experiments import study3_common as common
from experiments.zoo import figure, models, parroting, run_zoo

ZOO = common.ROOT / "experiments/zoo"


@pytest.fixture(scope="module")
def protocol():
    return common.load_protocol()


def _frame(length=120):
    steps = np.arange(length, dtype=float)
    return pd.DataFrame({"y": np.sin(2 * np.pi * steps / 12) + 0.05 * steps})


def _learners():
    try:
        from experiments import study3_learners
    except ImportError:
        return None
    return study3_learners


# Context parroting


def test_parroting_continues_a_periodic_sequence():
    period = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    values = np.tile(period, 5)
    start, distances = parroting.nearest_motif(values, 10)
    assert len(distances) == 35 - 2 * 10 + 1
    assert start == 4 and distances[4] == 0 and distances[11] == 0
    np.testing.assert_array_equal(parroting.parrot(values, 24), np.tile(period, 9)[35:59])


def test_parroting_breaks_ties_to_the_earliest_motif_and_repeats_the_copy():
    values = np.array([0.0, 1.0, 5.0, 0.0, 1.0, 7.0, 0.0, 1.0])
    start, distances = parroting.nearest_motif(values, 2)
    assert start == 0 and distances.tolist()[:4] == [0.0, pytest.approx(np.sqrt(17)), pytest.approx(np.sqrt(26)), 0.0]
    np.testing.assert_array_equal(parroting.parrot(values, 8, 2), [5, 0, 1, 7, 0, 1, 5, 0])
    near = np.array([3.0, 4.1, 9.0, 3.0, 4.0, 2.0, 3.0, 4.0])
    assert parroting.nearest_motif(near, 2)[0] == 3
    np.testing.assert_array_equal(parroting.parrot(near, 3, 2), [2.0, 3.0, 4.0])
    with pytest.raises(ValueError):
        parroting.parrot(np.arange(19.0), 24)


def test_parroting_forecasts_each_target_alone():
    frame = pd.DataFrame({"a": np.tile([1.0, 2.0, 3.0], 10), "b": np.tile([4.0, 5.0], 15)})
    model = parroting.ContextParroting(embedding=4)
    clone = model.clone_unfitted()
    assert clone is not model and clone.embedding == 4
    forecast = clone.fit(frame).predict(pd.RangeIndex(30, 35))
    assert isinstance(forecast, Forecast) and forecast.target_names == ("a", "b")
    np.testing.assert_array_equal(forecast.point, [[1, 4], [2, 5], [3, 4], [1, 5], [2, 4]])
    pipeline_forecast = Pipeline(parroting.ContextParroting(4)).predict(frame, range(30, 35))
    np.testing.assert_array_equal(pipeline_forecast.point, forecast.point)


# Model wrappers


def test_registered_models_match_the_protocol(protocol):
    assert models.registered_ids(protocol) == models.MODEL_IDS
    assert models.model_index("mlp_lags") == 6 and models.model_index("chronos_small") == 7
    assert isinstance(models.build("last_value", protocol), NaiveForecaster)
    seasonal = models.build("seasonal_naive", protocol)
    assert (seasonal.strategy, seasonal.period) == ("seasonal", 12)
    assert isinstance(models.build("context_parroting", protocol), parroting.ContextParroting)
    assert models.build("theta", protocol).name == "AutoTheta"


def _chronos_entry(protocol):
    return next(row for row in protocol["part_6_zoo"]["models"] if row["id"] == "chronos_small")


def test_chronos_is_refused_while_the_protocol_holds_no_revision(protocol):
    assert "revision" in _chronos_entry(protocol)
    for missing in (None, "", "main", "0123456789abcdef"):
        unpinned = copy.deepcopy(protocol)
        _chronos_entry(unpinned)["revision"] = missing
        assert models.chronos_revision(unpinned) is None
        with pytest.raises(RuntimeError, match="no Chronos revision"):
            models.build("chronos_small", unpinned)
    with pytest.raises(ValueError):
        models.ChronosModel(None)
    pinned = copy.deepcopy(protocol)
    entry = _chronos_entry(pinned)
    entry["revision"] = "0123456789abcdef0123456789abcdef01234567"
    model = models.build("chronos_small", pinned)
    assert model.revision == entry["revision"] and model.num_samples == 100
    assert model.clone_unfitted().revision == entry["revision"]


@pytest.mark.parametrize("name", ["AutoETS", "AutoARIMA", "AutoTheta"])
def test_statsforecast_wrappers_return_a_forecast_of_the_right_shape(name):
    pytest.importorskip("statsforecast")
    model = models.StatsForecastModel(name)
    clone = model.clone_unfitted()
    assert clone is not model and (clone.name, clone.season_length) == (name, 12)
    future = pd.RangeIndex(120, 144)
    forecast = clone.fit(_frame()).predict(future)
    assert isinstance(forecast, Forecast) and forecast.index.equals(future)
    assert forecast.point.shape == (24, 1) and forecast.point.dtype == float and forecast.target_names == ("y",)
    assert forecast.validate() == []


def test_wrappers_refuse_more_than_one_target():
    with pytest.raises(ValueError, match="one target"):
        models.StatsForecastModel("AutoETS").fit(pd.DataFrame({"a": [1.0] * 30, "b": [2.0] * 30}))


def test_subject_learner_wrappers(protocol):
    learners = _learners()
    if learners is None:
        with pytest.raises(ImportError):
            models.build("hgb_lags", protocol)
        return
    stream = models.training_stream(protocol, "mlp_lags")
    expected = np.random.default_rng(np.random.SeedSequence([14_500_000, 6])).integers(0, 2**31 - 1, size=3)
    assert [stream.draw() for _ in range(3)] == expected.tolist()
    mlp = models.build("mlp_lags", protocol)
    assert mlp.clone_unfitted().stream is mlp.stream and mlp.learner == "mlp_window_standardized"
    future = pd.RangeIndex(120, 144)
    for model_id in ("hgb_lags", "mlp_lags"):
        forecast = models.build(model_id, protocol).clone_unfitted().fit(_frame()).predict(future)
        assert isinstance(forecast, Forecast) and np.asarray(forecast.point).shape == (24, 1)
        assert forecast.index.equals(future)


# Seeds


class _Abort(BaseException):
    """Stops the scorecard after its run and before any cell is scored."""


class _Zeros:
    def clone_unfitted(self):
        return _Zeros()

    def fit(self, context):
        return self

    def predict(self, future_index):
        return np.zeros(len(future_index))


def test_the_registered_scorecard_uses_exactly_the_zoo_seed_blocks(protocol, monkeypatch):
    seen = {"series": [], "dependence": []}
    generate, run = timegym.Scenario.generate, timegym.pipeline.Pipeline.run

    def record_generate(self, seed=None):
        seen["series"].append(seed)
        return generate(self, seed=seed)

    def record_evaluate(case, forecast, **settings):
        seen["dependence"].append(settings["dependence_seed"])
        raise RuntimeError("not evaluated")

    def stop_after_run(self, *args, **kwargs):
        run(self, *args, **kwargs)
        raise _Abort

    monkeypatch.setattr(timegym.Scenario, "generate", record_generate)
    monkeypatch.setattr(timegym.pipeline, "evaluate_forecast", record_evaluate)
    monkeypatch.setattr(timegym.pipeline.Pipeline, "run", stop_after_run)
    with pytest.raises(_Abort):
        run_zoo.scorecard(_Zeros())
    series = common.case_seeds(protocol, "zoo_series")
    dependence = common.case_seeds(protocol, "zoo_dependence")
    assert series == list(range(14_000_000, 14_000_030)) and dependence == list(range(14_100_000, 14_100_030))
    assert seen["series"] == series * 20 and seen["dependence"] == dependence * 20
    assert run_zoo.check_scorecard_seeds(protocol) == (series, dependence)


# Runner and figure


class _Card:
    """A fabricated scorecard: made-up numbers in the shape that save_baseline and the table have."""

    def __init__(self, title, offset):
        rows = []
        for index, pattern in enumerate(timegym.PATTERNS):
            for sd in (0.5, 2.0):
                score = 10.0 + offset + 3 * index + sd
                rows.append({"pattern": pattern, "noise_sd": sd, "score": score, "score_low": score - 4,
                             "score_high": score + 4, "attainable": 90.0, "instances": 30, "failed_checks": 0})
        self.title, self.table = title, pd.DataFrame(rows)

    def save_baseline(self, path):
        record = {"title": self.title, "seed": run_zoo.SEED, "instances": run_zoo.INSTANCES,
                  "grid": {"patterns": list(timegym.PATTERNS), "noise_sds": [0.5, 2.0], "history_length": 120,
                           "horizon": 24},
                  "cells": [{"pattern": row["pattern"], "noise_sd": row["noise_sd"], "score": row["score"],
                             "avoidable": [None] * 30} for row in self.table.to_dict(orient="records")]}
        path.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
        return path


def test_runner_writes_each_model_and_the_figure_reads_only_saved_files(protocol, tmp_path):
    made = []

    def make(model_id, given):
        made.append(model_id)
        return object()

    cards = iter([_Card("first", 0.0), _Card("second", 20.0)])
    ids = ("last_value", "context_parroting")
    manifests = run_zoo.run_models(ids, tmp_path, protocol, make=make, score=lambda model: next(cards))
    assert made == list(ids)
    for model_id in ids:
        manifest = json.loads((tmp_path / f"{model_id}.manifest.json").read_text())
        assert manifest == manifests[model_id]
        assert manifest["scorecard"] == {"seed": 14_000_000, "instances": 30, "other_arguments": "defaults"}
        assert manifest["series_seeds"] == [14_000_000, 14_000_029]
        assert manifest["dependence_seeds"] == [14_100_000, 14_100_029]
        assert manifest["protocol_sha256"] == common.sha256(common.PROTOCOL)
        assert (tmp_path / f"{model_id}.table.csv").exists() and (tmp_path / f"{model_id}.runtime.json").exists()
    with pytest.raises(FileExistsError):
        run_zoo.run_models(("last_value",), tmp_path, protocol, make=make, score=lambda model: _Card("x", 0.0))
    loaded = figure.load(tmp_path)
    assert list(loaded) == list(ids) and loaded["context_parroting"]["cells"]["noise", 0.5] == (30.5, 26.5, 34.5)
    path = figure.draw(tmp_path, tmp_path / "zoo.png")
    assert path.stat().st_size > 0
    table = tmp_path / "last_value.table.csv"
    table.write_text(table.read_text().replace("10.5,6.5", "11.5,6.5", 1))
    with pytest.raises(ValueError, match="differs from the baseline"):
        figure.load(tmp_path)


def test_a_draft_protocol_stops_the_zoo_and_the_plan_computes_nothing(protocol, monkeypatch, capsys):
    draft = copy.deepcopy(protocol)
    draft.update(status="draft")
    draft["registration"].update(merge_commit="", registered_on="")
    _chronos_entry(draft)["revision"] = None
    monkeypatch.setattr(common, "load_protocol", lambda: draft)
    monkeypatch.setattr(sys, "argv", ["run_zoo"])
    with pytest.raises(RuntimeError, match="may not start"):
        run_zoo.main()
    monkeypatch.setattr(sys, "argv", ["run_zoo", "--plan", "--models", "ets", "chronos_small"])
    run_zoo.main()
    shown = json.loads(capsys.readouterr().out)
    assert shown["models"] == ["ets", "chronos_small"] and shown["chronos_revision"] is None
    assert shown["fits_per_model"] == 600


def test_lock_files():
    zoo = (ZOO / "requirements-zoo.txt").read_text().splitlines()
    assert zoo[0].startswith("# ") and "timegym" in zoo[0]
    pins = [line for line in zoo[1:] if line]
    assert all("==" in line and " @ " not in line and not line.startswith("-e") for line in pins)
    for pin in ("statsforecast==2.1.1", "coreforecast==0.0.20", "statsmodels==0.15.0", "pandas==2.3.3",
                "numpy==2.5.4", "scipy==1.18.1", "scikit-learn==1.9.1", "matplotlib==3.11.2"):
        assert pin in pins
    chronos = (ZOO / "requirements-chronos.txt").read_text().splitlines()
    assert chronos[0].startswith("# ") and "CPU-only torch" in chronos[0] and "manifest" in chronos[0]
    assert [line for line in chronos[1:] if line] == ["chronos-forecasting==2.3.2"]
