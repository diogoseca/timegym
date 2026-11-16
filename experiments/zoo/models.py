"""The nine models of protocol 3.0, Part 6 (part_6_zoo.models), as objects for timegym.Pipeline(model).

Every model has ``clone_unfitted()``, ``fit(context)`` and ``predict(future_index)`` returning a
``timegym.Forecast``; ``build(model_id, protocol)`` makes the registered configuration of each.
statsforecast is imported inside StatsForecastModel and torch and chronos inside ChronosModel, so the
other models need neither.

Seeds (seeds.namespaces.zoo_scorecard): a model with a free training seed draws one integer per fit,
in the scorecard's order of fits, from numpy.random.default_rng(numpy.random.SeedSequence([14500000,
model_index])), model_index being the position in part_6_zoo.models. The stream is the SeedStream of
experiments/study3_learners.py and is shared by every clone of the model, so each fit takes the next
integer. Chronos samples after one torch.manual_seed(14600000), which run_zoo.py calls before its
scorecard.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from timegym import Forecast
from timegym.models import NaiveForecaster

from .parroting import ContextParroting

MODEL_IDS = ("last_value", "seasonal_naive", "ets", "auto_arima", "theta", "hgb_lags", "mlp_lags", "chronos_small",
             "context_parroting")
SEASON_LENGTH = 12
STATSFORECAST = {"ets": "AutoETS", "auto_arima": "AutoARIMA", "theta": "AutoTheta"}
LEARNERS = {"hgb_lags": "hgb", "mlp_lags": "mlp_window_standardized"}
CHRONOS_ID = "amazon/chronos-t5-small"
CHRONOS_SAMPLES = 100


def _one_target(context):
    if not isinstance(context, pd.DataFrame) or context.shape[1] != 1:
        raise ValueError("The zoo models forecast one target; the default grid has one")
    return context.copy(deep=True)


def _forecast(future_index, context, point=None, samples=None):
    return Forecast(index=pd.Index(future_index).copy(), point=point, samples=samples,
                    target_names=tuple(context.columns))


class StatsForecastModel:
    """statsforecast 2.1.1 AutoETS, AutoARIMA or AutoTheta, season_length 12; point = "mean" of forecast(y, h)."""

    def __init__(self, name, season_length=SEASON_LENGTH):
        if name not in STATSFORECAST.values():
            raise ValueError(f"name must be one of {tuple(STATSFORECAST.values())}")
        self.name, self.season_length = name, int(season_length)

    def clone_unfitted(self):
        return StatsForecastModel(self.name, self.season_length)

    def fit(self, context):
        self.context_ = _one_target(context)
        return self

    def predict(self, future_index):
        from statsforecast import models

        model = getattr(models, self.name)(season_length=self.season_length)
        y = self.context_.to_numpy(dtype=float)[:, 0]
        found = model.forecast(y=y, h=len(future_index))
        point = np.asarray(found["mean"], dtype=float).reshape(len(future_index), 1)
        return _forecast(future_index, self.context_, point=point)


class SubjectLearnerModel:
    """The clean subject of Part 1 with a Part 2 learner, from experiments.study3_learners.build.

    ``stream`` is the learner's SeedStream for a learner with a free training seed (None otherwise); clones
    share it. Each fit builds a fresh subject forecaster with ``n_targets=1`` and fits it.
    """

    def __init__(self, learner, stream=None):
        from experiments import study3_learners

        if learner not in study3_learners.LEARNERS:
            raise ValueError(f"Unknown learner {learner!r}")
        self.learner, self.stream = learner, stream

    def clone_unfitted(self):
        return SubjectLearnerModel(self.learner, self.stream)

    def fit(self, context):
        from experiments import study3_learners

        self.context_ = _one_target(context)
        self.model_ = study3_learners.build(self.learner, stream=self.stream, n_targets=1)
        self.model_.fit(self.context_.copy(deep=True))
        return self

    def predict(self, future_index):
        found = self.model_.predict(pd.Index(future_index).copy())
        if isinstance(found, Forecast):
            return found
        if isinstance(found, (pd.DataFrame, pd.Series)):
            found = found.to_numpy()
        point = np.asarray(found, dtype=float).reshape(len(future_index), 1)
        return _forecast(future_index, self.context_, point=point)


_CHRONOS = {}


def chronos_pipeline(revision, model_id=CHRONOS_ID):
    """ChronosPipeline.from_pretrained(model_id, revision, device_map='cpu', torch_dtype=float32), once a process."""
    if not revision:
        raise ValueError("Chronos needs the registered model revision")
    key = (model_id, revision)
    if key not in _CHRONOS:
        import torch
        from chronos import ChronosPipeline

        torch.set_num_threads(1)
        _CHRONOS[key] = ChronosPipeline.from_pretrained(model_id, revision=revision, device_map="cpu",
                                                        torch_dtype=torch.float32)
    return _CHRONOS[key]


class ChronosModel:
    """Chronos (small) with 100 sample paths per forecast; the forecast carries the samples and point = their mean."""

    def __init__(self, revision, model_id=CHRONOS_ID, num_samples=CHRONOS_SAMPLES):
        if not revision:
            raise ValueError("Chronos needs the registered model revision")
        self.revision, self.model_id, self.num_samples = revision, model_id, int(num_samples)

    def clone_unfitted(self):
        return ChronosModel(self.revision, self.model_id, self.num_samples)

    def fit(self, context):
        self.context_ = _one_target(context)
        return self

    def predict(self, future_index):
        import torch

        pipeline = chronos_pipeline(self.revision, self.model_id)
        inputs = torch.tensor(self.context_.to_numpy(dtype=float)[:, 0], dtype=torch.float32)
        draws = pipeline.predict(inputs, prediction_length=len(future_index), num_samples=self.num_samples)
        draws = np.asarray(draws.detach().cpu().numpy(), dtype=float).reshape(self.num_samples, len(future_index))
        samples = draws[:, :, None]
        return _forecast(future_index, self.context_, point=samples.mean(axis=0), samples=samples)


def model_index(model_id):
    """The position of ``model_id`` in part_6_zoo.models."""
    return MODEL_IDS.index(model_id)


def registered_ids(protocol):
    """The model ids of part_6_zoo.models, checked against MODEL_IDS."""
    ids = tuple(row["id"] for row in protocol["part_6_zoo"]["models"])
    if ids != MODEL_IDS:
        raise ValueError(f"part_6_zoo.models lists {ids}; this module implements {MODEL_IDS}")
    return ids


def chronos_revision(protocol):
    """The registered Chronos revision: a 40-character commit hash under 'revision' in its model entry, else None."""
    entry = next(row for row in protocol["part_6_zoo"]["models"] if row["id"] == "chronos_small")
    revision = entry.get("revision")
    if isinstance(revision, str) and len(revision) == 40 and all(c in "0123456789abcdef" for c in revision):
        return revision
    return None


def training_stream(protocol, model_id):
    """The shared SeedStream of a model with a free training seed: SeedSequence([14500000, model_index])."""
    from experiments import study3_common, study3_learners

    tag = study3_common.generator_tag(protocol, "zoo_training")
    return study3_learners.SeedStream([tag, model_index(model_id)])


def build(model_id, protocol):
    """The registered configuration of one Part 6 model."""
    registered_ids(protocol)
    if model_id == "last_value":
        return NaiveForecaster()
    if model_id == "seasonal_naive":
        return NaiveForecaster("seasonal", SEASON_LENGTH)
    if model_id in STATSFORECAST:
        return StatsForecastModel(STATSFORECAST[model_id])
    if model_id == "hgb_lags":
        return SubjectLearnerModel(LEARNERS[model_id])
    if model_id == "mlp_lags":
        return SubjectLearnerModel(LEARNERS[model_id], training_stream(protocol, model_id))
    if model_id == "chronos_small":
        revision = chronos_revision(protocol)
        if revision is None:
            raise RuntimeError("part_6_zoo.models holds no Chronos revision; the Chronos scorecard may not run")
        return ChronosModel(revision)
    if model_id == "context_parroting":
        return ContextParroting()
    raise ValueError(f"Unknown model {model_id!r}")
