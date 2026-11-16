"""The learners of Study 3 (protocol 3.0, part_2_learners.learners) and their training seed streams.

Every learner is a regressor for experiments/study3_subject.py: fit(features, targets) and predict(features) take
2-D arrays and predict returns a 2-D array (rows, targets). A feature row is the flattened lag block (lags 1 to
window, time-major, series-minor) followed by one seasonal lag per series. The two perceptrons draw their
random_state from a SeedStream at every fit; the stream is shared by every clone of one forecaster, so successive
fits on a case draw successive seeds. The file is frozen with the protocol.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor

from . import study3_subject

LEARNERS = ("ridge", "hgb", "mlp_window_standardized", "mlp_raw")  # position = learner_index
SCALING = {"ridge": "global", "hgb": "none", "mlp_window_standardized": "none", "mlp_raw": "none"}
WINDOW, PERIOD = 6, 12
RIDGE_ALPHA = 1.0
HIDDEN_LAYERS, MAX_ITER = (32,), 500


class SeedStream:
    """Training seeds of one stream; shared by clones so that successive fits draw successive seeds."""

    def __init__(self, entropy):
        self.entropy = entropy
        self.generator = np.random.default_rng(np.random.SeedSequence(entropy))

    def draw(self) -> int:
        return int(self.generator.integers(0, 2**31 - 1))


def _perceptron(stream):
    return MLPRegressor(hidden_layer_sizes=HIDDEN_LAYERS, max_iter=MAX_ITER, random_state=stream.draw())


def _labels(targets):
    """Targets as MLPRegressor takes them: a vector for one target, a matrix for several."""
    targets = np.asarray(targets, dtype=float)
    return targets[:, 0] if targets.ndim == 2 and targets.shape[1] == 1 else targets


class WindowStandardizedMLP:
    """MLPRegressor on rows standardized by each series' own window lags.

    For every row and series, the mean and population standard deviation of that series' window lags in the row
    (a zero standard deviation replaced by 1) standardize its window lags, its seasonal lag and its target; the
    outputs are mapped back with the same row statistics. The window is read from the row length:
    (window + 1) x n_targets columns. random_state is drawn from the stream at every fit.
    """

    def __init__(self, stream, n_targets):
        self.stream, self.n_targets = stream, n_targets

    def _standardize(self, features):
        features = np.asarray(features, dtype=float)
        rows, columns = features.shape
        window = columns // self.n_targets - 1
        if window < 1 or (window + 1) * self.n_targets != columns:
            raise ValueError(f"A row of {columns} features does not hold window and seasonal lags of "
                             f"{self.n_targets} series")
        lags = features[:, :window * self.n_targets].reshape(rows, window, self.n_targets)
        seasonal = features[:, window * self.n_targets:]
        center, scale = lags.mean(axis=1), lags.std(axis=1)
        scale = np.where(scale == 0, 1.0, scale)
        standardized = (lags - center[:, None, :]) / scale[:, None, :]
        return center, scale, np.concatenate([standardized.reshape(rows, -1), (seasonal - center) / scale], axis=1)

    def fit(self, features, targets):
        center, scale, inputs = self._standardize(features)
        labels = _labels((np.asarray(targets, dtype=float) - center) / scale)
        self.network = _perceptron(self.stream).fit(inputs, labels)
        return self

    def predict(self, features):
        center, scale, inputs = self._standardize(features)
        return np.asarray(self.network.predict(inputs)).reshape(len(inputs), -1) * scale + center


class RawMLP:
    """MLPRegressor on the raw lag features and raw targets; random_state is drawn from the stream at every fit."""

    def __init__(self, stream):
        self.stream = stream

    def fit(self, features, targets):
        self.network = _perceptron(self.stream).fit(features, _labels(targets))
        return self

    def predict(self, features):
        return np.asarray(self.network.predict(features)).reshape(len(features), -1)


def make_regressor(learner, stream=None, n_targets=2):
    """A zero-argument factory of unfitted regressors of ``learner``; the perceptrons need a stream."""
    if learner not in LEARNERS:
        raise ValueError(f"learner must be one of {LEARNERS}")
    if learner == "ridge":
        return lambda: Ridge(alpha=RIDGE_ALPHA)
    if learner == "hgb":
        return lambda: MultiOutputRegressor(HistGradientBoostingRegressor())
    if stream is None:
        raise ValueError(f"{learner} has a free training seed and needs a SeedStream")
    if learner == "mlp_window_standardized":
        return lambda: WindowStandardizedMLP(stream, n_targets)
    return lambda: RawMLP(stream)


def build(learner, stream=None, n_targets=2, subject=None):
    """The forecaster of ``learner`` on ``subject`` (a module: the clean subject by default, or a loaded mutant)."""
    subject = study3_subject if subject is None else subject
    return subject.SubjectForecaster(make_regressor(learner, stream, n_targets), window=WINDOW, period=PERIOD,
                                     scaling=SCALING[learner])
