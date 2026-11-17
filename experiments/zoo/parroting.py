"""Context parroting (Zhang and Gilpin, arXiv 2505.11349v3, Algorithm 1) as part_6_zoo.models states it.

NumPy only, one target at a time, deterministic. For a context x_1..x_L and embedding dimension D,
the motif is the window of D consecutive values inside x_1..x_(L-D) nearest in Euclidean distance to
the last D values x_(L-D+1)..x_L; ties go to the earliest motif. The forecast copies the values that
followed the motif, x_(s+D)..x_L for a motif starting at s, and repeats that copy until it has as many
values as the horizon. Repeating the copy is the periodic continuation with period L - D - s + 1 that
the match implies.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from timegym import Forecast

EMBEDDING = 10


def nearest_motif(values, embedding=EMBEDDING):
    """0-based start of the earliest motif in values[:L - D] nearest to values[L - D:], and all the distances."""
    values = np.asarray(values, dtype=float)
    length = len(values)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("values must be a finite one-dimensional array")
    if embedding < 1 or length < 2 * embedding:
        raise ValueError(f"Context parroting with D = {embedding} needs at least {2 * embedding} values")
    query = values[length - embedding:]
    motifs = np.lib.stride_tricks.sliding_window_view(values[:length - embedding], embedding)
    distances = np.sqrt(np.sum((motifs - query) ** 2, axis=1))
    return int(np.argmin(distances)), distances


def parrot(values, horizon, embedding=EMBEDDING):
    """The ``horizon`` values that context parroting forecasts for one series."""
    values = np.asarray(values, dtype=float)
    start, _ = nearest_motif(values, embedding)
    copied = values[start + embedding:]
    return np.resize(copied, int(horizon))


class ContextParroting:
    """Context parroting as a TimeGym model: fit keeps the context, predict parrots every target."""

    def __init__(self, embedding=EMBEDDING):
        self.embedding = int(embedding)

    def clone_unfitted(self):
        return ContextParroting(self.embedding)

    def fit(self, context):
        if not isinstance(context, pd.DataFrame):
            raise TypeError("context must be a pandas DataFrame")
        self.context_ = context.copy(deep=True)
        return self

    def predict(self, future_index):
        values = self.context_.to_numpy(dtype=float)
        point = np.column_stack([parrot(values[:, column], len(future_index), self.embedding)
                                 for column in range(values.shape[1])])
        return Forecast(index=pd.Index(future_index).copy(), point=point,
                        target_names=tuple(self.context_.columns))
