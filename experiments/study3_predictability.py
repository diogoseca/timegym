"""Two model-free predictability estimators of protocol 3.0, Part 5 (part_5_predictability_estimators).

Pure functions of arrays. ``spectral_bound`` is the lower bound on the mean squared error of Feng,
Yuan, Ding and Li (arXiv 2509.23074v3) computed exactly as ``spectral_bound.steps`` states it: the
Welch averages are written out with ``numpy.fft.rfft`` instead of calling ``scipy.signal.welch`` or
``scipy.signal.coherence``, whose defaults (detrending of each window, density scaling, one-sided
doubling) differ from those steps. ``weighted_permutation_entropy`` is the normalized weighted
permutation entropy of Garland, James and Bradley (arXiv 1404.6823v2) as ``permutation_entropy``
states it.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.signal.windows import hann

EPSILON = 1e-8
WORD_LENGTH = 3


def _segment(values, name):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError(f"{name} must be a finite one-dimensional array")
    return values


def welch_windows(length):
    """Start positions of the Welch windows: length floor(0.25 N), 50% overlap, starting at 0."""
    size = math.floor(0.25 * length)
    if size < 2:
        raise ValueError("Segments need at least 8 values for a Welch window of 2")
    step = size // 2
    return size, list(range(0, length - size + 1, step))


def spectra(x, y):
    """Frequencies f = k / size and the Welch averages S_xx, S_yy and S_xy of the demeaned segments x and y."""
    size, starts = welch_windows(len(x))
    weights = hann(size, sym=False)
    scale = float(np.sum(weights * weights))
    xs = np.fft.rfft(np.stack([x[start:start + size] for start in starts]) * weights, axis=1)
    ys = np.fft.rfft(np.stack([y[start:start + size] for start in starts]) * weights, axis=1)
    s_xx = np.mean(np.abs(xs) ** 2, axis=0) / scale
    s_yy = np.mean(np.abs(ys) ** 2, axis=0) / scale
    s_xy = np.mean(xs * np.conj(ys), axis=0) / scale
    return np.fft.rfftfreq(size), s_xx, s_yy, s_xy


def spectral_bound(x, y, epsilon=EPSILON):
    """MSE_lb of the history segment x and the realized future segment y (equal lengths N).

    Returns ``bound`` (NaN when undefined), ``defined`` (False when the sample variance V of y, with
    denominator N, or the sum of S_yy over frequencies is zero; the ratio to the exact floor is then
    undefined), and the pieces ``delta_squared``, ``variance`` and ``weighted_coherence``.
    """
    x, y = _segment(x, "x"), _segment(y, "y")
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    delta_squared = float((y.mean() - x.mean()) ** 2)
    x, y = x - x.mean(), y - y.mean()
    _, s_xx, s_yy, s_xy = spectra(x, y)
    coherence = np.clip(np.abs(s_xy) ** 2 / ((s_xx + epsilon) * (s_yy + epsilon)), 0.0, 1.0)
    variance = float(np.mean(y * y))
    power = float(np.sum(s_yy))
    if variance == 0 or power == 0:
        return {"bound": math.nan, "defined": False, "delta_squared": delta_squared, "variance": variance,
                "weighted_coherence": math.nan}
    weighted = float(np.sum(s_yy * coherence) / power)
    return {"bound": delta_squared + variance * (1 - weighted), "defined": True, "delta_squared": delta_squared,
            "variance": variance, "weighted_coherence": weighted}


def ordinal_patterns(values, word_length=WORD_LENGTH):
    """The ordinal pattern (a stable argsort) of every window of ``word_length`` consecutive values, delay 1."""
    values = _segment(values, "values")
    if len(values) < word_length:
        raise ValueError("The series is shorter than one word")
    windows = np.lib.stride_tricks.sliding_window_view(values, word_length)
    return windows, np.argsort(windows, axis=1, kind="stable")


def weighted_permutation_entropy(values, word_length=WORD_LENGTH):
    """Normalized weighted permutation entropy, or NaN when every window has weight zero (a constant series).

    Each window's weight is (1/l) sum (x_j - window mean)^2; P_w(pi) is the weight of the windows with
    pattern pi over the total weight; WPE = -sum P_w log2 P_w / log2(l!).
    """
    windows, patterns = ordinal_patterns(values, word_length)
    weights = np.mean((windows - windows.mean(axis=1, keepdims=True)) ** 2, axis=1)
    _, inverse = np.unique(patterns, axis=0, return_inverse=True)
    by_pattern = np.bincount(inverse.reshape(-1), weights=weights)
    # The total is the sum of the per-pattern weights, so that a single pattern has a mass of exactly 1.
    total = float(by_pattern.sum())
    if total == 0:
        return math.nan
    mass = by_pattern[by_pattern > 0] / total
    return float(-np.sum(mass * np.log2(mass)) / math.log2(math.factorial(word_length)))
