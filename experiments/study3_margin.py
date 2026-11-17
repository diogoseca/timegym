"""The joint-spread decision with a margin of protocol 3.0, Part 4 (part_4_joint_spread_margin).

Pure functions of arrays. ``estimate`` computes the unbiased estimator ``T`` of the squared relative
Frobenius distance ``D^2 = ||Sigma - Sigma0||_F^2 / ||Sigma0||_F^2`` between the covariance of the
paths and a reference covariance, and the leave-one-out (jackknife) standard error of ``T``.
``decide`` turns an estimate and a standard error into the interval and the verdict of
``decision``. Nothing here draws a random number.

With ``A`` the Gram matrix ``X X'`` of the paths with its diagonal set to zero, ``r`` its row sums,
``S1 = sum A_ij^2``, ``S2 = sum r_j^2 - S1`` (ordered triples of distinct paths) and
``S3 = (sum A)^2 - 4 S2 - 2 S1`` (ordered quadruples of distinct paths), the three averages of
``statistic.estimator_of_D_squared`` are ``Y1 = S1 / (n)_2``, ``Y2 = S2 / (n)_3`` and
``Y3 = S3 / (n)_4``, with ``(n)_k`` the falling factorial. ``tr(S Sigma0)`` comes from the weighted
Gram matrix ``H = X Sigma0 X'`` as ``(tr H - sum H / n) / (n - 1)``. ``T`` does not change when
every path is translated by the same vector, so the paths are centred on their column means first,
which only reduces rounding. Leaving path ``i`` out removes row and column ``i`` from ``A`` and
``H``; the sums above are downdated in closed form, so all ``n`` leave-one-out values cost
``O(n^2)`` after the two Gram matrices.
"""
from __future__ import annotations

import math

import numpy as np

Z = 1.6448536
VERDICTS = ("FAIL", "WITHIN_TOLERANCE", "INCONCLUSIVE")


def _paths(samples, reference_covariance):
    paths = np.asarray(samples, dtype=float)
    paths = paths.reshape(len(paths), -1)
    sigma0 = np.asarray(reference_covariance, dtype=float)
    n, p = paths.shape
    if n < 5:
        raise ValueError("The leave-one-out estimator needs at least 5 paths")
    if sigma0.shape != (p, p) or not np.isfinite(sigma0).all() or not np.isfinite(paths).all():
        raise ValueError("samples must be finite with shape (n, ...) and the covariance (p, p) for p coordinates")
    scale = float(np.sum(sigma0 * sigma0))
    if scale == 0:
        raise ValueError("The relative distance is undefined for a zero reference covariance")
    return paths - paths.mean(axis=0), sigma0, scale


def _falling(n, k):
    return float(math.prod(range(n - k + 1, n + 1)))


def _combine(s1, s2, s3, trace_h, total_h, n, trace_sigma0_squared, scale):
    """T from the sums of the Gram matrices; works elementwise on arrays of leave-one-out sums."""
    trace_sigma_squared = s1 / _falling(n, 2) - 2 * s2 / _falling(n, 3) + s3 / _falling(n, 4)
    trace_s_sigma0 = (trace_h - total_h / n) / (n - 1)
    return (trace_sigma_squared - 2 * trace_s_sigma0 + trace_sigma0_squared) / scale


def estimator(samples, reference_covariance):
    """T, the unbiased estimator of the squared relative Frobenius distance (no standard error)."""
    return estimate(samples, reference_covariance, jackknife=False)["T"]


def estimate(samples, reference_covariance, *, jackknife=True):
    """T and, with ``jackknife``, the leave-one-out values T_(i) and the standard error of T.

    ``samples`` has shape ``(n, ...)``; each path is flattened in C order, as
    ``covariance_dependence_check`` flattens horizon and targets.
    """
    x, sigma0, scale = _paths(samples, reference_covariance)
    n = len(x)
    gram = x @ x.T
    weighted = x @ sigma0 @ x.T
    off = gram.copy()
    np.fill_diagonal(off, 0.0)
    rows = off.sum(axis=1)
    s1 = float(np.sum(off * off))
    s2 = float(rows @ rows) - s1
    total = float(rows.sum())
    s3 = total * total - 4 * s2 - 2 * s1
    trace_h, total_h = float(np.trace(weighted)), float(weighted.sum())
    trace_sigma0_squared = scale
    value = float(_combine(s1, s2, s3, trace_h, total_h, n, trace_sigma0_squared, scale))
    out = {"T": value, "n_paths": n, "n_coordinates": int(x.shape[1])}
    if not jackknife:
        return out
    # Leave path i out: A loses row and column i.
    squares = np.sum(off * off, axis=1)
    s1_out = s1 - 2 * squares
    # Row sums without path i: r_j - A_ji for j != i; sum over j != i of (r_j - A_ij)^2.
    reduced = rows[None, :] - off
    sum_reduced_squares = np.sum(reduced * reduced, axis=1) - rows * rows
    s2_out = sum_reduced_squares - s1_out
    total_out = total - 2 * rows
    s3_out = total_out * total_out - 4 * s2_out - 2 * s1_out
    weighted_rows = weighted.sum(axis=1)
    diagonal = np.diag(weighted)
    trace_out = trace_h - diagonal
    total_h_out = total_h - 2 * weighted_rows + diagonal
    leave_one_out = _combine(s1_out, s2_out, s3_out, trace_out, total_h_out, n - 1, trace_sigma0_squared, scale)
    centre = float(leave_one_out.mean())
    variance = (n - 1) / n * float(np.sum((leave_one_out - centre) ** 2))
    out.update(leave_one_out=leave_one_out, standard_error=math.sqrt(variance))
    return out


def decide(t, standard_error, margin, z=Z):
    """The interval for D^2 and for D, D_hat and the verdict of ``decision``, from T and its standard error."""
    t, standard_error, margin = float(t), float(standard_error), float(margin)
    if not (math.isfinite(t) and math.isfinite(standard_error) and standard_error >= 0):
        raise ValueError("T must be finite and its standard error finite and nonnegative")
    if not margin > 0:
        raise ValueError("The margin must be positive")
    lower, upper = t - z * standard_error, t + z * standard_error
    squared = margin * margin
    if lower > squared:
        verdict = "FAIL"
    elif upper < squared:
        verdict = "WITHIN_TOLERANCE"
    else:
        verdict = "INCONCLUSIVE"
    return {"T": t, "standard_error": standard_error, "margin": margin, "z": z,
            "interval_D_squared": [lower, upper],
            "interval_D": [math.sqrt(max(lower, 0.0)), math.sqrt(max(upper, 0.0))],
            "D_hat": math.sqrt(max(t, 0.0)), "verdict": verdict}


def margin_check(samples, reference_covariance, margin, z=Z):
    """``estimate`` followed by ``decide``: the full decision for one ensemble."""
    found = estimate(samples, reference_covariance)
    return decide(found["T"], found["standard_error"], margin, z)
