"""The second implementation of Study 3, Part 1 (protocol 3.0): the clean Ridge subject written independently.

reference_forecast receives only the context and the future index. It builds one block of columns per lag (lags 1
to window of every series) and a block for the seasonal lag, solves the augmented normal equations of a ridge
regression with an unpenalized intercept in the original units of the features and the targets, and forecasts
recursively. It imports nothing from the subject, scikit-learn or a TimeGym forecaster.

Why it equals the subject with scaling "global" and Ridge(alpha): the subject standardizes the targets, column j
by its mean m_j and scale s_j, and fits Ridge on standardized features Z = (X - c) / d (column means c and
population standard deviations d, a zero d replaced by 1). For one target column, multiplying Ridge's objective
||(y - m) / s - b - Z w||^2 + alpha ||w||^2 by s^2 gives ||y - (m + s b) - Z (s w)||^2 + alpha ||s w||^2, the
same ridge problem in the original target units, so target standardization changes nothing. Writing Z g =
X (g / d) - c . (g / d) moves the features to their original units with coefficients beta_k = g_k / d_k, an
intercept that absorbs the centering, and the penalty alpha sum_k d_k^2 beta_k^2. The ridge problem on raw lag
columns with an unpenalized intercept and the diagonal penalty alpha d_k^2 therefore has the subject's
solution. A constant column (d_k = 0, replaced by 1) gets coefficient 0 in both. The penalized normal matrix is
positive definite, so the solution is unique.
"""
from __future__ import annotations

import numpy as np

from timegym import Forecast


def lag_columns(values, window, period):
    """Feature matrix with one block per lag 1..window and a final seasonal block, for rows start..n-1."""
    n = len(values)
    start = max(window, period)
    blocks = [values[start - lag:n - lag] for lag in range(1, window + 1)]
    blocks.append(values[start - period:n - period])
    return np.concatenate(blocks, axis=1), values[start:]


def reference_forecast(context, future_index, *, window=6, period=12, alpha=1.0):
    values = context.to_numpy(dtype=float)
    features, targets = lag_columns(values, window, period)
    spread = features.std(axis=0)
    spread = np.where(spread == 0, 1.0, spread)
    design = np.column_stack([np.ones(len(features)), features])
    penalty = np.diag(np.r_[0.0, alpha * spread**2])
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ targets)
    history = values.copy()
    points = []
    for _ in range(len(future_index)):
        n = len(history)
        row = np.concatenate([history[n - lag] for lag in range(1, window + 1)] + [history[n - period]])
        prediction = np.r_[1.0, row] @ coefficients
        points.append(prediction)
        history = np.concatenate([history, prediction[None, :]], axis=0)
    return Forecast(index=future_index.copy(), point=np.asarray(points), target_names=tuple(context.columns))
