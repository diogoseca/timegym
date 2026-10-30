"""Selected implementation defects and independent comparators for supplement v2.

These are experiment fixtures, not public TimeGym diagnostic APIs. The fitted
model uses sklearn's solver; the comparison implementation uses independently
constructed lag columns and an augmented normal-equation ridge solution. Target
standardisation commutes with this affine, fixed-penalty regression, so their
correct predictions agree without access to the generating process or future.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from timegym import Forecast


FAULTS = (
    "reverse_prediction_lags",
    "stale_recursive_state",
    "omit_inverse_target_scaling",
    "swap_recursive_targets",
)


class LagRidgeImplementation:
    """Learn a standardized lag regression with one named prediction defect.

    Every branch fits the same training-only transformation and regression.
    Faults occur inside prediction and, when applicable, affect subsequent
    recursive inputs. No function mutates a completed clean Forecast.
    """

    def __init__(self, fault="clean", window=6, alpha=1.0):
        if fault not in ("clean",) + FAULTS:
            raise ValueError("Unknown implementation branch")
        if not isinstance(window, int) or isinstance(window, bool) or window < 1:
            raise ValueError("window must be a positive integer")
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be finite and positive")
        self.fault, self.window, self.alpha = fault, window, float(alpha)

    def clone_unfitted(self):
        return type(self)(fault=self.fault, window=self.window, alpha=self.alpha)

    def fit(self, context):
        values = context.to_numpy(dtype=float)
        if len(values) <= self.window:
            raise ValueError("Insufficient context for lag regression")
        if self.fault == "swap_recursive_targets" and values.shape[1] < 2:
            raise ValueError("The target-binding defect requires multiple targets")
        self.names_ = tuple(context.columns)
        self.history_ = values.copy()
        features = np.asarray([
            values[i - self.window:i].ravel()
            for i in range(self.window, len(values))
        ])
        targets = values[self.window:]
        self.features_ = StandardScaler().fit(features)
        self.targets_ = StandardScaler().fit(targets)
        self.regressor_ = Ridge(alpha=self.alpha).fit(
            self.features_.transform(features), self.targets_.transform(targets)
        )
        return self

    def predict(self, future_index):
        history, points = list(self.history_.copy()), []
        for _ in future_index:
            lag_window = np.asarray(history[-self.window:])
            if self.fault == "reverse_prediction_lags":
                lag_window = lag_window[::-1]
            features = self.features_.transform(lag_window.reshape(1, -1))
            standardized = self.regressor_.predict(features)
            value = (standardized if self.fault == "omit_inverse_target_scaling"
                     else self.targets_.inverse_transform(standardized))[0]
            if self.fault == "swap_recursive_targets":
                value = value[::-1].copy()
            points.append(value)
            if self.fault != "stale_recursive_state":
                history.append(value)
        return Forecast(index=future_index.copy(), point=np.asarray(points),
                        target_names=self.names_)


def normal_equation_forecast(context, future_index, *, window=6, alpha=1.0):
    """Independent mathematical implementation of the declared ridge reducer.

    This does not call LagRidgeImplementation, sklearn, or a TimeGym forecaster.
    Targets remain in original units. The intercept is unpenalized, while the
    standardized lag coefficients receive the fixed L2 penalty. This reference
    specifies implementation agreement, not optimal forecasting performance.
    """
    values = context.to_numpy(dtype=float)
    rows = len(values) - window
    if rows < 1:
        raise ValueError("Insufficient context")
    features = np.concatenate([values[lag:lag + rows] for lag in range(window)], axis=1)
    center, scale = features.mean(axis=0), features.std(axis=0)
    scale = np.where(scale == 0, 1.0, scale)
    design = np.column_stack([np.ones(rows), (features - center) / scale])
    penalty = np.diag(np.r_[0.0, np.full(features.shape[1], float(alpha))])
    coefficients = np.linalg.solve(design.T @ design + penalty,
                                   design.T @ values[window:])
    state = values[-window:].copy()
    points = []
    for _ in future_index:
        input_row = np.r_[1.0, (state.reshape(-1) - center) / scale]
        prediction = input_row @ coefficients
        points.append(prediction)
        state = np.concatenate([state[1:], prediction[None, :]], axis=0)
    return Forecast(index=future_index.copy(), point=np.asarray(points),
                    target_names=tuple(context.columns))


def point_agreement(actual, expected, *, atol, rtol):
    """Experiment-only differential comparator, with declared numeric tolerance."""
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape or not np.isfinite(actual).all():
        return {"alarm": True, "max_absolute_error": None}
    return {
        "alarm": not bool(np.allclose(actual, expected, atol=atol, rtol=rtol)),
        "max_absolute_error": float(np.max(np.abs(actual - expected))),
    }


def target_shift_agreement(pipeline, context, future_index, original, *, shifts, atol, rtol):
    """Declared per-target translation law; supplementary, not a public API."""
    shift = np.asarray(shifts, dtype=float)
    if shift.shape != (context.shape[1],) or not np.isfinite(shift).all():
        raise ValueError("shifts must have one finite entry per target")
    translated = pipeline.predict(context + shift, future_index)
    return point_agreement(translated.point, original.point + shift, atol=atol, rtol=rtol)
