"""The subject pipeline of Study 3, Part 1 (protocol 3.0): a lag regression with a seasonal lag.

Every training row holds lags 1 to window of every series followed by the seasonal lag of every series; one
pluggable regressor forecasts all series at once, recursively. With scaling "global" the features and the targets
are standardized by their column means and population standard deviations. The module validates no argument and
holds no module-level constant, so that generated mutants act on the forecasting computation. Each catalogue site
ends with a "# site: <name>" comment. The file is frozen with the protocol and never edited afterwards.
"""
import numpy as np
from timegym import Forecast


def training_matrix(values, window, period):
    start = max(window, period)
    rows = []
    for t in range(start, len(values)):
        lags = values[t - window:t]  # site: T-lag
        seasonal = values[t - period]  # site: T-season
        rows.append(np.concatenate([lags.ravel(), seasonal]))
    targets = values[start:]  # site: T-target
    return np.asarray(rows), targets


def prediction_row(history, window, period):
    lags = history[-window:]  # site: P-lag
    seasonal = history[-period]  # site: P-season
    return np.concatenate([lags.ravel(), seasonal])[None, :]


def fit_scaler(array):
    center = array.mean(axis=0)
    scale = array.std(axis= 1)
    return center, np.where(scale == 0, 1.0, scale)


def apply_scaler(array, center, scale):
    return (array - center) / scale


def invert_scaler(array, center, scale):
    return array * scale + center


class SubjectForecaster:
    """Lag regression forecaster for timegym.Pipeline; make_regressor() returns an unfitted regressor."""

    def __init__(self, make_regressor, window=6, period=12, scaling="global"):
        self.make_regressor = make_regressor
        self.window = window
        self.period = period
        self.scaling = scaling

    def clone_unfitted(self):
        return SubjectForecaster(self.make_regressor, self.window, self.period, self.scaling)

    def fit(self, context):
        values = context.to_numpy(dtype=float)
        features, targets = training_matrix(values, self.window, self.period)
        if self.scaling == "global":
            self.feature_center, self.feature_scale = fit_scaler(features)  # site: S-x
            self.target_center, self.target_scale = fit_scaler(targets)  # site: S-y
            features = apply_scaler(features, self.feature_center, self.feature_scale)
            targets = apply_scaler(targets, self.target_center, self.target_scale)
        self.regressor = self.make_regressor().fit(features, targets)
        self.history = values
        self.names = tuple(context.columns)
        return self

    def predict(self, future_index):
        history = self.history
        points = []
        for step in range(len(future_index)):  # site: H
            row = prediction_row(history, self.window, self.period)
            if self.scaling == "global":
                row = apply_scaler(row, self.feature_center, self.feature_scale)
            output = self.regressor.predict(row)
            if self.scaling == "global":
                output = invert_scaler(output, self.target_center, self.target_scale)  # site: I
            value = output[0]
            points.append(value)
            history = np.vstack([history, value])  # site: R
        return Forecast(index=future_index.copy(), point=np.asarray(points), target_names=self.names)
