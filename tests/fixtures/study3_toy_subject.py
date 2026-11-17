"""A toy subject for the Study 3 harness tests: forecast every series by its context mean plus a learner's jitter.

It has the subject's interface (a forecaster built by a learner module's build) and nothing of its computation, so
that the harness, the screens, the checks and the statistics are tested without running the study's subject.
"""
import numpy as np
from timegym import Forecast


class ToyForecaster:
    def __init__(self, jitter):
        self.jitter = jitter

    def clone_unfitted(self):
        return ToyForecaster(self.jitter)

    def fit(self, context):
        values = context.to_numpy(dtype=float)
        self.level = values.mean(axis=0)
        self.offset = self.jitter(self.level)
        self.names = tuple(context.columns)
        return self

    def predict(self, future_index):
        point = np.tile(self.level + self.offset, (len(future_index), 1))
        return Forecast(index=future_index.copy(), point=point, target_names=self.names)
