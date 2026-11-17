"""Toy learners, a toy second implementation and toy conditions for the Study 3 harness tests.

The learner module has the interface of experiments/study3_learners.py (SeedStream and build) for the toy subject
tests/fixtures/study3_toy_subject.py. Learners: "mean" adds nothing (exactly shift equivariant), "noisy" adds
seeded Gaussian noise of sd 0.2 drawn at every fit (equivariant in distribution) and "level_noise" adds noise whose
sd is 0.2 times the absolute level of the series (not equivariant). All seeds used with them are below 1,000,000.
"""
import numpy as np

from timegym import Forecast

LEARNERS = ("mean", "noisy", "level_noise")
SUBJECT = __file__.replace("study3_toys.py", "study3_toy_subject.py")
CONDITIONS = (
    {"id": "toy_white_noise", "kind": "white_noise", "context_length": 30, "horizon": 4, "n_targets": 2,
     "parameters": {"mean": [1.0, -2.0], "sigma": [1.0, 0.5], "correlation": 0.3}},
    {"id": "toy_ar1", "kind": "ar1", "context_length": 24, "horizon": 3, "n_targets": 2,
     "parameters": {"mean": [0.5, 2.0], "phi": 0.5, "sigma": [0.6, 0.9], "correlation": 0.2}},
)


class SeedStream:
    def __init__(self, entropy):
        self.entropy = entropy
        self.generator = np.random.default_rng(np.random.SeedSequence(entropy))

    def draw(self):
        return int(self.generator.integers(0, 2**31 - 1))


def build(learner, stream=None, n_targets=2, subject=None):
    if learner == "mean":
        return subject.ToyForecaster(lambda level: np.zeros_like(level))
    if learner == "noisy":
        return subject.ToyForecaster(lambda level: np.random.default_rng(stream.draw()).normal(0.0, 0.2, level.shape))
    if learner == "level_noise":
        return subject.ToyForecaster(
            lambda level: np.random.default_rng(stream.draw()).normal(0.0, 1.0, level.shape) * 0.2 * np.abs(level))
    raise ValueError(learner)


def reference_forecast(context, future_index):
    """The context mean, computed another way."""
    values = context.to_numpy(dtype=float)
    level = values.sum(axis=0) / len(values)
    return Forecast(index=future_index.copy(), point=np.repeat(level[None, :], len(future_index), axis=0),
                    target_names=tuple(context.columns))
