"""The calibrated shift check of Study 3, Part 2, on toy learners only (tests/fixtures/study3_toys.py).

Seeds are below 1,000,000; no learner of the study is used.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from timegym import Forecast, Pipeline

from experiments.model_faults import target_shift_agreement
from experiments.study3_shift import calibrated_shift_check, distance

sys.path.insert(0, str(Path(__file__).parent / "fixtures"))
import study3_toys as toys  # noqa: E402
import study3_toy_subject as toy_subject  # noqa: E402


def pipeline(learner, entropy):
    return Pipeline(toys.build(learner, toys.SeedStream(entropy), subject=toy_subject))


def context(seed, level=(1.0, -2.0), n=30):
    generator = np.random.default_rng(seed)
    return pd.DataFrame(np.asarray(level) + generator.normal(size=(n, 2)), columns=["y0", "y1"])


def test_draw_order_and_distances():
    data, future = context(1), pd.RangeIndex(30, 34)
    result = calibrated_shift_check(pipeline("noisy", [5, 1]), data, future)
    draws = toys.SeedStream([5, 1])
    seeds = [draws.draw() for _ in range(41)]

    def point(seed, values):
        offset = np.random.default_rng(seed).normal(0.0, 0.2, 2)
        return np.tile(values.to_numpy().mean(axis=0) + offset, (4, 1))

    reference = point(seeds[0], data)
    calibration = [distance(reference, point(seed, data)) for seed in seeds[1:40]]
    observed = distance(reference, point(seeds[40], data + 10.0) - 10.0)
    assert result["observed"] == pytest.approx(observed, rel=1e-9)
    assert result["calibration_max"] == pytest.approx(max(calibration), rel=1e-9)
    assert result["p_value"] == (1 + sum(value >= observed for value in calibration)) / 40
    assert result["status"] == ("FAIL" if result["p_value"] <= 0.05 else "PASS")


def test_a_given_forecast_is_draw_one():
    data, future = context(2), pd.RangeIndex(30, 33)
    first = pipeline("noisy", [6])
    forecast = first.predict(data, future)
    assert calibrated_shift_check(first, data, future, forecast=forecast) == calibrated_shift_check(
        pipeline("noisy", [6]), data, future)


def test_in_distribution_and_not_applicable_toys():
    future = pd.RangeIndex(30, 34)
    noisy = [calibrated_shift_check(pipeline("noisy", [7, seed]), context(seed), future)["status"]
             for seed in range(60)]
    level = [calibrated_shift_check(pipeline("level_noise", [8, seed]), context(seed), future)["status"]
             for seed in range(20)]
    assert set(noisy) <= {"PASS", "FAIL"} and noisy.count("FAIL") <= 9
    assert level.count("FAIL") >= 15


class Squares:
    """A deterministic forecaster that is not shift equivariant: the mean of the squared context."""

    def clone_unfitted(self):
        return Squares()

    def fit(self, data):
        self.level, self.names = (data.to_numpy() ** 2).mean(axis=0), tuple(data.columns)
        return self

    def predict(self, future_index):
        return Forecast(index=future_index.copy(), point=np.tile(self.level, (len(future_index), 1)),
                        target_names=self.names)


def test_a_deterministic_forecaster_is_not_decided_by_rounding():
    # Every refit equals R, so the calibration distances are all 0 and the rank p-value is 1/40 as soon as rounding
    # makes G - c differ from R. The check passes when G - c equals R within the tolerances.
    future = pd.RangeIndex(30, 32)
    exact = pd.DataFrame({"y0": np.arange(30.0), "y1": np.full(30, 2.0)})
    assert calibrated_shift_check(pipeline("mean", [9]), exact, future)["status"] == "PASS"
    rounded = calibrated_shift_check(pipeline("mean", [9]), context(3), future)
    assert rounded["calibration_max"] == 0.0 and rounded["within_tolerance"] is True
    assert rounded["status"] == "PASS"
    assert rounded["p_value"] == (1 / 40 if rounded["observed"] > 0 else 1.0)
    strict = calibrated_shift_check(pipeline("mean", [9]), context(3), future, atol=0.0, rtol=0.0)
    assert strict["status"] == ("FAIL" if strict["observed"] > 0 else "PASS")
    broken = calibrated_shift_check(Pipeline(Squares()), context(3), future)
    assert broken["calibration_max"] == 0.0 and broken["within_tolerance"] is False
    assert broken["status"] == "FAIL" and broken["p_value"] == 1 / 40


def test_errors_and_the_per_target_check():
    class Raising:
        def predict(self, context, future_index):
            raise RuntimeError("toy")

    result = calibrated_shift_check(Raising(), context(4), pd.RangeIndex(30, 32))
    assert result["status"] == "ERROR" and "toy" in result["error"] and result["p_value"] is None
    assert result["within_tolerance"] is None
    data, future = context(5), pd.RangeIndex(30, 33)
    model = pipeline("mean", [10])
    forecast = model.predict(data, future)
    agreement = target_shift_agreement(model, data, future, forecast, shifts=[7.0, -11.0], atol=1e-8, rtol=1e-7)
    assert agreement["alarm"] is False
