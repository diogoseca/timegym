"""The Study 3 subject, its second implementation and its learners, on hand-made data and on the stage-1 pilot seeds.

Only the clean subject runs here. The pilot seeds (6900000 + 1000 c + i, i < 2) are excluded from every registered
namespace; no check, alarm, loss or distance to an answer key is computed on them.
"""
import ast
import json
import warnings

import numpy as np
import pandas as pd
import pytest

from timegym import Pipeline

from experiments import study3_catalogue as catalogue
from experiments import study3_common as common
from experiments import study3_learners as learners
from experiments import study3_subject as subject
from experiments.study3_reference import lag_columns, reference_forecast
from experiments.run_supplement import scenario_from_condition

SOURCE = (common.ROOT / "experiments/study3_subject.py").read_text()
PILOT_SEEDS = [(c, 6_900_000 + 1000 * c + i) for c in range(6) for i in range(2)]


def hand_made(n=40, seed=3):
    generator = np.random.default_rng(seed)
    t = np.arange(n)
    return pd.DataFrame({"a": 5 + np.sin(2 * np.pi * t / 12) + 0.3 * generator.normal(size=n),
                         "b": np.cumsum(generator.normal(size=n)) - 2})


class LastSeasonal:
    """A stub regressor that predicts the seasonal-lag columns, to make recursion visible."""

    def fit(self, features, targets):
        self.width = targets.shape[1]
        return self

    def predict(self, features):
        return features[:, -self.width:].copy()


def test_training_matrix_and_prediction_row_on_hand_made_values():
    values = np.arange(40, dtype=float).reshape(20, 2)
    features, targets = subject.training_matrix(values, 3, 5)
    assert features.shape == (15, 3 * 2 + 2) and targets.shape == (15, 2)
    np.testing.assert_array_equal(features[0], [4, 5, 6, 7, 8, 9, 0, 1])
    np.testing.assert_array_equal(targets[0], [10, 11])
    np.testing.assert_array_equal(subject.prediction_row(values, 3, 5), [[34, 35, 36, 37, 38, 39, 30, 31]])
    reference_features, reference_targets = lag_columns(values, 3, 5)
    np.testing.assert_array_equal(np.sort(reference_features, axis=1), np.sort(features, axis=1))
    np.testing.assert_array_equal(reference_targets, targets)


def test_scalers():
    array = np.array([[1.0, 2.0, 7.0], [3.0, 2.0, 9.0]])
    center, scale = subject.fit_scaler(array)
    np.testing.assert_array_equal(center, [2.0, 2.0, 8.0])
    np.testing.assert_array_equal(scale, [1.0, 1.0, 1.0])
    scaled = subject.apply_scaler(array, center, scale)
    np.testing.assert_allclose(subject.invert_scaler(scaled, center, scale), array)


def test_recursive_forecast_shapes_and_clone():
    context = hand_made()
    model = subject.SubjectForecaster(LastSeasonal, window=6, period=12, scaling="none")
    clone = model.clone_unfitted()
    assert clone is not model and clone.make_regressor is model.make_regressor
    assert (clone.window, clone.period, clone.scaling) == (6, 12, "none")
    forecast = Pipeline(model).predict(context, pd.RangeIndex(40, 54))
    assert forecast.validate() == [] and forecast.point.shape == (14, 2)
    assert forecast.target_names == ("a", "b")
    values = context.to_numpy()
    np.testing.assert_array_equal(forecast.point[:12], values[-12:])
    np.testing.assert_array_equal(forecast.point[12:], values[-12:-10])


def test_the_frozen_structure():
    tree = ast.parse(SOURCE)
    top = [type(node).__name__ for node in tree.body]
    assert top == ["Expr", "Import", "ImportFrom"] + ["FunctionDef"] * 5 + ["ClassDef"]
    assert ast.unparse(tree.body[1]) == "import numpy as np"
    assert ast.unparse(tree.body[2]) == "from timegym import Forecast"
    assert [node.name for node in tree.body[3:8]] == ["training_matrix", "prediction_row", "fit_scaler",
                                                      "apply_scaler", "invert_scaler"]
    methods = [node.name for node in tree.body[8].body if isinstance(node, ast.FunctionDef)]
    assert tree.body[8].name == "SubjectForecaster"
    assert methods == ["__init__", "clone_unfitted", "fit", "predict"]
    assert not any(isinstance(node, (ast.Raise, ast.Assert, ast.Try)) for node in ast.walk(tree))
    defaults = [ast.unparse(value) for value in tree.body[8].body[1].args.defaults]
    assert defaults == ["6", "12", "'global'"]


def test_every_site_is_one_line():
    lines = catalogue.site_lines(SOURCE)
    tree = ast.parse(SOURCE)
    statements = [node for node in ast.walk(tree) if isinstance(node, ast.stmt)]
    for site, line in lines.items():
        starting = [node for node in statements if node.lineno == line]
        assert len(starting) == 1, site
        assert site == "H" or starting[0].end_lineno == line, site
        assert SOURCE.splitlines()[line - 1].endswith(f"  # site: {site}")


def test_reference_agrees_with_the_clean_ridge_subject_on_hand_made_data():
    for seed, length in ((1, 30), (2, 61), (5, 120)):
        context = hand_made(length, seed)
        future = pd.RangeIndex(length, length + 9)
        clean = Pipeline(learners.build("ridge")).predict(context, future)
        independent = reference_forecast(context, future)
        np.testing.assert_allclose(clean.point, independent.point, atol=1e-8, rtol=1e-7)
    constant = hand_made(50, 4).assign(c=1.5)
    future = pd.RangeIndex(50, 56)
    np.testing.assert_allclose(Pipeline(learners.build("ridge", n_targets=3)).predict(constant, future).point,
                               reference_forecast(constant, future).point, atol=1e-8, rtol=1e-7)


def test_reference_agrees_with_the_clean_ridge_subject_on_the_pilot_seeds():
    conditions = json.loads((common.ROOT / "experiments/protocol-v2.json").read_text())["model_study"]["conditions"]
    for c, seed in PILOT_SEEDS:
        case = scenario_from_condition(conditions[c]).generate(seed=seed)
        context, future = case.task.context, case.task.future_index
        clean = Pipeline(learners.build("ridge")).predict(context, future)
        assert clean.validate(case.task) == []
        np.testing.assert_allclose(clean.point, reference_forecast(context, future).point, atol=1e-8, rtol=1e-7)


def test_learner_interface():
    assert learners.LEARNERS == ("ridge", "hgb", "mlp_window_standardized", "mlp_raw")
    assert learners.SCALING == {"ridge": "global", "hgb": "none", "mlp_window_standardized": "none",
                                "mlp_raw": "none"}
    protocol = json.loads(common.PROTOCOL.read_text())
    for name, spec in protocol["part_2_learners"]["learners"].items():
        if isinstance(spec, dict):
            assert learners.SCALING[name] == spec["subject_scaling"]
    stream, again = learners.SeedStream([1, 2, 3]), np.random.default_rng(np.random.SeedSequence([1, 2, 3]))
    assert [stream.draw() for _ in range(3)] == [int(again.integers(0, 2**31 - 1)) for _ in range(3)]
    with pytest.raises(ValueError):
        learners.make_regressor("mlp_raw")
    with pytest.raises(ValueError):
        learners.make_regressor("forest")
    model = learners.build("hgb")
    assert isinstance(model, subject.SubjectForecaster) and model.scaling == "none"
    assert (model.window, model.period) == (6, 12)


def test_window_standardization_and_seed_draws():
    features, targets = subject.training_matrix(hand_made().to_numpy(), 6, 12)
    stream = learners.SeedStream([9])
    regressor = learners.make_regressor("mlp_window_standardized", stream)()
    center, scale, inputs = regressor._standardize(features)
    lags = features[:, :12].reshape(-1, 6, 2)
    np.testing.assert_allclose(center, lags.mean(axis=1))
    np.testing.assert_allclose(scale, lags.std(axis=1))
    np.testing.assert_allclose(inputs[:, :12].reshape(-1, 6, 2).mean(axis=1), 0, atol=1e-12)
    np.testing.assert_allclose(inputs[:, 12:], (features[:, 12:] - center) / scale)
    shifted = regressor._standardize(features + 10.0)[2]
    np.testing.assert_allclose(shifted, inputs, atol=1e-9)
    expected = learners.SeedStream([9])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        regressor.fit(features, targets)
    assert regressor.network.random_state == expected.draw()
    assert regressor.predict(features[:3]).shape == (3, 2)
    raw = learners.make_regressor("mlp_raw", stream)()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw.fit(features, targets)
    assert raw.network.random_state == expected.draw() and raw.predict(features[:1]).shape == (1, 2)


def test_every_learner_forecasts_one_series():
    from sklearn.exceptions import DataConversionWarning

    context = hand_made()[["a"]]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        warnings.simplefilter("error", DataConversionWarning)
        for name in learners.LEARNERS:
            forecast = Pipeline(learners.build(name, learners.SeedStream([11]), n_targets=1)).predict(
                context, pd.RangeIndex(40, 45))
            assert forecast.validate() == [] and forecast.point.shape == (5, 1), name
