"""Human labels distinguish mechanisms without changing replay identifiers."""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from timegym.scenarios import Scenario, scenario_label


@pytest.mark.parametrize("kind,parameters,label", [
    ("white_noise", {}, "Gaussian white noise (fixed mean)"),
    ("trend", {}, "Linear trend + white noise"),
    ("seasonal", {}, "Seasonality + white noise"),
    ("ar1", {"phi": -.6}, "Stationary AR(1)"),
    ("random_walk", {}, "Random walk without drift"),
    ("random_walk", {"drift": -0.0}, "Random walk without drift"),
    ("random_walk", {"drift": .03}, "Random walk with drift"),
    ("random_walk", {"drift": -.03}, "Random walk with drift"),
    ("random_walk", {"drift": 1e-15}, "Random walk with drift"),
    ("random_walk", {"drift": .03, "sigma": 0}, "Random walk with drift"),
])
def test_labels_describe_resolved_configuration_without_renaming_kind(kind, parameters, label):
    specification = Scenario(kind, context_length=8, horizon=3, parameters=parameters)
    before = deepcopy(specification.parameters)
    assert specification.display_name == label
    assert scenario_label(kind, parameters) == label
    case = specification.generate(seed=7)
    assert scenario_label(case.manifest["kind"], case.manifest["parameters"]) == label
    assert specification.kind == case.manifest["kind"] == kind
    assert specification.parameters == before
    assert "display_name" not in case.manifest


def test_mixed_drift_labels_select_the_requested_target_and_preserve_small_drift():
    parameters = {"drift": [0.0, 1e-15, -.2], "correlation": .4}
    specification = Scenario("random_walk", n_targets=3, parameters=parameters)
    assert specification.display_name == "Random walk (mixed drift across targets)"
    assert scenario_label("random_walk", parameters, target_index=0) == "Random walk without drift"
    for target_index in (1, 2):
        assert scenario_label("random_walk", parameters, target_index=target_index) == "Random walk with drift"
    # Scalar parameters broadcast over targets, just as in Scenario._resolve.
    assert scenario_label("random_walk", {"drift": .2}, target_index=2) == "Random walk with drift"
    assert scenario_label("random_walk", {"drift": [0., -0.]}) == "Random walk without drift"
    assert scenario_label("random_walk", {"drift": [.2, -.2]}) == "Random walk with drift"


def test_display_name_tracks_validated_mutable_parameters_and_custom_names_remain_verbatim():
    specification = Scenario("random_walk")
    assert specification.display_name == "Random walk without drift"
    specification.parameters["drift"] = .2
    assert specification.display_name == "Random walk with drift"
    specification.parameters["drift"] = np.nan
    with pytest.raises(ValueError, match="drift must be finite"):
        _ = specification.display_name
    assert scenario_label("Measured sensor signal", {"unrelated": object()}) == "Measured sensor signal"


def test_random_walk_drift_adds_deterministic_increments_to_the_same_accumulated_shocks():
    length, horizon, drift, sigma = 18, 5, .03, .7
    settings = dict(context_length=length, horizon=horizon, seed=23)
    without = Scenario("random_walk", parameters={"intercept": 2., "sigma": sigma}, **settings).generate()
    with_drift = Scenario("random_walk", parameters={"intercept": 2., "sigma": sigma, "drift": drift}, **settings).generate()
    baseline = pd.concat([without.task.context, without.observed]).to_numpy()[:, 0]
    shifted = pd.concat([with_drift.task.context, with_drift.observed]).to_numpy()[:, 0]
    np.testing.assert_allclose(shifted - baseline, drift * np.arange(length + horizon), atol=1e-12)
    np.testing.assert_allclose(np.diff(with_drift.reference.mean[:, 0]), drift)
    np.testing.assert_allclose(without.reference.mean[:, 0], baseline[length - 1])
    # Accumulated future innovations increase walk variance; adding deterministic
    # drift changes the conditional mean but neither its variance nor covariance.
    np.testing.assert_allclose(without.reference.variance[:, 0], sigma ** 2 * np.arange(1, horizon + 1))
    np.testing.assert_array_equal(without.reference.covariance, with_drift.reference.covariance)
    for kind in ("white_noise", "trend"):
        fresh_noise = Scenario(kind, parameters={"sigma": sigma}, **settings).generate()
        np.testing.assert_allclose(fresh_noise.reference.variance[:, 0], sigma ** 2)
