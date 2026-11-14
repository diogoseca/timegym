"""Independent boundary and moment checks for controlled non-Gaussian fixtures."""
import json

import numpy as np
import pandas as pd
import pytest

from timegym import Scenario
from timegym.processes import (
    PROCESS_KINDS, resolve_process,
    _filter_regimes,
)
from timegym.scenarios import scenario_label


def case(kind, parameters=None, length=24, horizon=4, seed=19):
    return Scenario(kind, length, horizon, parameters or {}, 1, seed).generate()


def assert_simulation_moments(reference, n=40_000):
    """Use empirical Monte Carlo errors, including non-Gaussian fourth moments."""
    samples = reference.sample(n, seed=98321)[:, :, 0]
    centered = samples-reference.mean[:, 0]
    mean_se = samples.std(axis=0, ddof=1)/np.sqrt(n)
    assert np.all(np.abs(samples.mean(0)-reference.mean[:, 0]) < 7*mean_se+1e-9)
    products = centered[:, :, None]*centered[:, None, :]
    covariance_se = products.std(axis=0, ddof=1)/np.sqrt(n)
    assert np.all(np.abs(products.mean(0)-reference.covariance) < 7*covariance_se+1e-9)


@pytest.mark.parametrize('kind', PROCESS_KINDS)
def test_replay_prefix_and_rng_isolation(kind):
    state = np.random.get_state()
    short = case(kind, horizon=2)
    longer = case(kind, horizon=9)
    pd.testing.assert_frame_equal(short.task.context, longer.task.context)
    np.testing.assert_allclose(short.reference.mean, longer.reference.mean[:2])
    np.testing.assert_allclose(short.reference.covariance, longer.reference.covariance[:2, :2])
    assert short.reference.conditioning['future_observations_used'] is False
    manifest = json.loads(json.dumps(short.manifest, allow_nan=False))
    replay = Scenario(**{key: manifest[key] for key in (
        'kind', 'parameters', 'context_length', 'horizon', 'n_targets', 'seed')}).generate()
    pd.testing.assert_frame_equal(short.observed, replay.observed)
    assert short.reference.forecast(n_samples=40, seed=2).validate(short.task) == []
    np.testing.assert_array_equal(short.reference.sample(30, 2), replay.reference.sample(30, 2))
    after = np.random.get_state()
    assert after[0] == state[0] and after[2:] == state[2:]
    np.testing.assert_array_equal(after[1], state[1])


@pytest.mark.parametrize('kind,parameters', [
    ('composite', {'slope': .1, 'amplitude': 2., 'residual': 'white_noise'}),
    ('composite', {'residual': 'random_walk', 'jump_rate': .5, 'jump_mean': .1, 'jump_sigma': .3}),
    ('merton_jump_diffusion', {'output': 'log_return', 'dt': .1}),
    ('merton_jump_diffusion', {'output': 'log_price', 'dt': .1}),
    ('merton_jump_diffusion', {'output': 'price', 'dt': .1}),
    ('garch', {'output': 'log_return'}),
    ('garch', {'output': 'log_price'}),
    ('markov_switching', {'output': 'log_return', 'dt': .1}),
    ('markov_switching', {'output': 'log_price', 'dt': .1}),
    ('markov_switching', {'output': 'price', 'dt': .1}),
    ('markov_switching', {'output': 'log_return', 'dt': .1,
                          'regime_jump_rate': [.2, 1.], 'regime_jump_mean': [-.02, -.1],
                          'regime_jump_sigma': [.02, .1]}),
    ('markov_switching', {'output': 'log_price', 'dt': .1,
                          'regime_jump_rate': [.2, 1.], 'regime_jump_mean': [-.02, -.1],
                          'regime_jump_sigma': [.02, .1]}),
    ('markov_switching', {'output': 'price', 'dt': .1,
                          'regime_jump_rate': [.2, 1.], 'regime_jump_mean': [-.02, -.1],
                          'regime_jump_sigma': [.02, .1]}),
])
def test_analytic_moments_match_conditional_simulation(kind, parameters):
    assert_simulation_moments(case(kind, parameters, horizon=3).reference)


def test_composite_change_is_explicit_and_fresh_vs_accumulated_shocks_differ():
    parameters = dict(intercept=2., slope=.1, amplitude=3., period=4., phase=.2,
                      sigma=.5, change_point=8, changes={'amplitude': 1., 'slope': .2, 'sigma': 2.})
    fresh = case('composite', parameters, length=6, horizon=5)
    t = np.arange(6, 11)
    expected = 2+np.where(t >= 8, .2, .1)*t + np.where(t >= 8, 1., 3.)*np.sin(2*np.pi*t/4+.2)
    np.testing.assert_allclose(fresh.reference.mean[:, 0], expected)
    np.testing.assert_allclose(fresh.reference.variance[:, 0], [.25, .25, 4, 4, 4])
    accumulated = case('composite', {**parameters, 'residual': 'random_walk'}, length=6, horizon=5)
    np.testing.assert_allclose(accumulated.reference.variance[:, 0], [.25, .5, 4.5, 8.5, 12.5])
    assert fresh.reference.kind == 'analytic_conditional'
    assert accumulated.reference.kind == 'analytic_conditional'
    assert fresh.reference.conditioning['future_parameter_schedule_known'] is True
    assert not hasattr(fresh.task, 'parameters')


def test_composite_same_seed_holds_noise_fixed_when_mean_changes():
    common = dict(amplitude=2., period=12., sigma=.3)
    baseline = case('composite', common, length=20, horizon=4)
    changed = case('composite', {**common, 'change_point': 10, 'changes': {'amplitude': 4.}}, length=20, horizon=4)
    y0 = np.r_[baseline.task.context.y, baseline.observed.y]
    y1 = np.r_[changed.task.context.y, changed.observed.y]
    t = np.arange(len(y0))
    np.testing.assert_allclose(y1-y0, np.where(t >= 10, 2*np.sin(2*np.pi*t/12), 0), atol=1e-14)


@pytest.mark.parametrize('kind,parameters,jump_settings', [
    ('composite', {'residual': 'random_walk'}, {'jump_rate': 3., 'jump_mean': 0., 'jump_sigma': 0.}),
    ('merton_jump_diffusion', {'jump_rate': 0., 'output': 'log_price'}, {'jump_rate': 3., 'jump_mean': 0., 'jump_sigma': 0.}),
    ('markov_switching', {}, {'regime_jump_rate': [2., 4.], 'regime_jump_mean': [0., 0.], 'regime_jump_sigma': [0., 0.]}),
])
def test_inert_jump_component_does_not_change_diffusion_or_state_path(kind, parameters, jump_settings):
    baseline = case(kind, parameters)
    jumps = case(kind, {**parameters, **jump_settings})
    pd.testing.assert_frame_equal(baseline.task.context, jumps.task.context)
    pd.testing.assert_frame_equal(baseline.observed, jumps.observed)


def test_merton_zero_jumps_reduce_to_gbm_and_price_mean_obeys_physical_drift():
    parameters = dict(mu=.12, sigma=.3, jump_rate=0., dt=.2)
    price = case('merton_jump_diffusion', parameters, horizon=3)
    last = price.task.context.y.iloc[-1]
    steps = np.arange(1, 4)
    np.testing.assert_allclose(price.reference.mean[:, 0], last*np.exp(.12*.2*steps))
    logs = case('merton_jump_diffusion', {**parameters, 'output': 'log_price'}, horizon=3)
    np.testing.assert_allclose(np.log(price.task.context.y), logs.task.context.y)
    np.testing.assert_allclose(logs.reference.variance[:, 0], .3**2*.2*steps)
    assert logs.reference.kind == 'analytic_conditional'
    assert np.all(price.reference.mean[:, 0] > np.exp(logs.reference.mean[:, 0]))
    jumps = case('merton_jump_diffusion', {**parameters, 'jump_rate': 1.}, horizon=3)
    np.testing.assert_allclose(jumps.reference.mean[:, 0], jumps.task.context.y.iloc[-1]*np.exp(.12*.2*steps))


def test_garch_variance_is_history_conditioned_and_paths_use_their_own_innovations():
    parameters = dict(mu=.01, omega=.02, alpha=.35, beta=.4, initial_variance=.1)
    generated = case('garch', parameters, horizon=3)
    variance = .1
    for value in generated.task.context.y:
        variance = .02+.35*(value-.01)**2+.4*variance
    expected = [variance]
    for _ in range(2):
        expected.append(.02+.75*expected[-1])
    np.testing.assert_allclose(generated.reference.variance[:, 0], expected)
    np.testing.assert_allclose(generated.reference.covariance, np.diag(expected))
    samples = generated.reference.sample(40_000, seed=4)[:, :, 0]-.01
    # A fake sampler using only the expected variance misses volatility clustering.
    assert np.corrcoef(samples[:, 0]**2, samples[:, 1]**2)[0, 1] > .15


def test_constant_garch_variance_reduces_to_independent_gaussian_returns():
    generated = case('garch', {'omega': .04, 'alpha': 0., 'beta': 0., 'initial_variance': .04}, horizon=4)
    np.testing.assert_allclose(generated.reference.covariance, np.eye(4)*.04)
    np.testing.assert_allclose(generated.reference.mean, 0.)


def test_hidden_regime_filter_matches_manual_bayes_and_does_not_use_future():
    parameters = dict(transition=[[.9, .1], [.2, .8]], initial_probs=[.6, .4],
                      regime_mu=[.5, -.5], regime_sigma=[1., 1.], dt=1.)
    p = resolve_process('markov_switching', parameters, 10, 3, 1, 2)
    observations = np.array([.2, -.3, .8])
    posterior = np.array([.6, .4])
    for value in observations:
        likelihood = np.exp(-.5*(value-np.array([0., -1.]))**2)
        posterior = (posterior@np.array(p['transition']))*likelihood
        posterior /= posterior.sum()
    filtered, tail = _filter_regimes(observations, p)
    np.testing.assert_allclose(filtered, posterior)
    assert tail == 0
    short = case('markov_switching', parameters, horizon=1)
    long = case('markov_switching', parameters, horizon=12)
    np.testing.assert_array_equal(short.reference.conditioning['state_posterior'], long.reference.conditioning['state_posterior'])
    assert not hasattr(short.task, 'state_posterior')
    assert 'latent_states' not in short.manifest


def test_identical_regimes_reduce_to_single_merton_distribution():
    common = dict(mu=.1, sigma=.2, jump_rate=.3, jump_mean=-.05, jump_sigma=.1, dt=.2, output='log_return')
    single = case('merton_jump_diffusion', common, horizon=3)
    regimes = case('markov_switching', dict(
        regime_mu=[.1, .1], regime_sigma=[.2, .2], regime_jump_rate=[.3, .3],
        regime_jump_mean=[-.05, -.05], regime_jump_sigma=[.1, .1], dt=.2), horizon=3)
    np.testing.assert_allclose(regimes.reference.mean, single.reference.mean)
    np.testing.assert_allclose(regimes.reference.covariance, single.reference.covariance, atol=1e-15)
    assert regimes.reference.sampler_kind == 'approximate_conditional'
    assert 0 <= regimes.reference.conditioning['emission_poisson_tail_bound'] <= 1e-12
    assert 'not_a_posterior_error_bound' in regimes.reference.conditioning['filter_note']


def test_deterministic_regimes_and_single_observation_price_context():
    parameters = dict(regime_mu=[0., 0.], regime_sigma=[0., 0.], output='price')
    generated = case('markov_switching', parameters, length=1, horizon=3)
    np.testing.assert_allclose(generated.reference.mean, 100.)
    np.testing.assert_allclose(generated.reference.covariance, 0., atol=1e-10)
    np.testing.assert_allclose(generated.reference.sample(4), 100.)
    longer = case('markov_switching', parameters, length=10, horizon=3)
    np.testing.assert_allclose(longer.task.context.y, 100.)


@pytest.mark.parametrize('kind,parameters', [
    ('composite', {'sigma': -1}),
    ('composite', {'period': 0}),
    ('composite', {'changes': {'sigma': 2}}),
    ('composite', {'change_point': 4}),
    ('composite', {'change_point': 4, 'changes': {'jump_rate': 1}}),
    ('composite', {'jump_rate': 1}),
    ('composite', {'residual': 'arbitrary_latent_sum'}),
    ('merton_jump_diffusion', {'initial_price': 0}),
    ('merton_jump_diffusion', {'dt': -1}),
    ('merton_jump_diffusion', {'output': 'returns'}),
    ('merton_jump_diffusion', {'jump_mean': 1000}),
    ('merton_jump_diffusion', {'sigma': 1j}),
    ('garch', {'alpha': .5, 'beta': .5}),
    ('garch', {'omega': 0}),
    ('garch', {'output': 'price'}),
    ('garch', {'initial_variance': -1}),
    ('markov_switching', {'transition': [[.1, .1], [.5, .5]]}),
    ('markov_switching', {'transition': [[1, 0], [-1, 2]]}),
    ('markov_switching', {'initial_probs': [1, 1]}),
    ('markov_switching', {'regime_sigma': [1]}),
    ('markov_switching', {'regime_jump_rate': [1e6, 1e6]}),
    ('markov_switching', {'regime_sigma': [float('nan'), 1]}),
])
def test_invalid_parameters_fail_explicitly(kind, parameters):
    with pytest.raises(ValueError):
        case(kind, parameters)


@pytest.mark.parametrize('kind', PROCESS_KINDS)
def test_extended_processes_support_joint_targets_and_reject_unknown_fields(kind):
    assert Scenario(kind, 20, 4, {}, 2, 1).generate().reference.mean.shape == (4, 2)
    with pytest.raises(ValueError, match='Unsupported'):
        case(kind, {'typo': 3})
    for seed in (True, -1, 1.5):
        with pytest.raises(ValueError, match='seed'):
            case(kind, seed=seed)


def test_nonfinite_outputs_are_rejected_instead_of_clipped():
    with pytest.raises(ValueError, match='overflowed'):
        case('merton_jump_diffusion', {'mu': 1000., 'dt': 1., 'sigma': 0., 'jump_rate': 0.})
    with pytest.raises(ValueError, match='underflowed'):
        case('merton_jump_diffusion', {'mu': -1000., 'dt': 1., 'sigma': 0., 'jump_rate': 0.})


def test_process_labels_separate_seasonality_accumulation_and_parameter_change():
    assert scenario_label('composite', {'amplitude': 2.}) == 'Seasonality + white noise'
    assert 'accumulated shocks' in scenario_label('composite', {'amplitude': 2., 'residual': 'random_walk'})
    assert 'parameter change' in scenario_label('composite', {'change_point': 10, 'changes': {'sigma': 2.}})
    assert scenario_label('markov_switching', {'regime_jump_rate': [0., 2.]}) == 'Hidden regimes + jumps (log returns)'
    assert scenario_label('merton_jump_diffusion') == 'Merton jump diffusion (price)'


def test_fresh_and_accumulated_residuals_share_innovations_at_fixed_seed():
    fresh = case('composite', {'residual': 'white_noise'}, length=20, horizon=4)
    walk = case('composite', {'residual': 'random_walk'}, length=20, horizon=4)
    fresh_values = np.r_[fresh.task.context.y, fresh.observed.y]
    walk_values = np.r_[walk.task.context.y, walk.observed.y]
    np.testing.assert_allclose(np.diff(walk_values), fresh_values[1:], atol=1e-14)


@pytest.mark.parametrize('output', ['log_price', 'price'])
def test_hidden_regime_joint_moments_against_exhaustive_path_enumeration(output):
    from itertools import product
    transition = np.array([[.8, .2], [.3, .7]])
    prior = np.array([.6, .4])
    drifts = np.array([.2, -.3])
    generated = case('markov_switching', dict(
        transition=transition.tolist(), initial_probs=prior.tolist(),
        regime_mu=drifts.tolist(), regime_sigma=[0., 0.], dt=1., output=output),
        length=1, horizon=3)
    weights, paths = [], []
    for path in product(range(2), repeat=4):
        weight = prior[path[0]]
        for before, after in zip(path[:-1], path[1:]):
            weight *= transition[before, after]
        log_path = np.log(100.)+np.cumsum(drifts[list(path[1:])])
        weights.append(weight)
        paths.append(np.exp(log_path) if output == 'price' else log_path)
    weights, paths = np.asarray(weights), np.asarray(paths)
    expected = weights@paths
    covariance = (paths-expected).T@(weights[:, None]*(paths-expected))
    np.testing.assert_allclose(generated.reference.mean[:, 0], expected)
    np.testing.assert_allclose(generated.reference.covariance, covariance, atol=1e-10)


def test_merton_zero_jump_label_makes_the_gbm_reduction_explicit():
    assert scenario_label('merton_jump_diffusion', {'jump_rate': 0., 'output': 'log_price'}) == 'Geometric Brownian motion (log price)'
    assert scenario_label('merton_jump_diffusion', {'jump_rate': .1, 'output': 'price'}) == 'Merton jump diffusion (price)'


@pytest.mark.parametrize('output', ['price', 'log_price', 'log_return'])
@pytest.mark.parametrize('transition,prior', [([[1.]], [1.]), ([[.8, .2], [.3, .7]], [.6, .4])])
def test_identical_deterministic_regimes_retain_exact_zero_covariance_at_long_horizons(output, transition, prior):
    n = len(prior)
    generated = case('markov_switching', dict(
        transition=transition, initial_probs=prior, regime_mu=[.05]*n,
        regime_sigma=[0.]*n, regime_jump_rate=[0.]*n,
        regime_jump_mean=[0.]*n, regime_jump_sigma=[0.]*n, output=output),
        length=12, horizon=100, seed=2)
    np.testing.assert_array_equal(generated.reference.covariance, np.zeros((100, 100)))
    np.testing.assert_array_equal(generated.reference.variance, np.zeros((100, 1)))
    draws = generated.reference.sample(3, seed=7)
    np.testing.assert_allclose(draws, np.broadcast_to(generated.reference.mean, draws.shape), atol=1e-10)


def test_manifest_preserves_canonical_reference_capabilities():
    generated = case('merton_jump_diffusion')
    assert generated.manifest['conditioning'] == generated.reference.conditioning
    assert generated.manifest['conditioning'] is not generated.reference.conditioning
    assert generated.manifest['conditioning']['sampler_kind'] == generated.reference.sampler_kind


def test_mixed_atomic_and_continuous_regime_emissions_are_rejected_even_at_tiny_scale():
    with pytest.raises(ValueError, match='must not mix point masses'):
        case('markov_switching', dict(transition=[[1., 0.], [0., 1.]], initial_probs=[.5, .5],
             regime_mu=[0., 0.], regime_sigma=[0., 1e-15], dt=1., output='log_return'), length=1, horizon=2, seed=4)
    with pytest.raises(ValueError, match='must not mix point masses'):
        case('markov_switching', dict(regime_sigma=[0., 0.], regime_jump_rate=[1., 1.],
             regime_jump_sigma=[.1, .1]))


def test_discrete_regime_filter_uses_exact_support_not_a_tolerance_band():
    parameters = dict(transition=[[1., 0.], [0., 1.]], initial_probs=[.5, .5],
                      regime_mu=[0., 1e-14], regime_sigma=[0., 0.], dt=1., output='log_return')
    resolved = resolve_process('markov_switching', parameters, 1, 2, 1, 4)
    filtered, _ = _filter_regimes([0.], resolved)
    np.testing.assert_array_equal(filtered, [1., 0.])
    for output in ('price', 'log_price'):
        with pytest.raises(ValueError, match='reconstructing discrete returns'):
            case('markov_switching', {**parameters, 'output': output}, length=2)


@pytest.mark.parametrize('kind,parameters', [
    ('merton_jump_diffusion', {'output': 'log_return'}),
    ('garch', {'omega': 1., 'alpha': 0., 'beta': 0., 'initial_variance': 1.}),
])
def test_reference_zero_seed_does_not_reuse_observed_or_held_out_innovations(kind, parameters):
    generated = case(kind, parameters, length=12, horizon=2, seed=0)
    samples = generated.reference.sample(8192, seed=0)
    assert not np.array_equal(samples[:12, 0, 0], generated.task.context.y.to_numpy())
    assert samples[12, 0, 0] != generated.observed.y.iloc[0]
    np.testing.assert_array_equal(samples, generated.reference.sample(8192, seed=0))


def test_all_named_reference_streams_have_a_separate_seed_namespace():
    from timegym._process_common import _streams, _rng
    observation, reference = _streams(0), _streams(0, role=1)
    for key in observation:
        assert not np.array_equal(observation[key].standard_normal(20), reference[key].standard_normal(20))
    assert not np.array_equal(np.random.default_rng(0).standard_normal(20), _rng(0, role=1).standard_normal(20))


@pytest.mark.parametrize('kind', PROCESS_KINDS)
@pytest.mark.parametrize('correlation', [-1., 0., .7, 1., [[1.]]])
def test_one_target_processes_accept_correlation_without_changing_scalar_laws(kind, correlation):
    baseline = case(kind)
    correlated = case(kind, {'correlation': correlation})
    assert correlated.manifest == baseline.manifest
    pd.testing.assert_frame_equal(correlated.task.context, baseline.task.context)
    np.testing.assert_array_equal(correlated.reference.sample(8, 9), baseline.reference.sample(8, 9))


@pytest.mark.parametrize('kind,scalar,vector', [
    ('composite', {'sigma': .3, 'change_point': 4, 'changes': {'slope': .2}},
                  {'sigma': [.3], 'change_point': 4, 'changes': {'slope': [.2]}}),
    ('merton_jump_diffusion', {'mu': .2}, {'mu': [.2]}),
    ('garch', {'alpha': .05, 'initial_variance': .01}, {'alpha': [.05], 'initial_variance': [.01]}),
    ('markov_switching', {'regime_mu': [.1, -.2]}, {'regime_mu': [[.1], [-.2]]}),
])
def test_length_one_parameter_vectors_collapse_to_scalar_manifests(kind, scalar, vector):
    left, right = case(kind, scalar), case(kind, vector)
    assert left.manifest == right.manifest
    pd.testing.assert_frame_equal(left.task.context, right.task.context)
    np.testing.assert_array_equal(left.reference.sample(8, 9), right.reference.sample(8, 9))


@pytest.mark.parametrize('kind', PROCESS_KINDS)
@pytest.mark.parametrize('correlation', [-1.1, 1.1, [[.5]], [[1, 0], [0, 1]]])
def test_one_target_processes_validate_correlation(kind, correlation):
    with pytest.raises(ValueError, match='correlation'):
        case(kind, {'correlation': correlation})


@pytest.mark.parametrize('kind', PROCESS_KINDS)
@pytest.mark.parametrize('n_targets', [1, 2])
def test_controlled_scenario_resolves_once_and_owns_its_parameter_snapshot(monkeypatch, kind, n_targets):
    import timegym.scenarios as scenarios
    calls = []
    original = scenarios.resolve_process

    def counted(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(scenarios, 'resolve_process', counted)
    specification = scenarios.Scenario(kind, context_length=12, horizon=2, n_targets=n_targets)
    first = specification.generate(seed=8)
    specification.display_name
    again = specification.generate(seed=8)
    assert len(calls) == 1
    assert first.manifest == again.manifest
    first.manifest['parameters']['unrelated'] = 1
    assert 'unrelated' not in specification.generate(seed=8).manifest['parameters']
    with pytest.raises(ValueError, match='seed'):
        specification.generate(seed=-1)


def test_controlled_parameters_are_copied_at_construction():
    from timegym import Scenario
    parameters = {'change_point': 3, 'changes': {'slope': .2}}
    specification = Scenario('composite', context_length=8, horizon=2, parameters=parameters)
    original = specification.generate()
    parameters['changes']['slope'] = 2.
    pd.testing.assert_frame_equal(original.observed, specification.generate().observed)
    assert not original.observed.equals(Scenario('composite', 8, 2, parameters).generate().observed)


def test_case_contract_retains_public_import_compatibility():
    from timegym import Case
    from timegym.contracts import Case as ContractCase
    from timegym.scenarios import Case as ScenarioCase
    assert Case is ContractCase is ScenarioCase
