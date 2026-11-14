"""Independent joint-law calculations, observation-only filtering and replay."""
from itertools import product
import json

import numpy as np
import pandas as pd
import pytest
from scipy.stats import multivariate_normal

from timegym import Scenario, Pipeline, TestReport as Report, SimulationReference
from timegym.diagnostics import covariance_dependence_check, evaluate_forecast


KINDS = ('composite', 'merton_jump_diffusion', 'garch', 'markov_switching')


def generate(kind, parameters=None, *, length=24, horizon=3, d=2, seed=19):
    return Scenario(kind, length, horizon, parameters or {}, n_targets=d).generate(seed)


@pytest.mark.parametrize('kind', KINDS)
@pytest.mark.parametrize('d', [2, 3])
def test_joint_shapes_prefix_rng_and_public_replay(kind, d):
    p = {'correlation': .4}
    before = np.random.get_state()
    short = generate(kind, p, d=d, horizon=2)
    longer = generate(kind, p, d=d, horizon=6)
    pd.testing.assert_frame_equal(short.task.context, longer.task.context)
    pd.testing.assert_frame_equal(short.observed, longer.observed.iloc[:2])
    np.testing.assert_allclose(short.reference.mean, longer.reference.mean[:2])
    np.testing.assert_allclose(short.reference.covariance, longer.reference.covariance[:2*d, :2*d])
    assert short.reference.sample(32, 7).shape == (32, 2, d)
    assert short.reference.forecast(32, 7).validate(short.task) == []
    manifest = json.loads(json.dumps(short.manifest, allow_nan=False))
    replay = Scenario(**{key: manifest[key] for key in ('kind', 'parameters', 'context_length', 'horizon', 'n_targets', 'seed')}).generate()
    assert replay.manifest == manifest
    np.testing.assert_array_equal(short.reference.sample(32, 7), replay.reference.sample(32, 7))
    report = Pipeline().test([Scenario(kind, 24, 2, p, n_targets=d)], seeds=[19])
    replay_report = Report.from_json(report.to_json()).replay_case()
    assert all(v for k, v in replay_report.metadata['replay'].items() if k.endswith('match') or k.endswith('matches'))
    after = np.random.get_state()
    assert before[0] == after[0] and before[2:] == after[2:]
    np.testing.assert_array_equal(before[1], after[1])


@pytest.mark.parametrize('kind,params', [
    ('composite', {'correlation': -.4, 'sigma': [.3, .7], 'amplitude': [1., 2.]}),
    ('composite', {'correlation': .5, 'residual': 'random_walk', 'jump_rate': .3,
                   'jump_mean': [.1, -.2], 'jump_sigma': [.2, .3], 'jump_correlation': -.6}),
    *[('merton_jump_diffusion', {'output': output, 'dt': .2, 'correlation': .5,
                                'jump_correlation': -.3, 'sigma': [.2, .35],
                                'mu': [.04, -.03], 'jump_mean': [-.05, .08]})
      for output in ('price', 'log_price', 'log_return')],
    *[('garch', {'output': output, 'correlation': -.5, 'alpha': [.13, .05], 'beta': [.7, .85],
                 'omega': [.001, .002], 'moment_samples': 50000}) for output in ('log_price', 'log_return')],
    *[('markov_switching', {'output': output, 'dt': .2, 'correlation': .4,
                           'regime_mu': [[.03, .1], [-.2, -.1]],
                           'regime_sigma': [[.1, .2], [.4, .3]],
                           'regime_jump_rate': [.2, .8] if jumps else [0., 0.],
                           'regime_jump_mean': [[-.03, -.05], [-.1, -.2]],
                           'regime_jump_sigma': [.03, .1], 'jump_correlation': .5})
      for output in ('price', 'log_price', 'log_return') for jumps in (False, True)],
])
def test_joint_conditional_moments_agree_with_independent_draws(kind, params):
    case = generate(kind, params)
    ref = case.reference
    x = ref.sample(50000, 65134).reshape(50000, -1)
    centered = x-ref.mean.ravel()
    assert np.all(np.abs(centered.mean(0)) <= 7*x.std(0)/np.sqrt(len(x))+1e-10)
    products = centered[:, :, None]*centered[:, None, :]
    tolerance = 7*products.std(0)/np.sqrt(len(x))+1e-9
    # The independently estimated GARCH cross-covariance has its own MC error.
    if ref.conditioning.get('covariance_kind') == 'monte_carlo':
        tolerance += .01*np.sqrt(np.outer(ref.variance.ravel(), ref.variance.ravel()))
    assert np.all(np.abs(products.mean(0)-ref.covariance) <= tolerance)


def test_composite_vector_change_schedule_and_cross_horizon_covariance():
    p = dict(intercept=[1., 3.], slope=[.1, -.1], amplitude=[0., 2.], period=[4., 6.],
             phase=0., sigma=[.3, .7], correlation=.6, residual='random_walk',
             change_point=7, changes={'sigma': [.9, .2], 'amplitude': [1., 3.]})
    case = generate('composite', p, length=6, horizon=3)
    means = []
    for t in range(5, 9):
        amp = np.array([1., 3.] if t >= 7 else [0., 2.])
        means.append(np.array([1., 3.])+np.array([.1, -.1])*t+amp*np.sin(2*np.pi*t/np.array([4., 6.])))
    offset = case.task.context.iloc[-1].to_numpy()-means[0]
    np.testing.assert_allclose(case.reference.mean, np.array(means[1:])+offset)
    r = np.array([[1., .6], [.6, 1.]])
    q = [np.outer(s, s)*r for s in ([.3, .7], [.9, .2], [.9, .2])]
    expected = np.block([[sum(q[:min(i, j)+1]) for j in range(3)] for i in range(3)])
    np.testing.assert_allclose(case.reference.covariance, expected)
    from timegym.scenarios import scenario_label
    assert scenario_label('composite', {**p, 'amplitude': [0., 2.], 'changes': {}}, target_index=0) == 'Linear trend + accumulated shocks (parameter change)'


def test_shared_poisson_clock_produces_dependence_even_without_correlated_marks():
    p = dict(sigma=0., mu=0., jump_rate=2., jump_mean=[1., 2.], jump_sigma=0.,
             jump_correlation=0., correlation=0., dt=.2, output='log_return')
    case = generate('merton_jump_diffusion', p, horizon=2)
    expected = .4*np.array([[1., 2.], [2., 4.]])
    np.testing.assert_allclose(case.reference.covariance[:2, :2], expected)
    x = case.reference.sample(1000, 41)[:, 0]
    compensation = -np.expm1([1., 2.])*.4
    np.testing.assert_allclose(x[:, 1]-compensation[1], 2*(x[:, 0]-compensation[0]), atol=1e-14)


def test_shared_clock_can_leave_zero_covariance_but_dependent_squared_returns():
    c=generate('merton_jump_diffusion',dict(sigma=0.,mu=0.,jump_rate=1.,dt=.5,
               jump_mean=0.,jump_sigma=[.2,.3],correlation=0.,jump_correlation=0.,output='log_return'),horizon=1)
    assert c.reference.covariance[0,1]==0
    draws=c.reference.sample(50000,31)[:,0]-c.reference.mean[0]
    # E[J_i^2 J_j^2] shares the random event count despite zero signed covariance.
    assert np.corrcoef(draws[:,0]**2,draws[:,1]**2)[0,1]>.15


@pytest.mark.parametrize('kind', KINDS)
def test_joint_observation_and_reference_random_roles_are_separate(kind):
    p={} if kind=='composite' else {'output':'log_return'}
    c=generate(kind,p,length=1,horizon=3,seed=81)
    assert not np.array_equal(c.observed.values,c.reference.sample(1,81)[0])


def test_small_nonzero_garch_correlation_is_still_labelled_numerical():
    c=generate('garch',{'correlation':1e-13,'moment_samples':128},horizon=2)
    assert c.reference.moments_kind=='analytic_mean_monte_carlo_covariance'


def test_merton_price_cross_moments_from_direct_cumulant():
    p = dict(mu=[.04, -.02], sigma=[.2, .3], correlation=-.25, jump_rate=.7,
             jump_mean=[-.08, .02], jump_sigma=[.1, .15], jump_correlation=.6,
             dt=.2, output='price', initial_price=[30., 90.])
    c = generate('merton_jump_diffusion', p, length=1, horizon=3)
    sigma=np.array(p['sigma']); jm=np.array(p['jump_mean']); js=np.array(p['jump_sigma'])
    brown=np.outer(sigma,sigma)*np.array([[1.,-.25],[-.25,1.]])
    jump=np.outer(js,js)*np.array([[1.,.6],[.6,1.]])
    kappa=np.exp(jm+js**2/2)-1
    b=(np.array(p['mu'])-sigma**2/2-p['jump_rate']*kappa)*p['dt']
    def cumulant(u):
        return u@b+.5*p['dt']*(u@brown@u)+p['jump_rate']*p['dt']*(np.exp(u@jm+.5*u@jump@u)-1)
    expected=np.empty((6,6)); mean=np.empty((3,2))
    for h in range(3):
        for i in range(2):mean[h,i]=p['initial_price'][i]*np.exp((h+1)*cumulant(np.eye(2)[i]))
    for h,i,j,k in product(range(3),range(2),range(3),range(2)):
        ei,ek=np.eye(2)[i],np.eye(2)[k]
        spread=cumulant(ei+ek)-cumulant(ei)-cumulant(ek)
        expected[2*h+i,2*j+k]=mean[h,i]*mean[j,k]*np.expm1(min(h+1,j+1)*spread)
    np.testing.assert_allclose(c.reference.mean,mean)
    np.testing.assert_allclose(c.reference.covariance,expected,rtol=1e-12,atol=1e-11)


def test_garch_observed_variance_and_second_step_cross_moment_quadrature():
    p = dict(mu=[.01,-.03], omega=[.01,.02], alpha=[.15,.3], beta=[.65,.5],
             correlation=.7, moment_samples=100000, moment_seed=63)
    c=generate('garch',p,length=10,horizon=2)
    mu,omega,alpha,beta=[np.array(p[key]) for key in ('mu','omega','alpha','beta')]
    v=omega/(1-alpha-beta)
    for r in c.task.context.values:v=omega+alpha*(r-mu)**2+beta*v
    np.testing.assert_allclose(c.reference.conditioning['next_variance_from_observed_history'],v)
    np.testing.assert_allclose(c.reference.covariance[:2,:2],np.outer(np.sqrt(v),np.sqrt(v))*[[1,.7],[.7,1]])
    # Two-dimensional Gauss-Hermite integration, independent of reference MC.
    nodes,weights=np.polynomial.hermite.hermgauss(40)
    x,y=np.meshgrid(nodes*np.sqrt(2),nodes*np.sqrt(2),indexing='ij')
    z0=x;z1=.7*x+np.sqrt(1-.7**2)*y
    vv0=omega[0]+alpha[0]*v[0]*z0**2+beta[0]*v[0]
    vv1=omega[1]+alpha[1]*v[1]*z1**2+beta[1]*v[1]
    expected=.7*np.sum(np.outer(weights,weights)*np.sqrt(vv0*vv1))/np.pi
    assert c.reference.covariance[2,3] == pytest.approx(expected,rel=.005)
    np.testing.assert_array_equal(c.reference.covariance[:2,2:],np.zeros((2,2)))
    assert c.reference.moments_kind=='analytic_mean_monte_carlo_covariance'
    altered=generate('garch',{**p,'moment_seed':64},length=10,horizon=2)
    pd.testing.assert_frame_equal(c.task.context,altered.task.context)
    np.testing.assert_array_equal(c.reference.mean,altered.reference.mean)
    np.testing.assert_array_equal(c.reference.sample(17,8),altered.reference.sample(17,8))
    assert not np.array_equal(c.reference.covariance,altered.reference.covariance)


@pytest.mark.parametrize('parameters,horizon', [({'correlation':0.},4),({'alpha':0.,'correlation':.6},4),({'correlation':.6},1)])
def test_garch_analytic_covariance_special_cases(parameters,horizon):
    c=generate('garch',parameters,horizon=horizon)
    assert c.reference.moments_kind=='analytic'
    assert 'moment_samples' not in c.reference.capabilities


def test_markov_joint_filter_matches_independent_forward_calculation():
    p=dict(transition=[[.85,.15],[.2,.8]],initial_probs=[.4,.6],dt=.1,
           regime_mu=[[.2,-.1],[-.15,.3]],regime_sigma=[[.1,.2],[.3,.15]],
           regime_correlation=[[[1.,.7],[.7,1.]],[[1.,-.4],[-.4,1.]]])
    c=generate('markov_switching',p,length=12)
    posterior=np.array(p['initial_probs']);single=posterior.copy()
    for value in c.task.context.values:
        likelihood=[]; univariate=[]
        for s in range(2):
            sigma=np.array(p['regime_sigma'][s]);mu=np.array(p['regime_mu'][s])
            mean=(mu-.5*sigma**2)*p['dt'];cov=np.outer(sigma,sigma)*p['regime_correlation'][s]*p['dt']
            likelihood.append(multivariate_normal.pdf(value,mean=mean,cov=cov))
            univariate.append(np.exp(-.5*(value[0]-mean[0])**2/cov[0,0])/np.sqrt(2*np.pi*cov[0,0]))
        posterior=(posterior@p['transition'])*likelihood;posterior/=posterior.sum()
        single=(single@p['transition'])*univariate;single/=single.sum()
    np.testing.assert_allclose(c.reference.conditioning['state_posterior'],posterior,atol=1e-13)
    assert np.max(np.abs(single-posterior))>1e-4
    assert not hasattr(c.task,'state_posterior')


@pytest.mark.parametrize('output', ['log_return','log_price','price'])
def test_markov_joint_moments_against_enumerated_hidden_state_paths(output):
    p=dict(transition=[[.8,.2],[.3,.7]],initial_probs=[.4,.6],dt=.2,
           regime_mu=[[.1,-.08],[-.2,.15]],regime_sigma=[[.1,.2],[.3,.15]],
           regime_correlation=[[[1.,.6],[.6,1.]],[[1.,-.4],[-.4,1.]]],
           initial_price=[15.,30.],output=output)
    c=generate('markov_switching',p,length=1,horizon=3)
    posterior=np.array(c.reference.conditioning['state_posterior'])
    total_mean=np.zeros(6); total_second=np.zeros((6,6))
    for path in product(range(2),repeat=3):
        weight=(posterior@p['transition'])[path[0]]
        for h in range(1,3):weight*=p['transition'][path[h-1]][path[h]]
        mean=[];cov=np.zeros((6,6))
        for h,state in enumerate(path):
            sigma=np.array(p['regime_sigma'][state]);mu=np.array(p['regime_mu'][state])
            mean.append((mu-.5*sigma**2)*p['dt'])
            cov[2*h:2*h+2,2*h:2*h+2]=np.outer(sigma,sigma)*p['regime_correlation'][state]*p['dt']
        mean=np.array(mean)
        if output!='log_return':
            transform=np.kron(np.tril(np.ones((3,3))),np.eye(2))
            cov=transform@cov@transform.T
            last=c.task.context.iloc[-1].to_numpy()
            mean=(np.log(last) if output=='price' else last)+mean.cumsum(0)
        mean=mean.ravel()
        if output=='price':
            m=np.exp(mean+np.diag(cov)/2)
            second=np.outer(m,m)*np.exp(cov)
        else:
            m=mean;second=cov+np.outer(m,m)
        total_mean+=weight*m;total_second+=weight*second
    np.testing.assert_allclose(c.reference.mean.ravel(),total_mean,atol=1e-12)
    np.testing.assert_allclose(c.reference.covariance,total_second-np.outer(total_mean,total_mean),atol=2e-12,rtol=1e-10)


def test_shared_regime_can_create_cross_series_dependence_at_zero_shock_correlation():
    p=dict(correlation=0.,regime_mu=[[1.,2.],[-1.,-2.]],regime_sigma=[.01,.01],dt=.2,output='log_price')
    c=generate('markov_switching',p,length=1,horizon=2)
    assert c.reference.covariance[0,1]>.01
    assert c.reference.covariance[0,3]>.01


@pytest.mark.parametrize('kind', ['composite','merton_jump_diffusion','garch'])
def test_singular_valid_correlation_supported_outside_hidden_state_density_filter(kind):
    p={'correlation':1.}
    if kind=='merton_jump_diffusion':p.update(jump_rate=0.,output='log_return')
    c=generate(kind,p,horizon=2)
    draws=c.reference.sample(100,41)
    np.testing.assert_allclose(draws[:,:,0],draws[:,:,1],atol=1e-7)


@pytest.mark.parametrize('kind,parameters,match', [
    ('composite',{'sigma':[1.,2.,3.]},'n_targets'),
    ('composite',{'correlation':[[1.,2.],[2.,1.]]},'positive'),
    ('composite',{'correlation':[[1.,.1],[.2,1.]]},'symmetric'),
    ('composite',{'jump_rate':[1.,2.]},'scalar'),
    ('composite',{'jump_rate':1.},'random_walk'),
    ('composite',{'changes':{'period':[12.,0.]},'change_point':5},'positive'),
    ('garch',{'alpha':[.5,.1],'beta':[.6,.8]},'alpha'),
    ('garch',{'moment_samples':1},'integer'),
    ('garch',{'moment_seed':True},'integer'),
    ('garch',{'output':'price'},'output'),
    ('markov_switching',{'regime_sigma':[0.,.2]},'positive'),
    ('markov_switching',{'correlation':1.},'definite'),
    ('markov_switching',{'regime_mu':[[.1,.2,.3],[.1,.2,.3]]},'shape'),
    ('markov_switching',{'regime_correlation':[[1.,.3],[.3,1.]]},'shape'),
    ('markov_switching',{'regime_jump_rate':[[1.,2.],[3.,4.]]},'per regime'),
    ('merton_jump_diffusion',{'sigma':[-.2,.3]},'>='),
    ('merton_jump_diffusion',{'jump_correlation':[[1.,np.nan],[np.nan,1.]]},'symmetric'),
    ('merton_jump_diffusion',{'initial_price':[0.,3.]},'positive'),
])
def test_invalid_joint_parameters_reject_explicitly(kind,parameters,match):
    with pytest.raises(ValueError,match=match):generate(kind,parameters)


def test_joint_dependence_mutation_keeps_marginals_and_is_detected():
    c=generate('merton_jump_diffusion',{'correlation':.95,'jump_correlation':.95,'dt':.2,'output':'log_return'},horizon=3)
    clean=c.reference.forecast(1000,91)
    shuffled=clean.samples.copy();permutation=np.random.default_rng(18).permutation(len(shuffled))
    shuffled[:,:,1]=shuffled[permutation,:,1]
    np.testing.assert_array_equal(np.sort(shuffled,axis=0),np.sort(clean.samples,axis=0))
    np.testing.assert_array_equal(np.cov(shuffled[:,:,1],rowvar=False),np.cov(clean.samples[permutation,:,1],rowvar=False))
    bad=covariance_dependence_check(c.reference,shuffled,n_calibration=39,seed=741)
    assert bad.status=='FAIL'
    assert bad.evidence['p_value']<=.05
    # False alarms are possible under the null; require validity and repeatability.
    good=covariance_dependence_check(c.reference,clean.samples,n_calibration=39,seed=741)
    assert good.status in ('PASS','FAIL')
    assert good.to_dict()==covariance_dependence_check(c.reference,clean.samples,n_calibration=39,seed=741).to_dict()


def test_garch_exact_sampler_calibrates_fixed_numerical_covariance_target():
    c=generate('garch',{'correlation':.6},horizon=3)
    forecast=c.reference.forecast(64,71)
    result=evaluate_forecast(c,forecast,check_dependence=True,n_calibration=19)
    check=next(d for d in result.diagnostics if d.name=='reference_covariance')
    assert check.status in ('PASS','FAIL')
    assert 'fixed_independently_estimated' in check.evidence['covariance_target']
    assert check.evidence['reference']['covariance_kind']=='monte_carlo'
    assert all(np.isfinite(v) for v in result.metrics.values())


def test_joint_jump_regimes_are_explicitly_approximate_and_guarded():
    c=generate('markov_switching',{'regime_jump_rate':[.4,2.], 'correlation':.5})
    assert isinstance(c.reference,SimulationReference)
    assert c.reference.sampler_kind=='approximate_conditional'
    check=covariance_dependence_check(c.reference,c.reference.sample(64,71))
    assert check.status=='UNSUPPORTED'
    assert 'p_value' not in check.evidence
    assert 'not_a_posterior_error_bound' in c.manifest['conditioning']['filter_note']


def test_joint_fresh_and_accumulated_residuals_share_innovations_at_fixed_seed():
    fresh = generate('composite', {'residual': 'white_noise', 'correlation': .6}, length=20)
    walk = generate('composite', {'residual': 'random_walk', 'correlation': .6}, length=20)
    fresh_values = np.concatenate([fresh.task.context.values, fresh.observed.values])
    walk_values = np.concatenate([walk.task.context.values, walk.observed.values])
    np.testing.assert_allclose(np.diff(walk_values, axis=0), fresh_values[1:], atol=1e-14)


@pytest.mark.parametrize('correlation', [0., .7])
def test_joint_garch_uses_canonical_simulation_kind_and_covariance_metadata(correlation):
    c = generate('garch', {'correlation': correlation, 'moment_samples': 128}, horizon=3)
    assert c.reference.kind == c.manifest['reference_kind'] == 'analytic_moments_simulation'
    assert c.manifest['conditioning'] == c.reference.conditioning
    assert 'variance_kind' not in c.reference.conditioning
    assert 'covariance_method' not in c.reference.conditioning
    if correlation:
        assert c.reference.conditioning['covariance_kind'] == 'monte_carlo'
        assert c.reference.conditioning['mean_kind'] == 'analytic'
    else:
        assert c.reference.moments_kind == 'analytic'
        assert 'moment_samples' not in c.reference.capabilities


@pytest.mark.parametrize('kind', KINDS)
def test_joint_entry_points_share_manifest_keyword_vocabulary_and_resolved_builder(kind, monkeypatch):
    import timegym._process_common as common
    import timegym.joint_processes as joint
    from timegym.processes import resolve_process
    arguments = dict(kind=kind, parameters={'correlation': .5}, context_length=12,
                     horizon=2, n_targets=2, seed=9)
    resolved = resolve_process(**arguments)

    def unexpected_validation(*args, **kwargs):
        raise AssertionError('resolved builder must not validate parameters again')

    monkeypatch.setattr(common, 'resolve_process', unexpected_validation)
    with pytest.warns(DeprecationWarning, match='generate_joint_process is deprecated'):
        direct = joint.generate_joint_process(**{**arguments, 'parameters': resolved})
    expected = generate(kind, arguments['parameters'], length=12, horizon=2, seed=9)
    assert direct.manifest == expected.manifest
    np.testing.assert_array_equal(direct.reference.sample(8, 2), expected.reference.sample(8, 2))
