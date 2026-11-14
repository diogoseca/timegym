# Scenario equations

This page gives the generating equations of the four composed and financial
scenario kinds (`composite`, `merton_jump_diffusion`, `garch` and
`markov_switching`), the answer key each one supplies, how its sampler draws
futures, and where the answer key stops being exact. [Synthetic
scenarios](stress-cases.md) explains the same scenarios in words. Throughout,
`T` is the last observed index, `h = 1, ..., H` counts steps ahead, `I[T]` is
the observed history, and `E[. | I[T]]` and `Cov(., . | I[T])` are the
conditional mean and covariance given it.

## Composite

The `composite` scenario adds a deterministic component to one residual
process:

```text
g[t] = b + c*t + A*sin(2*pi*t/p + phi)
Y[t] = g[t] + r[t]
```

The residual `r[t]` is either fresh Gaussian noise or an accumulated-shock
process, optionally with compound-Poisson jumps:

```text
fresh noise:          r[t] = sigma[t]*eps[t]
accumulated shocks:   r[t] = r[t-1] + sigma[t]*eps[t] + J[1] + ... + J[N[t]]
                      eps[t] ~ Normal(0, 1),  N[t] ~ Poisson(lambda),  J[i] ~ Normal(a, bJ^2)
```

A declared change replaces `b`, `c`, `A`, `p`, `phi` or `sigma` from an
explicit index on. The mean uses absolute `t` before and after the change, so
changing the slope need not keep the level continuous. These are additive
level constructions, not multiplicative price models. A change of this kind is
a change of the generating law in the sense of Gama et al. (2014), as distinct
from process drift, an expected increment.

**Answer key.** With fresh noise the conditional mean is `g[T+h]` and the
covariance is diagonal with entries `sigma[T+h]^2`. With accumulated shocks,
`r[T]` is read from the observed history and

```text
E[Y[T+h] | I[T]]                = g[T+h] + r[T] + h*lambda*a
Cov(Y[T+h], Y[T+k] | I[T])      = sum over j = 1, ..., min(h, k) of  sigma[T+j]^2 + lambda*(bJ^2 + a^2)
```

The evaluator knows the change schedule; an ordinary fitted model does not.
With jumps, the sampler draws each future by accumulating Gaussian shocks and
compound-Poisson increments from `r[T]`.

**Several targets.** With a vector of scales `s[t]` and a correlation matrix
`R`, the innovation covariance is `diag(s[t]) R diag(s[t])`. An optional
shared Poisson clock adds vector marks `J ~ Normal(a, B)`, so the increment
covariance is

```text
Q[t] = diag(s[t]) R diag(s[t]) + lambda*(B + a a')
```

Fresh residuals have block-diagonal time covariance; accumulated residuals sum
`Q[t]` through the earlier of the two forecast horizons. Scheduled changes may
affect different targets differently. These joint moments are analytical.

## Merton jump diffusion

Independent one-step log returns are

```text
R = (mu - sigma^2/2 - lambda*kappa)*dt + sigma*sqrt(dt)*Z + J[1] + ... + J[N]
Z ~ Normal(0, 1),  N ~ Poisson(lambda*dt),  J[i] ~ Normal(a, bJ^2),  kappa = exp(a + bJ^2/2) - 1
```

This is the construction of Merton (1976) with a physical expected-price-growth
convention, not a risk-neutral pricing calibration. With `lambda = 0` it is
geometric Brownian motion. The target is the price, the log price or the log
return, which have different conditional means.

**Answer key.** Log returns are independent across steps, with

```text
mean       (mu - sigma^2/2 - lambda*kappa + lambda*a)*dt
variance   (sigma^2 + lambda*(bJ^2 + a^2))*dt
```

and log prices accumulate them from the last observed log price. For prices,
the cumulant `K(u) = log E[exp(u*R)]` gives

```text
m[h] = E[S[T+h] | I[T]] = S[T]*exp(h*K(1))
Cov(S[T+h], S[T+k] | I[T]) = m[h]*m[k]*(exp(min(h, k)*(K(2) - 2*K(1))) - 1)
```

Exponentiating a conditional log-price mean generally gives the wrong price
mean. The sampler draws independent returns from the same law.

**Several targets.** Diffusions are correlated and one Poisson arrival process
is shared, with a separate correlation matrix for the Gaussian log-jump marks.
Let `K(u) = log E[exp(u'R)]` for the vector return `R` of one observation step,
`e[i]` the `i`-th coordinate vector and `m[h,i]` the conditional price mean of
target `i`. Then

```text
Cov(S[T+h,i], S[T+k,j] | I[T])
    = m[h,i]*m[k,j]*(exp(min(h, k)*(K(e[i] + e[j]) - K(e[i]) - K(e[j]))) - 1)
```

Zero diffusion correlation does not remove the dependence that a shared jump
clock creates. The construction has common jump arrivals only; it does not add
an independent jump process for each target.

## GARCH

Gaussian innovations follow

```text
r[t]   = mu + sqrt(v[t])*z[t],   z[t] ~ Normal(0, 1)
v[t+1] = omega + alpha*(r[t] - mu)^2 + beta*v[t]
with omega > 0, alpha >= 0, beta >= 0, alpha + beta < 1
```

This is the GARCH(1,1) generalization of Bollerslev (1986) of the ARCH process
of Engle (1982). The scenario supplies log returns or log prices, not price
moments.

**Answer key.** The next variance `v[T+1]` is reconstructed from the observed
returns, and later expected variances follow

```text
E[v[T+h+1] | I[T]] = omega + (alpha + beta)*E[v[T+h] | I[T]]
```

Log returns have mean `mu`, these variances, and no covariance across steps;
log prices accumulate them. The sampler is exact: each simulated future updates
its own variance recursively. Replacing that recursion by independent Gaussian
draws with the expected variances loses part of the joint law, and multi-step
distributions need not be Gaussian.

**Several targets.** The joint form is constant-conditional-correlation GARCH
(Bollerslev, 1990), with one GARCH(1,1) recursion per target:

```text
eps[t] = diag(sqrt(v[t])) z[t],   z[t] ~ Normal(0, R)
```

Means and marginal variances are analytical and covariances across time
vanish, but the same-time covariance between series `i` and `j` contains

```text
R[i,j]*E[sqrt(v[T+h,i]*v[T+h,j]) | I[T]]
```

which is generally not `R[i,j]*sqrt(E[v[T+h,i]]*E[v[T+h,j]])`. An independent
Monte Carlo ensemble (16,384 draws by default, with recorded count and seed)
estimates it, rescaled to the analytical diagonal so that the marginal
variances stay exact and the matrix stays positive semidefinite. The first
step, zero correlation and deterministic variances use analytical covariance,
and log levels accumulate the return blocks. These moments are labelled
`analytic_mean_monte_carlo_covariance`; the conditional sampler stays exact.

## Markov switching

A finite hidden Markov chain `S[t]` with transition matrix `P` sets the
parameters `(mu[s], sigma[s], lambda[s], a[s], bJ[s])` of a Merton-type return
in each regime `s`; with `lambda[s] = 0` the regime is Gaussian. The regime is
never observed. Following Hamilton (1989), the answer key filters it from the
observed returns `y[t]`, with `f[j]` the return density of regime `j`:

```text
pi[t|t](j)  proportional to  f[j](y[t]) * sum over i of pi[t-1|t-1](i)*P[i,j]
regime probabilities h steps ahead:  pi[T|T] P^h
```

It never conditions on the simulator's actual terminal regime.

**Answer key.** Means and covariances of log returns and log prices mix the
regime moments over the propagated regime probabilities; price moments follow
from the law of total covariance. The sampler draws a starting regime from
`pi[T|T]` and then regimes and returns forward. With Gaussian emissions the
filter, and so the sampler, are exact under the declared law. With jumps, the
emission likelihood `f[j]` is a truncated Poisson mixture: its omitted
probability mass is recorded, but it is not a bound on the posterior error,
and the reference is marked approximate.

**Several targets.** The targets share one hidden chain. Regime parameters may
be arrays with one row per regime and one column per target, and the
correlation within a regime may differ by state. The forward filter uses the
complete observed return vector, and analytical recursions propagate the
filtered distribution and all moments across targets and horizons. With
shared jumps a truncated multivariate Gaussian mixture approximates each
emission likelihood. Multivariate filtering needs positive diffusion scales
and positive-definite state correlations; the other joint families also accept
singular correlations.

## Simulation references and their limits

For these four kinds, `SimulationReference` keeps analytical means and either
analytical or explicitly estimated covariance, and draws joint futures from an
explicit sampler. Marginal quantiles are reproducible Monte Carlo estimates
with recorded sample count and seed, not Gaussian quantiles inferred from two
moments. The stated limits are:

- With jumps, the regime filter is approximate, and the covariance rank check
  returns `UNSUPPORTED` for that sampler. Output checks and descriptive scores
  stay available, with no exact test guarantee.
- The CCC-GARCH cross-covariance is a Monte Carlo estimate. Rank calibration
  holds it fixed and uses independent forecast and null ensembles, so its
  error can affect the power of the check but not its validity. The check
  still cannot detect every form of higher-order dependence.
- The joint forms implement shared jump clocks and shared hidden chains only,
  not idiosyncratic jumps, cross-lag or dynamic-correlation systems.

## References

- Bollerslev, T. (1986). Generalized autoregressive conditional
  heteroskedasticity. *Journal of Econometrics*, 31(3), 307-327.
  [doi:10.1016/0304-4076(86)90063-1](https://doi.org/10.1016/0304-4076(86)90063-1)
- Bollerslev, T. (1990). Modelling the coherence in short-run nominal exchange
  rates: a multivariate generalized ARCH model. *The Review of Economics and
  Statistics*, 72(3), 498-505.
  [doi:10.2307/2109358](https://doi.org/10.2307/2109358)
- Engle, R. F. (1982). Autoregressive conditional heteroscedasticity with
  estimates of the variance of United Kingdom inflation. *Econometrica*,
  50(4), 987-1007. [doi:10.2307/1912773](https://doi.org/10.2307/1912773)
- Gama, J., Žliobaitė, I., Bifet, A., Pechenizkiy, M., and Bouchachia, A.
  (2014). A survey on concept drift adaptation. *ACM Computing Surveys*,
  46(4), 44:1-44:37. [doi:10.1145/2523813](https://doi.org/10.1145/2523813)
- Hamilton, J. D. (1989). A new approach to the economic analysis of
  nonstationary time series and the business cycle. *Econometrica*, 57(2),
  357-384. [doi:10.2307/1912559](https://doi.org/10.2307/1912559)
- Merton, R. C. (1976). Option pricing when underlying stock returns are
  discontinuous. *Journal of Financial Economics*, 3(1-2), 125-144.
  [doi:10.1016/0304-405X(76)90022-2](https://doi.org/10.1016/0304-405X(76)90022-2)
