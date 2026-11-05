---
hide:
  - toc
---

# Capability map

Which behaviours of a forecaster does TimeGym exercise, and with what kind of
evidence? This page crosses the two. Rows are capabilities a model may need;
columns are the four kinds of evidence from
[How TimeGym decides](how-timegym-decides.md). The layout follows CheckList's
capability by test-type matrix
([Ribeiro et al. 2020](https://doi.org/10.18653/v1/2020.acl-main.442)).

Each entry in a cell names a scenario family, a diagnostic and the tutorial that
shows the pair, by number. An empty cell is a gap: no tutorial shows that pair
yet. A gap does not always mean the check is skipped. Forecast contracts, for
example, run on every case of every family.

## The four kinds of evidence

| Column | Oracle type | Diagnostics | When it runs |
| --- | --- | --- | --- |
| Forecast contract | Partial oracle | `forecast_contract`, `contract_<code>` | On every case, before scoring |
| Reference score | Simplified data with known results | `point_accuracy`, `quantile_accuracy`, `sample_scores`, `metric_<name>`, `interval_coverage` | On every case that meets its contract; descriptive, never `PASS` or `FAIL` |
| Declared relation | Known properties | `shift_equivariance` | When called with `check_shift_equivariance`, for a model that declares the property |
| Statistical check | Statistical metamorphic test | `reference_covariance` | With `check_dependence=True`, or called with `covariance_dependence_check` |

The [diagnostics catalogue](../api/diagnostics.md#diagnostics-catalogue) gives
each diagnostic's statuses, tolerance and evidence.

## The map

Core families are documented in [Scenarios and references](../api/scenarios.md),
composed and financial families in
[Composed and joint processes](../api/processes.md).

<div class="capability-map" markdown>

| Capability and scenario families | Forecast contract | Reference score | Declared relation | Statistical check |
| --- | --- | --- | --- | --- |
| **Level and trend**<br>`white_noise`, `trend`, `composite` | `white_noise` · [`contract_index_mismatch`](../api/diagnostics.md#contract_code) · [23](../../notebooks/23_faults_repairs_and_replay.ipynb)<br>`white_noise` · [`contract_quantile_crossing`](../api/diagnostics.md#contract_code) · [23](../../notebooks/23_faults_repairs_and_replay.ipynb) | `trend` · [`point_accuracy`](../api/diagnostics.md#point_accuracy) · [25](../../notebooks/25_capability_sweeps.ipynb)<br>`trend` · [`point_accuracy`](../api/diagnostics.md#point_accuracy) · [28](../../notebooks/28_custom_model.ipynb) | `white_noise` · [`shift_equivariance`](../api/diagnostics.md#shift_equivariance) · [23](../../notebooks/23_faults_repairs_and_replay.ipynb) |  |
| **Persistence**<br>`ar1`, `random_walk`, `composite` | `ar1` · [`forecast_contract`](../api/diagnostics.md#forecast_contract) · [20](../../notebooks/20_univariate_uncertainty.ipynb) | `ar1` · [`interval_coverage`](../api/diagnostics.md#interval_coverage) · [20](../../notebooks/20_univariate_uncertainty.ipynb)<br>`random_walk` · [`point_accuracy`](../api/diagnostics.md#point_accuracy) · [25](../../notebooks/25_capability_sweeps.ipynb)<br>`ar1` · [`point_accuracy`](../api/diagnostics.md#point_accuracy) · [30](../../notebooks/30_reductions.ipynb) |  |  |
| **Seasonality**<br>`seasonal`, `composite` | `seasonal` · [`contract_index_mismatch`](../api/diagnostics.md#contract_code) · [23](../../notebooks/23_faults_repairs_and_replay.ipynb)<br>`seasonal` · [`contract_point_shape`](../api/diagnostics.md#contract_code) · [28](../../notebooks/28_custom_model.ipynb) | `composite` · [`interval_coverage`](../api/diagnostics.md#interval_coverage) · [29](../../notebooks/29_target_transforms.ipynb)<br>`seasonal` · [`point_accuracy`](../api/diagnostics.md#point_accuracy) · [30](../../notebooks/30_reductions.ipynb) |  |  |
| **Parameter changes**<br>`composite` |  | `composite` · [`point_accuracy`](../api/diagnostics.md#point_accuracy) · [26](../../notebooks/26_composed_and_financial_stresses.ipynb) |  |  |
| **Jumps**<br>`merton_jump_diffusion`, `composite`, `markov_switching` |  | `merton_jump_diffusion` · [`point_accuracy`](../api/diagnostics.md#point_accuracy) · [26](../../notebooks/26_composed_and_financial_stresses.ipynb) |  |  |
| **Volatility clustering**<br>`garch` |  | `garch` · [`point_accuracy`](../api/diagnostics.md#point_accuracy) · [26](../../notebooks/26_composed_and_financial_stresses.ipynb) |  |  |
| **Hidden regimes**<br>`markov_switching` |  | `markov_switching` · [`point_accuracy`](../api/diagnostics.md#point_accuracy) · [26](../../notebooks/26_composed_and_financial_stresses.ipynb) |  |  |
| **Dependence between targets**<br>Every family, with `n_targets` of 2 or more | `ar1` · [`contract_target_mismatch`](../api/diagnostics.md#contract_code) · [21](../../notebooks/21_multivariate_forecasting.ipynb) | `ar1` · [`point_accuracy`](../api/diagnostics.md#point_accuracy) · [21](../../notebooks/21_multivariate_forecasting.ipynb)<br>`ar1` · [`sample_scores`](../api/diagnostics.md#sample_scores) · [22](../../notebooks/22_multivariate_uncertainty.ipynb) |  | `ar1` · [`reference_covariance`](../api/diagnostics.md#reference_covariance) · [22](../../notebooks/22_multivariate_uncertainty.ipynb)<br>`composite`, `merton_jump_diffusion`, `garch`, `markov_switching` · [`reference_covariance`](../api/diagnostics.md#reference_covariance) · [27](../../notebooks/27_joint_financial_uncertainty.ipynb) |

</div>

## What the gaps mean

- **Forecast contract.** A contract does not depend on the family:
  `Forecast.validate` checks every case before it is scored. The cells show
  where a tutorial injects a contract fault, and none does so on a composed or
  financial family. The `horizon_truncation`, `nonfinite_point` and
  `sample_axis_swap` [fault fixtures](../api/fault-fixtures.md) appear in no
  tutorial.
- **Reference score.** Every row has one, but the composed and financial rows
  rest on tutorial 26 alone, which reports mean-reference MSE. Parameter changes
  appear only in its ladder of forecast plots, with the score in each panel
  title.
- **Declared relation.** Shift equivariance is the only declared relation, and
  only tutorial 23 runs it, on white noise. A relation holds only for a model
  that declares it, so some of these cells stay empty by design.
- **Statistical check.** `PASS` and `FAIL` come only from direct
  `covariance_dependence_check` calls in tutorials 22 and 27, on reference
  ensembles with and without a cross-target fault. No tutorial runs the check on
  a fitted model's samples, or against a jump, volatility or regime fault in one
  target. Tutorials 24 and 28 pass `check_dependence=True` on real data, which
  has no reference, so the check returns `UNSUPPORTED`. Regime switching with
  jumps has an approximate reference, so tutorials 26 and 27 show the check
  returning `UNSUPPORTED` there too.

Each gap is a candidate for a new tutorial section or check.

## Keeping the map current

`tests/test_docs_catalogue.py` fails when a scenario family the code defines
has no row, when the evidence table above disagrees with the catalogue's oracle
types, when a cell names a diagnostic the catalogue lacks or files it under the
wrong column, and when a linked tutorial no longer mentions its family and
diagnostic.
