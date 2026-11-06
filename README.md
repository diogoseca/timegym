# TimeGym

**Controlled verification and diagnosis for forecasting pipelines.**

[![Documentation](https://img.shields.io/badge/docs-diogoseca.github.io%2Ftimegym-black)](https://diogoseca.github.io/timegym/)
[![Checks](https://github.com/diogoseca/timegym/actions/workflows/checks.yml/badge.svg)](https://github.com/diogoseca/timegym/actions/workflows/checks.yml)
[![PyPI](https://img.shields.io/pypi/v/timegym)](https://pypi.org/project/timegym/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

A forecast can have the same numerical error and still be on the wrong dates
or the wrong targets. TimeGym's forecast contract rejects such forecasts before
anything is scored; a wrong scale shows up as a large distance from the known
reference. TimeGym generates synthetic forecasting cases whose true conditional
distribution is known, runs your model on them, checks the output against an
explicit contract, scores it against the known reference, and writes down why
anything failed.

[![Six synthetic configurations, three seeds each: observations as dots, forecasts as crosses, conditional references as thin curves](notebooks/assets/quick-start-matrix.png)](https://diogoseca.github.io/timegym/notebooks/00_quick_start/)

**[Documentation](https://diogoseca.github.io/timegym/)** ·
[Quick start](https://diogoseca.github.io/timegym/notebooks/00_quick_start/) ·
[Tutorials](https://diogoseca.github.io/timegym/docs/notebooks/) ·
[API](https://diogoseca.github.io/timegym/docs/reference/) ·
[Changelog](CHANGELOG.md)

## Install

<!-- BEGIN PYTHON SUPPORT -->
Python 3.11–3.15 (`>=3.11,<3.16`), following pandas support.
<!-- END PYTHON SUPPORT -->

```bash
pip install timegym               # core: NumPy, pandas, SciPy
pip install "timegym[plots]"      # plus Matplotlib for the figures
pip install "timegym[sklearn]"    # plus the scikit-learn adapter
```

From a clone, `pip install -e '.[dev]'` installs what the tests need;
`.[notebooks]` adds Jupyter to run the notebooks.

## Sixty seconds

```python
from timegym import NaiveForecaster, Pipeline, Scenario

pipeline = Pipeline(NaiveForecaster(strategy="last"))
report = pipeline.test(
    scenarios=[Scenario("random_walk", context_length=120, horizon=12)],
    seeds=[0, 1, 2],
)
print(report.summary)
report.assert_passed()   # raises on FAIL or ERROR, never on unsupported capabilities
report.to_json("report.json")
```

The report records each case's scenario, seed, manifest, metrics and
diagnostics (status and evidence), plus the pipeline configuration, the Python,
TimeGym, NumPy, pandas, SciPy and scikit-learn versions and a source digest, so
a failure with a built-in model can be replayed from the file alone. Sklearn
steps, model factories, custom target transforms and your own classes also need
their code, and `GaussianAR1Forecaster` needs a fixed seed.

To look at the forecasts instead of the numbers, keep the run:

```python
run = pipeline.run(seeds=[0, 1, 2])
run.plot_forecasts()
run.plot_errors(metric="mean_reference_mse")
```

## What it checks

**Contracts.** A `Forecast` holds at least one of a point array `(H, D)`,
quantiles `(H, D, Q)` and joint samples `(S, H, D)`. TimeGym checks the index
against the requested future, the target names and order, shapes, finiteness
and quantile ordering before any score is computed. A misaligned index is a
`contract_index_mismatch` `FAIL`, not a slightly worse MSE.

[![A forecast with the right values on the wrong dates: the same error, a different diagnosis](notebooks/assets/alignment-demo.png)](https://diogoseca.github.io/timegym/notebooks/23_faults_repairs_and_replay/)

**References.** Each synthetic scenario knows its conditional mean, variance,
cross-horizon covariance, quantiles and samples given the observed context.
Scores against the reference (mean-reference MSE, CRPS, energy and variogram
scores) measure distance to the truth, not to one noisy realization.

**Diagnostics with evidence.** Optional Monte Carlo covariance checks compare an
ensemble to the known law at a stated significance level; a shift-equivariance
check tests a property the model declares; eight output-fault fixtures with
paired repairs show what each check catches. Accuracy scores stay
`INCONCLUSIVE`: TimeGym sets no accuracy threshold, so apply your own rule to
the recorded metrics.

**Scenarios.** White noise, trend, seasonality, AR(1) and random walks with or
without drift; multiple correlated targets; composed trend, seasonality and
noise with scheduled changes, Merton jumps, GARCH volatility and hidden
regimes; joint shocks, jumps and regimes across targets, each with explicit
reference capabilities. The [scenario guide](docs/stress-cases.md) explains
what the composed and financial families can and cannot show.

## Bring your own model

Any object with `fit(context)` and `predict(future_index)` returning a
`Forecast` is a model. The [custom model tutorial](https://diogoseca.github.io/timegym/notebooks/28_custom_model/)
writes a nearest-analogues forecaster from scratch, tests it on synthetic
suites, then on real data with and without a known truth.

[![A nearest-analogues forecaster on white-noise, trend and seasonal scenarios, two seeds each: observations as dots, forecasts as crosses, known conditional means as lines](notebooks/assets/custom-model-knn.png)](https://diogoseca.github.io/timegym/notebooks/28_custom_model/)

scikit-learn regressors plug in through a recursive, direct or multioutput
reduction that fits its transforms on the training window only:

```python
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from timegym import Pipeline

pipeline = Pipeline(window_length=12).add(StandardScaler()).add(Ridge(alpha=1.0))
report = pipeline.test(seeds=[0])
```

Native models pass as an unfitted factory, `Pipeline(model_factory=lambda: MyModel(...))`,
or as an instance with `clone_unfitted()`; TimeGym refits from scratch for every
case and never assumes a deep copy clears fitted state.

## Tutorials

| Group | Notebook | What it shows |
|---|---|---|
| Start | [Quick start](https://diogoseca.github.io/timegym/notebooks/00_quick_start/) | A pipeline, a matrix of cases, error distributions |
| Find what breaks | [Faults, repairs and replay](https://diogoseca.github.io/timegym/notebooks/23_faults_repairs_and_replay/) | Invalid outputs, inverse scaling, replaying a saved report |
| | [Write and test a custom model](https://diogoseca.github.io/timegym/notebooks/28_custom_model/) | A forecaster from scratch, synthetic suites, real data |
| Shape the forecast | [Target transforms](https://diogoseca.github.io/timegym/notebooks/29_target_transforms/) | Log, Box-Cox and difference transforms, inverted on the forecast |
| | [Reductions](https://diogoseca.github.io/timegym/notebooks/30_reductions/) | Recursive, direct and multioutput sklearn forecasts compared |
| | [Multivariate forecasting](https://diogoseca.github.io/timegym/notebooks/21_multivariate_forecasting/) | Several targets, lag features, per-target scores |
| | [Real-data forecasts](https://diogoseca.github.io/timegym/notebooks/24_real_data_forecasts/) | Ordered origins on the AirPassengers series |
| Measure uncertainty | [Univariate uncertainty](https://diogoseca.github.io/timegym/notebooks/20_univariate_uncertainty/) | Marginal intervals, coverage, CRPS |
| | [Multivariate uncertainty](https://diogoseca.github.io/timegym/notebooks/22_multivariate_uncertainty/) | Joint paths, dependence, covariance checks |
| Stress-test | [Capability sweeps](https://diogoseca.github.io/timegym/notebooks/25_capability_sweeps/) | Five families, context and noise sweeps |
| | [Composed and financial stresses](https://diogoseca.github.io/timegym/notebooks/26_composed_and_financial_stresses/) | Jumps, GARCH, hidden regimes, one mechanism at a time |
| | [Joint financial uncertainty](https://diogoseca.github.io/timegym/notebooks/27_joint_financial_uncertainty/) | Shared shocks and regimes, CCC-GARCH, broken dependence |

The notebooks run on CPU with local data. `pip install -e '.[notebooks]'` and
`python -m jupyterlab notebooks` opens them; the
[notebook guide](notebooks/README.md) describes the folder.

## The paper

The method and two controlled studies are described in the manuscript kept in
this repository: [`paper/main.tex`](paper/main.tex), built into a
[single-file version](paper/timegym.tex) whose tables and figures are
generated by script from the recorded study results and the tutorial settings.
The studies, their frozen records
and the exact reproduction recipe are in [`experiments/`](experiments/README.md)
and [`paper/README.md`](paper/README.md). This manuscript will be posted as
version 2 of arXiv:[2105.01404](https://arxiv.org/abs/2105.01404);
[version 1](https://arxiv.org/abs/2105.01404v1), from May 2021, describes the
original 2021 prototype.

## Development

The package is written in notebooks. The source notebooks under `notebooks/`
generate every module in `timegym/`; edit the notebook, export, test:

```bash
python tools/export_notebooks.py
python tools/export_notebooks.py --check
python -m pytest
```

[Contributing](docs/contributing.md) describes the notebook-to-module cycle;
[releasing](docs/release.md) describes the versioning rule, the Python
compatibility policy, and how a labeled merge becomes a tagged PyPI release.

## Scope

TimeGym does not locate the root cause of a bug, repair models, detect general
leakage, infer conditional references for arbitrary generators, or handle
panels, hierarchies, missing or irregular observations. Unsupported checks are
reported as unsupported, never silently skipped.

## Cite

Cite the software by version; GitHub's "Cite this repository" reads the same
metadata from [CITATION.cff](CITATION.cff). When a release is archived on
Zenodo, cite that release's DOI.

<!-- BEGIN GENERATED CITATION -->

```bibtex
@software{seca2026timegym,
  author = {Seca, Diogo},
  title = {{TimeGym: verification and diagnosis for forecasting pipelines}},
  year = {2026},
  version = {1.1.0},
  url = {https://github.com/diogoseca/timegym},
}
```

Seca, D. (2026). *TimeGym: verification and diagnosis for forecasting pipelines*
(Version 1.1.0) [Computer software]. https://github.com/diogoseca/timegym

The 2021 preprint (arXiv version 1, the original prototype):

```bibtex
@article{seca2021timegym,
  author = {Seca, Diogo},
  title = {{TimeGym: Debugging for Time Series Modeling in Python}},
  year = {2021},
  doi = {10.48550/arXiv.2105.01404},
  url = {https://arxiv.org/abs/2105.01404v1},
}
```

<!-- END GENERATED CITATION -->

MIT license. Datasets and cited works keep their own attribution; see
[data provenance](experiments/data/README.md).
