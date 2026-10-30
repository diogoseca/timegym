# Original notebook collection

These are the ten original development notebooks and the original Quick Start,
moved into the unified notebook collection with their contents and saved outputs
preserved. The [SHA-256 inventory](inventory.json) records their paths before the
move and their preserved bytes. The Quick Start already contained a v2 notice
when this inventory was created.

The notebooks show TimeGym's original notebook-first development process:
concepts, implementation cells, experiments, and figures lived together.
`%%runwrite` executed implementation cells and wrote or appended them to package
modules. The active [source notebooks](../README.md) continue that approach with
explicit nbdev exports of complete modules. Their figures retain the original
discrete observations, forecast crosses, component views, and scenario matrices.

| Original notebook | Content and current connection |
|---|---|
| [0 · Synthetic Data](0.%20Synthetic%20Data.ipynb) | Rich generator and component exploration; active [01](../01_synthetic_data.ipynb) implements the supported v2 families and conditional references |
| [1 · Real Data](1.%20Real%20Data.ipynb) | Stock helper and proposed datasets; active [02](../02_real_data.ipynb) and [24](../24_real_data_forecasts.ipynb) cover current data examples |
| [2 · Smoothing](2.%20Smoothing.ipynb) | Moving averages, with Kalman/GP ideas; active [03](../03_smoothing.ipynb) covers implemented smoothing |
| [3 · Semi-Synthetic Data](3.%20Semi-Synthetic%20Data.ipynb) | Rationale and proposed integrations; this remains a research direction |
| [4 · Feature Engineering](4.%20Feature%20Engineering.ipynb) | Lag, calendar, and feature-library exploration; active [04](../04_feature_engineering.ipynb) covers supported helpers |
| [5 · Target Engineering](5.%20Target%20Engineering.ipynb) | Differencing and proposed transformations; active [05](../05_target_engineering.ipynb) covers implemented differencing, and [20](../20_univariate_uncertainty.ipynb)/[22](../22_multivariate_uncertainty.ipynb) develop uncertainty examples |
| [6 · Forecasting Strategy](6.%20Forecasting%20Strategy.ipynb) | Target construction and proposed strategy comparisons; active [06](../06_forecasting_models.ipynb)/[07](../07_pipeline.ipynb) use the supported recursive adapter |
| [7 · Pipeline Testing](7.%20Pipeline%20Testing.ipynb) | Prototype implementation, scenario matrices, and probabilistic forecasting notes; active [00](../00_quick_start.ipynb), [07](../07_pipeline.ipynb), and [09](../09_scores_and_diagnostics.ipynb) develop those ideas |
| [8 · Feature Testing](8.%20Feature%20Testing.ipynb) | LIME/SHAP exploration and unfinished examples; no general feature-attribution API is claimed |
| [9 · Optimization](9.%20Optimization.ipynb) | Proposed Optuna integration; automatic pipeline optimization remains unimplemented |
| [Quick Start (2021)](Quick%20Start%20%282021%29.ipynb) | The original concise pipeline demonstration and forecast/error plots |

These files retain their historical APIs, environment assumptions, and relative
paths. The current notebook checker excludes them; it does not run their source
generation cells. Use the [active collection](../README.md) for current package
development and execution. Not every historical idea or TODO has been ported.
