# API reference

The top-level package keeps the main workflow small. Scores, checks, preprocessing
and other helpers are public through their modules; for example,
`from timegym.metrics import mase` and
`from timegym.checks import check_shift_equivariance`.
The import column names the module to import each listed name from.
The page column links to the pages that document each row.
“Public” means supported in this release; consult the changelog when upgrading.
Schema constants describe stored formats independently of the package version.
Underscore-prefixed names and modules, including `_internal` and
`_process_common`, are private and have no compatibility guarantee.

## Public surface

| Name | Import path | Stability | Page |
| --- | --- | --- | --- |
| `ForecastTask`, `Forecast`, `Reference`, `SimulationReference`, `Case`, `Scenario` | `timegym` | Public | [Tasks and forecasts](api/tasks-and-forecasts.md); [Scenarios and references](api/scenarios.md); [Composed and joint processes](api/processes.md) |
| `Pipeline`, `NaiveForecaster`, `GaussianAR1Forecaster` | `timegym` | Public | [Pipeline and reports](api/pipeline.md); [Models and reductions](api/models.md) |
| `ForecastRun`, `EvaluatedCase`, `TestReport`, `DiagnosticResult`, `CaseResult` | `timegym` | Public | [Pipeline and reports](api/pipeline.md) |
| `DiagnosticStatus`, `DiagnosticCategory`, `__version__` | `timegym` | Public | [Pipeline and reports](api/pipeline.md); [Diagnostics catalogue](api/diagnostics.md#diagnostics-catalogue) |
| `DEFAULT_LEVELS` | `timegym.contracts` | Public | [Scenarios and references](api/scenarios.md) |
| `mse`, `mae`, `mase`, `mean_reference_mse`, `pinball`, `crps`, `empirical_crps`, `energy_score`, `variogram_score`, `UndefinedMetricError` | `timegym.metrics` | Public; `empirical_crps` aliases `crps` | [Scores and diagnostics](api/diagnostics.md) |
| `evaluate_forecast`, `covariance_dependence_check`, `exception_evidence` | `timegym.diagnostics` | Public | [Scores and diagnostics](api/diagnostics.md) |
| `check_shift_equivariance` | `timegym.checks` | Public; opt-in model property | [Scores and diagnostics](api/diagnostics.md) |
| `SklearnForecaster`, `fresh_model`, `is_sklearn_regressor` | `timegym.models` | Public; sklearn loaded when needed | [Models and reductions](api/models.md) |
| `REDUCTIONS` | `timegym.models` | Public | [Models and reductions](api/models.md) |
| `MUTATIONS`, `mutate_forecast` | `timegym.mutations` | Public controlled fixtures | [Fault fixtures](api/fault-fixtures.md) |
| `plot_forecasts`, `plot_errors` | `timegym.plotting` | Public; requires plots extra | [Pipeline and reports](api/pipeline.md) |
| `wilson_interval`, `paired_t_interval` | `timegym.statistics` | Public; independent cases | [Scores and diagnostics](api/diagnostics.md) |
| `default_scenarios` | `timegym.data.synthetic` | Public | [Scenarios and references](api/scenarios.md) |
| `get_stock` | `timegym.data.real` | Public; provider data may change | [Market data](api/market-data.md) |
| `add_lagged_target`, `cyclical_encoding`, `add_time_features`, `add_tsfel_features` | `timegym.features` | Public; TSFEL requires features extra | [Data preparation](api/data-preparation.md) |
| `multioutput_target`, `onestep_target`, `step_ahead_target`, `fractional_weights`, `fractional_differencing`, `differencing` | `timegym.target` | Public | [Data preparation](api/data-preparation.md) |
| `LogTransform`, `BoxCoxTransform`, `DifferenceTransform`, `ComposedTransform`, `transform_config`, `transform_from_config` | `timegym.target` | Public; replay rebuilds bundled transforms only | [Data preparation](api/data-preparation.md) |
| `simple_moving_average`, `exponential_moving_average` | `timegym.smoothing` | Public | [Data preparation](api/data-preparation.md) |
| `CORE_KINDS`, `scenario_label`, `MANIFEST_SCHEMA_VERSION` | `timegym.scenarios` | Public; manifest schema versioned separately | [Scenarios and references](api/scenarios.md); [Composed and joint processes](api/processes.md); [Pipeline and reports](api/pipeline.md) |
| `PROCESS_KINDS`, `resolve_process`, `process_label`, `generate_process` | `timegym.processes` | Public low-level generators | [Scenarios and references](api/scenarios.md); [Composed and joint processes](api/processes.md) |
| `generate_joint_process` | `timegym.joint_processes` | Public; takes `resolve_process` output | [Scenarios and references](api/scenarios.md); [Composed and joint processes](api/processes.md) |
| `replay_case` | `timegym.pipeline` | Public; allowlisted models only | [Pipeline and reports](api/pipeline.md) |
| `PIPELINE_CONFIG_SCHEMA_VERSION`, `pipeline_config`, `pipeline_from_config`, `runtime_provenance`, `diagnostic_settings` | `timegym.provenance` | Public; config schema versioned separately | [Pipeline and reports](api/pipeline.md) |
| `REPORT_SCHEMA_VERSION` | `timegym.reporting` | Public; report schema versioned separately | [Pipeline and reports](api/pipeline.md) |
