# API reference

The top-level package keeps the main workflow small. Scores, checks, preprocessing
and other helpers are public through their modules; for example,
`from timegym.metrics import mase` and
`from timegym.checks import check_shift_equivariance`.
“Public” means supported in this release; consult the changelog when upgrading.
Deprecated names still work in 1.x, warn with what to use instead and leave in
TimeGym 2.0; the changelog lists them and they are not listed here.
Schema constants describe stored formats independently of the package version.
Underscore-prefixed names and modules, including `_internal` and
`_process_common`, are private and have no compatibility guarantee.

## Public surface

| Name | Import path | Stability | Page |
| --- | --- | --- | --- |
| `ForecastTask`, `Forecast`, `Reference`, `SimulationReference`, `Case`, `Scenario` | `timegym` | Public | [Tasks and forecasts](api/tasks-and-forecasts.md); [Scenarios and references](api/scenarios.md); [Composed and joint processes](api/processes.md) |
| `Pipeline`, `NaiveForecaster`, `LastValueForecaster`, `GaussianAR1Forecaster` | `timegym` | Public | [Pipeline and reports](api/pipeline.md); [Models and reductions](api/models.md) |
| `ForecastRun`, `EvaluatedCase`, `TestReport`, `DiagnosticResult`, `CaseResult` | `timegym` | Public | [Pipeline and reports](api/pipeline.md) |
| `Scorecard`, `PATTERNS`, `GridCell`, `scenario_grid` | `timegym` | Public | [Scorecards and the scenario grid](api/scorecard.md) |
| `verdict`, `VERDICTS` | `timegym.scorecard` | Public | [Scorecards and the scenario grid](api/scorecard.md) |
| `Pattern`, `noise_levels`, `NOISE_LOW`, `NOISE_HIGH` | `timegym.grid` | Public | [Scorecards and the scenario grid](api/scorecard.md) |
| `DiagnosticStatus`, `DiagnosticCategory`, `__version__` | `timegym` | Public | [Pipeline and reports](api/pipeline.md); [Diagnostics catalogue](api/diagnostics.md#diagnostics-catalogue) |
| `DEFAULT_LEVELS` | `timegym.contracts` | Public | [Scenarios and references](api/scenarios.md) |
| `mse`, `mae`, `mase`, `mean_reference_mse`, `pinball`, `crps`, `energy_score`, `variogram_score`, `UndefinedMetricError` | `timegym.metrics` | Public | [Scores and diagnostics](api/diagnostics.md) |
| `evaluate_forecast`, `covariance_dependence_check` | `timegym.diagnostics` | Public | [Scores and diagnostics](api/diagnostics.md) |
| `check_shift_equivariance` | `timegym.checks` | Public; opt-in model property | [Scores and diagnostics](api/diagnostics.md) |
| `SklearnForecaster` | `timegym.models` | Public; sklearn loaded when needed | [Models and reductions](api/models.md) |
| `REDUCTIONS` | `timegym.models` | Public | [Models and reductions](api/models.md) |
| `MUTATIONS`, `mutate_forecast` | `timegym.mutations` | Public controlled fixtures | [Fault fixtures](api/fault-fixtures.md) |
| `plot_forecasts`, `plot_errors` | `timegym.plotting` | Public; Matplotlib imported when drawing | [Pipeline and reports](api/pipeline.md) |
| `wilson_interval`, `paired_t_interval` | `timegym.statistics` | Public; independent cases | [Scores and diagnostics](api/diagnostics.md) |
| `default_scenarios` | `timegym.scenarios` | Public | [Scenarios and references](api/scenarios.md) |
| `differencing` | `timegym.target` | Public | [Data preparation](api/data-preparation.md) |
| `LogTransform`, `BoxCoxTransform`, `DifferenceTransform`, `ComposedTransform` | `timegym.target` | Public; replay rebuilds bundled transforms only | [Data preparation](api/data-preparation.md) |
| `CORE_KINDS`, `scenario_label`, `MANIFEST_SCHEMA_VERSION` | `timegym.scenarios` | Public; manifest schema versioned separately | [Scenarios and references](api/scenarios.md); [Composed and joint processes](api/processes.md); [Pipeline and reports](api/pipeline.md) |
| `PROCESS_KINDS`, `resolve_process` | `timegym.processes` | Public low-level generators | [Scenarios and references](api/scenarios.md); [Composed and joint processes](api/processes.md) |
| `PIPELINE_CONFIG_SCHEMA_VERSION` | `timegym.provenance` | Public; config schema versioned separately | [Pipeline and reports](api/pipeline.md) |
| `REPORT_SCHEMA_VERSION` | `timegym.reporting` | Public; report schema versioned separately | [Pipeline and reports](api/pipeline.md) |
