# Claim and evidence ledger

Evidence reviewed 28 October 2026; citation identifiers and the financial-source entries refreshed 30 October 2026; testing, oracle, and scoring-rule sources added from retrieved full texts on 4 November 2026. The contribution is an executable workflow that combines established testing and forecasting principles with inspectable conditional references. A claim becomes an empirical result only when its generated artifacts exist and the manuscript reports their actual output. Passing tests supports the tested cases; it does not prove correctness for all inputs.

| Claim | Evidence and boundary | Manuscript treatment |
| --- | --- | --- |
| Forecast output contracts separate structural violations from model capability. | `timegym/contracts.py`, `tests/test_contracts.py`, and `tests/test_validation_boundaries.py`: explicit horizon/target/sample/quantile axes, numeric and index validation. Capability failures are reported separately. This does not infer the unique source-code cause. | Implemented method; scope shown through controlled examples. |
| Synthetic references match the requested functional and information set. | `timegym/scenarios.py` and `tests/test_references.py`: analytic conditional Gaussian means, marginal quantiles, covariance, simulation checks, and explicit boundary cases. The simulator and reference share assumptions and some code; simulation agreement is not independent proof. Parameters are known to the evaluator, not generally to learned forecasters. | Mathematical specification plus implementation validation. |
| Selected faults can be detected and their known repairs checked. | The output-corruption pilot remains in `experiments/results/results.json`; implementation and covariance studies use separate `experiments/protocol-v2.json` and `experiments/results-v2/results.json`. Clean, faulty, and repaired runs share cases; calibration cases are separate. Misses and false alarms remain in the record. | Two controlled, developer-selected studies; no estimate of natural fault prevalence or coverage. |
| Different relations expose different implementation defects. | `experiments/model_faults.py`, `tests/test_research_supplement.py`, and `experiments/results-v2/model-ablation.csv`: all four internal regressions preserve the output contract; scalar translation detects one branch, per-target translation two, and independent NumPy agreement all four on the specified cases. The last two comparators are experiment-only. | Complementary evidence, not a universal ranking or a claim that the public package automatically identifies every defect. |
| A capability failure does not establish a software fault. | Last-value, seasonal-naive, and fitted Gaussian AR(1) baselines run on the same mechanisms in `experiments/results/results.json`. Misspecified but well-formed predictions remain possible. | Explicit interpretation of capability scores. |
| Marginal sample scores cannot identify every dependence defect. | Coordinate-wise sample permutations preserve empirical marginal measures. The manuscript proves the finite-sample identity; `tests/test_metrics.py` and paired study records check its numerical consequences. This identity is elementary, not a new probability theorem. | Algebraic negative control plus finite-sample experiment. |
| A repair can restore the intended algorithm without improving forecast loss. | `experiments/results-v2/model-repair-losses.csv`: the corrected stale-state branch worsens average realized MSE for the random-walk condition with drift while restoring implementation agreement. The generated manuscript retains the paired effect and interval, not only the improved cases. | Verification and predictive model selection are distinct objectives. Repairs are specified, not discovered. |
| Marginal and family-wise error rates answer different questions. | `experiments/results-v2/covariance-conditions.csv` and suite records: clean, partial-shuffle, and reduced-spread variants retain raw and Holm decisions. Factorized white noise under permutation remains a true null. The declared family contains four checks within one suite and variant. | Theoretical control relies on valid null p-values; observed counts and Wilson intervals do not prove calibration. Power losses after adjustment remain visible. |
| Reports and experiments support replay and inspection. | Scenario specifications, model identifiers, dimensions, seeds, settings, and outcomes are recorded. `tests/test_replay.py` checks scenario reconstruction; experiment manifests record the code/environment and generated artifacts. Reproducing arbitrary external training requires that model's own source and environment. | Reproducibility of the specified case and controlled studies; no blanket determinism claim. |
| The 2021 manuscript is corrected and extended. | Original source available from [arXiv:2105.01404v1](https://arxiv.org/abs/2105.01404v1); current manuscript gives corrected conditioning, metrics, uncertainty and scope. | Explicit historical relationship. |

## Claims excluded from this version

- First synthetic time-series benchmark, first forecasting unit tests, or first use of mutation testing for forecasting.
- Universal fault localization, automatic causal explanation, or automatic code repair.
- Superiority over SynTSBench, SimpleTimeBench, sktime, or other tools without a direct controlled comparison.
- Empirical ability to detect leakage or unavailable future covariates without a dedicated implemented information-flow experiment.
- Validity of every score for every forecast functional. In particular, mean-reference discrepancy is not a general loss for median forecasts.
- Synthetic-to-real generalization, external validity, production reliability, or guaranteed absence of faults.
- Elimination of multiplicity or adaptive-selection problems by drawing fresh seeds.
- A calibrated family-wise false-alarm rate from a handful of seed-level standard errors or heuristic thresholds.
- General repair effectiveness based on reverting a known injected corruption.
- Treating agreement with the independent lag-Ridge implementation as proof of optimal forecasting, or treating a loss decrease as proof of a correct implementation.
- Treating experimental NumPy-agreement and target-specific-shift comparators as implemented public diagnostics.

## Literature evidence checked from primary sources

Canonical sources use publication DOIs where available and versioned arXiv
abstract pages for preprints. Official documentation and the stable JSTOR
record are retained where neither applies.

| Key | Source | Supported use |
| --- | --- | --- |
| `seca2021timegym` | https://arxiv.org/abs/2105.01404v1 | Historical predecessor. |
| `tan2025syntsbench` | https://doi.org/10.52202/085713-4860 | Controlled synthetic feature evaluation and theoretical reference comparisons already exist. |
| `ghoroghchian2026simpletimebench` | https://arxiv.org/abs/2610.02058v1 | SimpleTimeBench already frames elementary forecasting tasks as diagnostic unit tests; cited as a preprint. |
| `dwarakanath2019metamorphic` | https://doi.org/10.1109/MET.2019.00014 | Forecasting metamorphic tests, eight reported application issues, and mutation evaluation predate TimeGym; this predecessor has real-issue evidence absent here. |
| `sktimeConformance` | https://www.sktime.net/docs/api-reference/sktimeutilsestimator-checkscheck-estimator/ | Framework-specific estimator conformance checks are an existing comparison class; no head-to-head evaluation is claimed. |
| `qiu2024tfb` | https://doi.org/10.14778/3665844.3665863 | Standardized forecasting evaluation pipelines address a related implementation problem. The checked publication is PVLDB 17(9), 2024. |
| `hope1968montecarlo` | https://doi.org/10.1111/j.2517-6161.1968.tb00759.x | Monte Carlo significance testing predates the implemented reference-covariance rank check. |
| `lei2018conformal` | https://doi.org/10.1080/01621459.2017.1307116 | Split calibration yields marginal finite-sample guarantees under exchangeability; it does not guarantee conditional error control for each realized calibration set. |
| `holm1979multiple` | https://www.jstor.org/stable/4615733 | Step-down multiplicity adjustment applies to a declared family of valid tests, not arbitrary adaptive searches. |
| `gneiting2011point` | https://doi.org/10.1198/jasa.2011.r10138 | Point scoring functions must match the requested statistical functional. |
| `gneiting2007proper` | https://doi.org/10.1198/016214506000001437 | Proper distributional scoring; energy score and CRPS. |
| `scheuerer2015variogram` | https://doi.org/10.1175/MWR-D-14-00269.1 | Variogram scores complement energy scores when evaluating dependence. |
| `rAirPassengers` | https://stat.ethz.ch/R-manual/R-devel/library/datasets/html/AirPassengers.html | Official provenance for the single historical real-series repair example. |
| `gama2014drift` | https://doi.org/10.1145/2523813 | The distinction between process drift and changes in the generating parameters or predictive relationship in “Seasonality, process drift, and changing mechanisms” (`main.tex`, Appendix A). It does not make every changing latent state concept drift. |
| `merton1976jumps` | https://doi.org/10.1016/0304-405X(76)90022-2 | The Merton construction in “Financial stress mechanisms and non-Gaussian references” (`main.tex`, Appendix A): continuous diffusion plus compound-Poisson jumps. TimeGym specifies a physical drift, not an option-pricing validation. |
| `bollerslev1986garch` | https://doi.org/10.1016/0304-4076(86)90063-1 | The GARCH(1,1) conditional-variance recursion and its parameter restrictions in “Financial stress mechanisms and non-Gaussian references” (`main.tex`, Appendix A). |
| `hamilton1989regimes` | https://doi.org/10.2307/1912559 | The hidden-regime principle in “Financial stress mechanisms and non-Gaussian references” (`main.tex`, Appendix A): inference integrates over an unobserved Markov state; the terminal regime is not observed forecaster input. |
| `sun2026finstressts` | https://doi.org/10.1145/3770855.3817578 | The related-work sentence identifying isolated and composed financial mechanisms, including volatility, regimes, and jumps (`main.tex`). FinStressTS is direct prior work; financial composition alone is not a novelty claim. |
| `hu2026fintsb` | https://doi.org/10.1007/s11704-026-51064-5 | The related-work sentence about market constraints and transaction costs in financial evaluation (`main.tex`). The present synthetic tests do not establish trading performance. |
| `bollerslev1990ccc` | https://doi.org/10.2307/2109358 | The constant conditional correlation construction (CCC-GARCH) in “Joint mechanisms and dependence-aware uncertainty” (`main.tex`, Appendix A). It supplies the model precedent, not a claim of exact multi-step cross-moments or arbitrary dynamic correlations. |
| `weyuker1982testing` | https://doi.org/10.1093/comjnl/25.4.465 | The oracle-problem framing in “The oracle problem” (`main.tex`) and Table “oracles”: non-testable programs, partial oracles, simplified data with known results, and known properties. Read in full. |
| `davis1981pseudo` | https://doi.org/10.1145/800175.809889 | The pseudo-oracle: an independent implementation compared within a tolerance. Supports calling the experimental NumPy comparator a pseudo-oracle; it does not make that comparator a public diagnostic. Read in full. |
| `guderlei2007statistical` | https://doi.org/10.1109/QSIC.2007.4385527 | Statistical metamorphic testing: relations between output distributions checked by hypothesis tests. Names the class of the covariance rank check and the calibrated loss alarm. Read in full. |
| `seca2021review` | https://arxiv.org/abs/2105.01407v1 | Earlier review of oracle issues in machine learning by the author; cited for context only. |
| `pham2019cradle` | https://doi.org/10.1109/ICSE.2019.00107 | Cross-implementation comparison as an established oracle for deep-learning libraries (12 bugs, 104 inconsistencies across three backends). No claim about forecasting libraries. |
| `nejadgholi2019study` | https://doi.org/10.1109/ASE.2019.00078 | Tolerance-based (“oracle approximation”) assertions are 5–25% of assertions in four deep-learning libraries and their thresholds are frequently edited. Supports stating tolerances explicitly and the `INCONCLUSIVE` outcome. |
| `dutta2020detecting` | https://doi.org/10.1145/3395363.3397366 | Flaky tests in probabilistic and ML projects are mostly fixed by loosening thresholds; fixed seeds can hide bugs. Supports seed sweeps; no claim that TimeGym detects flaky tests. |
| `claessen2000quickcheck` | https://doi.org/10.1145/351240.351266 | Property-based testing: properties stated beside the code and checked on generated inputs. Declared relations are properties; scenario generators supply inputs. |
| `jia2011analysis` | https://doi.org/10.1109/TSE.2010.62 | Survey definitions: mutation score, competent programmer hypothesis, coupling effect, equivalent mutants. Vocabulary only. |
| `andrews2005mutation` | https://doi.org/10.1145/1062455.1062530 | Generated mutants resembled real faults in detectability; hand-seeded faults were harder to detect. Supports the statement that the hand-selected corpus is closer to the hand-seeded case; it does not estimate how the present faults compare with real ones. |
| `just2014mutants` | https://doi.org/10.1145/2635868.2635929 | 357 real faults in five Java programs: 73% coupled to common mutants, 27% not, 17% coupled to no mutant. Numbers quoted as reported; they concern Java programs, not forecasting code. |
| `humbatova2020taxonomy` | https://doi.org/10.1145/3377811.3380395 | A taxonomy of real deep-learning faults from 1,059 artifacts and 20 interviews, intended as a source of mutation operators. Supports the sentence that no comparable taxonomy exists for forecasting pipelines. |
| `humbatova2021deepcrime` | https://doi.org/10.1145/3460319.3464825 | Mutation operators derived from real-fault taxonomies with a statistical killing definition (repeated retraining, significance and effect size). Analogue of the calibrated loss alarm; no tool comparison is claimed. |
| `jahangirova2020empirical` | https://doi.org/10.1109/ICST46399.2020.00018 | Abstract-level support (full text not retrieved) that mutant killing for trained models needs a stochastic definition because retraining alone moves accuracy. |
| `arcuri2011practical` | https://doi.org/10.1145/1985793.1985795 | Guidance to assess randomized procedures over repeated independent runs with statistical tests and effect sizes. Supports the reporting convention; the paper's intervals follow its own stated conventions. |
| `breck2017test` | https://doi.org/10.1109/BigData.2017.8258038 | Twenty-eight production-readiness tests for ML systems written as assertions and scored. Context for related work; no comparison with the rubric is claimed. |
| `amershi2019software` | https://doi.org/10.1109/ICSE-SEIP.2019.00042 | Nine-stage ML workflow and three ML-specific engineering differences. Places TimeGym at the model evaluation stage. |
| `gneiting2011quantiles` | https://doi.org/10.1016/j.ijforecast.2009.12.015 | The mean is optimal under quadratic loss, any median under absolute loss, and a quantile under generalized piecewise-linear loss. Supports the estimand sentences in “Statistical foundations” and “Information and estimands”. |
| `gneiting2007probabilistic` | https://doi.org/10.1111/j.1467-9868.2007.00587.x | Maximizing sharpness subject to calibration; calibration is a joint property of forecasts and observations, sharpness of forecasts alone; PIT uniformity is necessary but not sufficient. |
| `gneiting2014probabilistic` | https://doi.org/10.1146/annurev-statistics-062713-085831 | Review restating the paradigm and that point scoring functions must be consistent for the functional. |
| `matheson1976scoring` | https://doi.org/10.1287/mnsc.22.10.1087 | Origin of the CRPS as an integrated quadratic score (attribution per Gneiting and Ranjan 2011; only the first page of the scanned copy was legible). |
| `hersbach2000decomposition` | https://doi.org/10.1175/1520-0434(2000)015%3C0559:DOTCRP%3E2.0.CO;2 | CRPS reduces to absolute error for a point forecast and decomposes into reliability, resolution, and uncertainty for ensembles. The paper reports averaged CRPS only; no decomposition is implemented. |
| `gneiting2008assessing` | https://doi.org/10.1007/s11749-008-0114-x | The energy score generalizes the CRPS and is proper; the multivariate rank histogram checks ensemble calibration. Supports the energy-score sentence; the paper's estimator includes diagonal pairs, as stated. |
| `gneiting2011comparing` | https://doi.org/10.1198/jbes.2010.08110 | Threshold- and quantile-weighted CRPS remain proper. Context only; no weighted score is implemented. |
| `christoffersen1998evaluating` | https://doi.org/10.2307/2527341 | Interval forecasts need conditional coverage (unconditional coverage plus independence of violations). Supports the sentence that coverage across one path's dependent coordinates is descriptive. First two pages of the scanned copy read. |
| `goodwin1999asymmetry` | https://doi.org/10.1016/S0169-2070(99)00007-2 | The symmetric MAPE treats positive and negative errors asymmetrically. Supports the sentence on the 2021 version's use of sMAPE. Read in full. |
| `tashman2000out` | https://doi.org/10.1016/S0169-2070(00)00065-0 | Fixed-origin versus rolling-origin evaluation, updating versus recalibration, and multiple test periods. Supports describing the study as fixed-origin with recalibration; no rolling-origin evaluation is implemented. |
| `bergmeir2018note` | https://doi.org/10.1016/j.csda.2017.11.003 | Standard K-fold cross-validation is valid for purely autoregressive models with uncorrelated errors. Context for the evaluation design; the author preprint was read. |
| `makridakis2018objectivity` | https://doi.org/10.1016/j.ijforecast.2018.05.001 | Reproducibility recommendations for forecasting research: data or a well-defined generating process with seeds, described methods, software versions, defined measures. Supports the reporting sentence. Read in full. |
| `spiliotis2020forecasting` | https://doi.org/10.1016/j.ijforecast.2018.12.007 | Forecasting competition datasets differ from each other and from a large real-world sample in feature space. Supports the limitation that the synthetic families are not representative by design. |
| `engle1982autoregressive` | https://doi.org/10.2307/1912773 | The ARCH process that GARCH(1,1) generalizes. Attribution only. |

## Main-figure evidence

- **Forecast matrix:** the same settings and fixed seeds as `notebooks/00_quick_start.ipynb`, parsed from cell `quick-03` and refitted by `paper/figure_matrix.py`; plotted observations, forecasts, references, and display limits are recorded in `paper/figures/paper-figure-provenance.json`. The walk rows contrast zero drift with drift 0.03 per step using identical innovations; their labels come from the package. This is an illustrative capability view, not a new performance comparison or extra study replication.
- **Dependence:** the first protocol-1 evaluation case and original sampling/mutation seed offsets. The renderer checks exact equality of sorted samples at all 16 coordinates, reproduces the saved case scores, and recomputes all plotted paired intervals from the 64-case records. The empirical CDF depicts one coordinate; equality at every coordinate is checked separately.
- **Diagnostic evidence:** counts and paired repair losses from protocol 2. Pooled alarm counts are descriptive; the clean row retains false alarms; corrected means clean refitted, with no separate repaired arm. The repair counterexample is explicitly selected to illustrate the distinction already reported in the manuscript, and its interval is not multiplicity-adjusted.

SVG, PNG, and embedded TikZ share plotted coordinates; type is set by the document. No raster images or external image
files are required by the standalone manuscript. Figure generation does not
alter experimental records or silently select replacement seeds.


## Controlled processes

The package adds four scalar and joint kinds (`composite`,
`merton_jump_diffusion`, `garch`, `markov_switching`) through the same Scenario
API. This software capability is supported by formula-based, limiting-case,
conditional simulation, regime-path enumeration, information-boundary, and
replay tests. It does **not** broaden either recorded Gaussian fault-study
corpus or establish state-of-the-art stock forecasting.

`SimulationReference` supplies analytic means and explicitly analytic or numerical
covariance under its declared conditioning distribution and explicit trajectory simulation. Its quantiles are empirical.
Hidden regimes plus jumps use numerical filtering; the covariance rank check
is unsupported for that approximate sampler. No hidden terminal regime or
future realized volatility is supplied as observed information.

The financial matrix reads configurations directly from application notebook
26 and records its 18 illustrative cases in the figure provenance file.

## Joint controlled processes

All four additional families support dependent targets. The declared laws use
correlated innovations, shared jump clocks and/or a shared hidden chain;
CCC-GARCH uses target-wise variance recursions. This is not arbitrary composition
of independent and common jump processes, a separate chain per asset, DCC-GARCH,
or a claim about real-market forecasting effectiveness.

Joint Merton and composite moments are analytic. Markov moments integrate the
posterior inferred from all observed targets; joint jump filtering remains
approximate and cannot support the exact-reference covariance check.
Multivariate Markov density filtering requires positive diffusion scales and
positive-definite within-state correlations.

General multi-step CCC-GARCH cross-covariance is an explicitly declared Monte
Carlo estimate; means and marginal variances are analytic. Its trajectory
sampler is exact under the specified law. The estimated covariance is held
fixed independently of forecast/null ensembles, preserving exchangeability of
rank statistics conditional on that target. This is not an exact-moment claim
or a guarantee of power against all dependence errors.

Validation includes independent hidden-state-path enumeration, multivariate
Gaussian forward-filter calculations, direct cumulant calculations,
Gauss-Hermite integration of two-step GARCH cross-moments, conditional simulation,
shape/parameter boundaries, retained scalar behavior, and public report replay.
Notebook 27 and the joint figure demonstrate re-pairing whole target trajectories:
per-target marginal distributions and temporal paths remain unchanged while
cross-series dependence and aggregate uncertainty change.
