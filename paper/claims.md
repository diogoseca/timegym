# Claim and evidence ledger

Evidence reviewed 28 October 2026; citation identifiers and the financial-source entries refreshed 30 October 2026; testing, oracle, and scoring-rule sources added from retrieved full texts on 4 November 2026; the exploratory Study 2 follow-up and the sources of the 13 November 2026 corrections added on that date. The contribution is an executable workflow that combines established testing and forecasting principles with inspectable conditional references. A claim becomes an empirical result only when its generated artifacts exist and the manuscript reports their actual output. Passing tests supports the tested cases; it does not prove correctness for all inputs.

| Claim | Evidence and boundary | Manuscript treatment |
| --- | --- | --- |
| Forecast output contracts separate structural violations from model capability. | `timegym/contracts.py`, `tests/test_contracts.py`, and `tests/test_validation_boundaries.py`: explicit horizon/target/sample/quantile axes, numeric and index validation. Capability failures are reported separately. This does not infer the unique source-code cause. | Implemented method; scope shown through controlled examples. |
| Synthetic references match the requested functional and information set. | `timegym/scenarios.py` and `tests/test_references.py`: analytic conditional Gaussian means, marginal quantiles, covariance, simulation checks, and explicit boundary cases. The simulator and reference share assumptions and some code; simulation agreement is not independent proof. Parameters are known to the evaluator, not generally to learned forecasters. | Mathematical specification plus implementation validation. |
| Selected faults can be detected and their known repairs checked. | The output-corruption pilot remains in `experiments/results/results.json`; implementation and covariance studies use separate `experiments/protocol-v2.json` and `experiments/results-v2/results.json`. Clean, faulty, and repaired runs share cases; calibration cases are separate. Misses and false alarms remain in the record. | Two controlled, developer-selected studies; no estimate of natural fault prevalence or coverage. |
| Different relations expose different implementation defects. | `experiments/model_faults.py`, `tests/test_research_supplement.py`, and `experiments/results-v2/model-ablation.csv`: all four internal regressions preserve the output contract; scalar translation detects one branch, per-target translation two, and agreement with the separately written NumPy implementation all four on the specified cases. The last two comparators are experiment-only. The shift-check outcomes follow from the equivariance each branch preserves (the scalers are refitted on the shifted context), so columns C, S, T and N confirm predictions; the loss alarms and paired losses are measurements. | Complementary evidence, not a universal ranking or a claim that the public package automatically identifies every defect. The second implementation was written by the same author, who knew the faults. |
| An alarm on the distance from the answer key catches subtle implementation faults that the realized-loss alarm misses. | `experiments/protocol-v2-followup.json` and `experiments/results-v2-followup/results.json` (protocol 2.1, exploratory, chosen 13 November 2026 after the protocol 2.0 results were known): the 768 clean calibration fits re-run and checked against the frozen realized MSEs; the same conformal rule at 5% applied to `mean_reference_mse` of the frozen evaluation records. Reversed lags or stale state: 148 of 640 cases outside the seasonal condition against 36 for the loss alarm; clean: 12 against 13 of 320. `tests/test_experiment_pipeline.py` recomputes the thresholds and counts from the stored distances. | Exploratory addition, labelled as such in the table, figure and text; not part of the frozen protocol. Helps on the autoregressive conditions, little on trends and random walks. |
| The repair counterexample on the random walk with drift has a known mechanism. | Follow-up record: own-lag coefficients of the clean fit sum to 0.95 and 0.91 on average (minimum 0.78 and 0.66); the correct recursion's squared distance from the answer key grows from 0.08 to 0.95 over eight steps, the stale branch's to 0.17. Frozen record: clean-minus-faulty `mean_reference_mse` +0.354 [0.242, 0.467], faulty branch closer in 53 of 64 cases. Cited mechanism: `kendall1954note`, `dickey1979distribution`, `marcellino2006comparison`, `chevillon2007direct`. | Explanation of one selected counterexample; not a general result about repairs. |
| A ten-instance scorecard cell has a seed-to-seed spread of about 3 points on the noise rows and up to 19 on the random-walk rows. | Follow-up record: the Figure 1 pipeline with 30 instances per cell (seed 0); the first ten reproduce every Figure 1 score; 4,000 bootstrap resamples of ten instances per cell. Standard deviations run from 2.7 to 19.2 over the 20 cells. | Exploratory; describes this pipeline, not every model. |
| The fitted AR(1) gets worse with noise on the trend because its coefficient estimate falls. | Follow-up record: on the Study 1 capability trend cases (seeds 10000–10015), the mean fitted coefficient is 0.995 at sigma 0.25 and 0.77 at sigma 2, with the forecast 3.9 below the answer key on average at sigma 2; the re-run reproduces the frozen Study 1 distances. | Explanation of one table cell. |
| The score is a relative efficiency against the answer key. | Equation 3: 100 U/(U+A) is the answer key's expected squared error over the forecast's; the inverse of the relative MSE of `hyndman2006another` with the answer key as benchmark, in the family of `murphy1988skill`. Band edges 90, 70, 40 mean an avoidable error of 0.11, 0.43, 1.5 times the unavoidable one (algebra). | Definition; no empirical claim. |
| A correctly specified model still scores below 100 on a finite history. | Algebra: the sample mean has avoidable error sigma²/120 on noise around a fixed level (score about 99). Figure 1 provenance and follow-up record: the 24-lag Ridge scores 84 on the noise rows, losing points to 25 coefficients estimated from 96 rows. `clements1998forecasting` for the decomposition of forecast error. | Interpretation of the avoidable error; no new estimate. |
| The Mistakes column of the scorecard is defined from an exact decomposition. | `timegym/scorecard.py` `_mistakes`: offset, drift and shape parts of each error against the answer key whose mean squares add up to the avoidable error; parts with at least 25% (10% for a visible drift) are named, for cells scoring below 70. | Description of the package. |
| The score separates patterns a model cannot learn from noise nobody can predict (RQ1). | By construction through the answer key (Equation 2); Study 1 capability table (`experiments/results/results.json`, generated by `experiments/render_results.py`) and the Figure 1 scorecard: scores fall where a forecast misses what the history determines, are equal at both noise levels on the scale-free families, and rise with noise where ignoring a fixed-size pattern becomes cheaper. | Reading of the existing table and figure on the tested families; not a ranking of models on real series. |
| A capability failure does not establish a software fault. | Last-value, seasonal-naive, and fitted Gaussian AR(1) baselines run on the same mechanisms in `experiments/results/results.json`. Misspecified but well-formed predictions remain possible. | Explicit interpretation of capability scores. |
| Marginal sample scores cannot identify every dependence defect. | Coordinate-wise sample permutations preserve empirical marginal measures. The manuscript states the finite-sample identity in one sentence (shuffling reorders each coordinate's samples without changing them); `tests/test_metrics.py` and paired study records check its numerical consequences. This identity is elementary, not a new probability theorem. | Algebraic negative control plus finite-sample experiment. |
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
- Treating agreement with the separately written lag-Ridge implementation as proof of optimal forecasting, or treating a loss decrease as proof of a correct implementation.
- Treating experimental NumPy-agreement and target-specific-shift comparators as implemented public diagnostics.

## Literature evidence checked from primary sources

Canonical sources use publication DOIs where available and versioned arXiv
abstract pages for preprints. Official documentation and the stable JSTOR
record are retained where neither applies.

| Key | Source | Supported use |
| --- | --- | --- |
| `seca2021timegym` | https://arxiv.org/abs/2105.01404v1 | Historical predecessor. |
| `tan2025syntsbench` | https://doi.org/10.52202/085713-4860 | Controlled synthetic feature evaluation and theoretical reference comparisons already exist. Read in full (arXiv:2510.20273v1) on 12 November 2026: a theoretical optimum is derived for each synthetic task (Appendix D.2.2) and reported in its own column beside raw MSE and MAE; no score is stated relative to it; the metrics are point metrics; real datasets appear as an appendix validation. |
| `ghoroghchian2026simpletimebench` | https://arxiv.org/abs/2610.02058v1 | SimpleTimeBench already frames elementary forecasting tasks as diagnostic unit tests; cited as a preprint. Read in full (v1) on 12 November 2026: the ideal forecast of each process is described (Appendix A.1), scores are relative to naive baselines, accuracy is measured on one target variable, and the CRPS appears only in the comparison on real datasets. The arXiv PDF carries an ICML 2026 (PMLR 306) footer that was not checked against the proceedings. |
| `dwarakanath2019metamorphic` | https://doi.org/10.1109/MET.2019.00014 | Forecasting metamorphic tests, eight reported application issues, and mutation evaluation predate TimeGym; this predecessor has real-issue evidence absent here. Read in full (arXiv:1907.06632v1) on 12 November 2026: 19 relations, and tool-generated mutants of which 29 of 44 valid ones were caught. |
| `sktimeConformance` | https://www.sktime.net/docs/api-reference/sktimeutilsestimator-checkscheck-estimator/ | Framework-specific estimator conformance checks are an existing comparison class; no head-to-head evaluation is claimed. The statements that the suite exercises the interface on unstructured noise series and does not judge what the forecast says come from the suite's source on the main branch (`sktime/utils/estimator_checks.py`, `sktime/forecasting/tests/test_all_forecasters.py`, `sktime/utils/_testing/series.py`), read on 12 November 2026. |
| `qiu2024tfb` | https://doi.org/10.14778/3665844.3665863 | Standardized forecasting evaluation pipelines address a related implementation problem. The checked publication is PVLDB 17(9), 2024. |
| `brockman2016gym` | https://arxiv.org/abs/1606.01540v1 | The opening analogy only: reinforcement learning has standard environments with known rules in which agents are trained and tested. No technical claim rests on it. |
| `yule1927method` | https://doi.org/10.1098/rsta.1927.0007 | Origin of the second-order autoregression as a model of disturbed periodic series; supports the sentence that an AR(2) with complex roots rings as a damped oscillation. Bibliographic record checked through Crossref; the statement itself is textbook material. |
| `kalman1960new` | https://doi.org/10.1115/1.3662552 | The Kalman filter as the exact conditional law of a linear Gaussian state-space model; supports the answer key under measurement noise (`timegym/scenarios.py`, `_kalman_forecast`). Bibliographic record checked through Crossref. |
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
| `sun2026finstressts` | https://doi.org/10.1145/3770855.3817578 | The related-work paragraph on the financial scenarios (`main.tex`): a synthetic benchmark of 30 environments from six mechanism families, including volatility, heavy tails, regimes, and jumps, on which 15 models are compared. Read in full (arXiv:2606.03184v1, first posted 2 June 2026) on 12 November 2026. FinStressTS is direct prior work for the financial scenarios in the catalog only: TimeGym dates from 2021 and tests one pipeline, whereas FinStressTS is a benchmark for comparing models. Financial composition alone is not a novelty claim. |
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
| `andrews2005mutation` | https://doi.org/10.1145/1062455.1062530 | Generated mutants resembled real faults in detectability; hand-seeded faults were harder to detect. The hand-seeded faults were seeded by others to resemble real ones; the present faults were picked to exercise particular checks, a different selection that favors detection by those checks. Neither kind of corpus estimates how often the checks catch faults found in practice. |
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
| `kendall1954note` | https://doi.org/10.1093/biomet/41.3-4.403 | Small-sample bias of autocorrelation estimates; with `dickey1979distribution`, the downward bias of least-squares autoregressive coefficients near a unit root, cited for the random-walk repair mechanism. Bibliographic record from Crossref, 13 November 2026. |
| `dickey1979distribution` | https://doi.org/10.1080/01621459.1979.10482531 | Distribution of the least-squares autoregressive estimator under a unit root. Same use as `kendall1954note`. Crossref record. |
| `marcellino2006comparison` | https://doi.org/10.1016/j.jeconom.2005.07.020 | Iterated multi-step forecasts versus direct ones; iterated forecasts suffer when the one-step model is misspecified. Crossref record. |
| `chevillon2007direct` | https://doi.org/10.1111/j.1467-6419.2007.00518.x | Survey of direct multi-step estimation; same use as `marcellino2006comparison`. Crossref record. |
| `hyndman2006another` | https://doi.org/10.1016/j.ijforecast.2006.03.001 | Relative MSE against a benchmark; the score is its inverse with the answer key as benchmark. Crossref record. |
| `murphy1988skill` | https://doi.org/10.1175/1520-0493(1988)116<2417:SSBOTM>2.0.CO;2 | MSE skill scores; places the score in that family. Crossref record. |
| `clements1998forecasting` | https://doi.org/10.1017/CBO9780511599286 | Decomposition of forecast error into its sources, and simulation studies scored against the forecast a known process implies. Book; Crossref record. |
| `geweke2004getting` | https://doi.org/10.1198/016214504000001132 | Validating posterior simulators by simulating from known parameters. Precedent for answer keys and the covariance rank check; intended use of the joint-spread check. Crossref record. |
| `cook2006validation` | https://doi.org/10.1198/106186006X136976 | Validating Bayesian software by the rank of known parameters among posterior draws. Same use as `geweke2004getting`. Crossref record. |
| `modrak2025simulation` | https://doi.org/10.1214/23-BA1404 | Simulation-based calibration checking. Same use as `geweke2004getting`. Crossref record. |
| `knight1986experimental` | https://doi.org/10.1109/TSE.1986.6312924 | Separately developed versions fail together more often than independence predicts. Supports calling the NumPy lag-Ridge "separately written", not independent. Crossref record. |
| `barr2015oracle` | https://doi.org/10.1109/TSE.2014.2372785 | Survey of the test oracle problem. Background citation. Crossref record. |
| `segura2016survey` | https://doi.org/10.1109/TSE.2016.2532875 | Survey of metamorphic testing. Background citation. Crossref record. |
| `chen2018metamorphic` | https://doi.org/10.1145/3143561 | Review of metamorphic testing. Background citation. Crossref record. |
| `zhang2022machine` | https://doi.org/10.1109/TSE.2019.2962027 | Survey of machine-learning testing. Background citation. Crossref record. |
| `xie2011testing` | https://doi.org/10.1016/j.jss.2010.11.920 | Early metamorphic relations for machine-learning classifiers. Background citation. Crossref record. |
| `kang2020gratis` | https://doi.org/10.1002/sam.11461 | Generator of time series with controllable features for evaluating forecasting methods. Context for the answer key not being new. Crossref record. |
| `gilpin2021chaos` | https://arxiv.org/abs/2110.05266v1 | Chaotic systems with known dynamics as a forecasting benchmark. Same use as `kang2020gratis`. arXiv record (v1, October 2021). |
| `alexander2024evaluating` | https://doi.org/10.1007/s10479-022-04611-9 | Simulation comparison of multivariate scoring rules in which the variogram score with p = 0.5 discriminated best; supports the choice of p = 1/2. Crossref record (published online 2022, in print 2024). |
| `liang2025tsgym` | https://arxiv.org/abs/2509.17063v1 | TSGym, an unrelated 2025 project with a similar name; footnote only. arXiv record. |
| `cai2025timeseriesgym` | https://arxiv.org/abs/2505.13291v1 | TimeSeriesGym, an unrelated 2025 project with a similar name; footnote only. arXiv record. |

## Related-work comparison table

Table “positioning” in `main.tex` compares SynTSBench, SimpleTimeBench,
FinStressTS, Dwarakanath et al., the sktime conformance suite, and TimeGym by
use, data, scoring, and software checks. Every cell for another work was read
from that work's full text (for sktime, from the conformance suite's source)
on 12 November 2026. A check mark means the paper reports the capability; a
check mark in brackets means in part, and the caption says what the part is;
a dash means the text read does not report it, not that the capability is
demonstrably absent. Repository code beyond each README was not inspected.
The table describes capabilities and use; it claims no head-to-head result.
The scorecard figure, the scenario grid and the capability table are
descriptions of this package and of the frozen Study 1 records.

## Main-figure evidence

- **Scorecard:** the package scorecard (`Pipeline.scorecard()` with its defaults: ten kinds of series by two noise levels, 10 instances per cell, seed 0) of the quick start's contender, `Pipeline(window_length=24)` with `StandardScaler` and `Ridge(alpha=1.0)`, written in `paper/figure_scorecard.py` and refitted when the figure is rendered. Scores, verdicts, mistakes, titles and sentences are read from the `Scorecard` object; each panel plots instance 1 of its cell. Every cell's score, verdict, mistakes, unavoidable and avoidable error, the plotted values and display limits are recorded in `paper/figures/paper-figure-provenance.json`. This illustrates the package output for one model; it is not a study result, a performance comparison or an extra study replication.
- **Dependence:** the first protocol-1 evaluation case and original sampling/mutation seed offsets. The renderer checks exact equality of sorted samples at all 16 coordinates, reproduces the saved case scores, and recomputes all plotted paired intervals from the 64-case records. The empirical CDF depicts one coordinate; equality at every coordinate is checked separately.
- **Diagnostic evidence:** counts and paired repair losses from protocol 2; the answer-key column comes from the exploratory follow-up (protocol 2.1) and is marked with a dagger. Pooled alarm counts are descriptive; the clean row retains false alarms; corrected means clean refitted, with no separate repaired arm. The repair counterexample is explicitly selected to illustrate the distinction already reported in the manuscript, and its interval is not multiplicity-adjusted.

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

The financial matrix reads configurations directly from tutorial 11 (composed
and financial stresses) and records its 18 illustrative cases in the figure provenance file.

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
The joint financial uncertainty tutorial (`tutorials/12_joint_financial_uncertainty.ipynb`) and the joint figure demonstrate re-pairing whole target trajectories:
per-target marginal distributions and temporal paths remain unchanged while
cross-series dependence and aggregate uncertainty change.
