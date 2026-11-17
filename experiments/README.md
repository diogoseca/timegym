# Reproduce the controlled studies

From the repository root, with Python 3.13 (the version the lock records),
create the environment once:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-study.txt
.venv/bin/python -m pip install --no-build-isolation --no-deps -e .
```

Render the frozen records without running either study:

```bash
.venv/bin/python -m experiments.render_results
.venv/bin/python -m experiments.render_supplement
.venv/bin/python paper/render_figures.py
.venv/bin/python paper/build_standalone.py
```

Both runners refuse existing `results.json`, `manifest.json`, `runtime.json`,
`example-*.json`, or CSV artifacts before doing any computation. `--force` is an
explicit replacement option for an approved re-freeze; routine reproductions
use a new output directory. Both runners enforce one numerical thread with
`threadpoolctl` and record the observed environment and thread pools in separate
`runtime.json` files.

## Study 1: controlled pilot

```bash
.venv/bin/python -m experiments.run_study --output /tmp/timegym-pilot-repeat
.venv/bin/python -m experiments.compare_results --results /tmp/timegym-pilot-repeat/results.json --manifest experiments/results/manifest.json
```

Packages and the public dataset are available locally after setup; the
experiment itself uses no network services or paid compute.

`protocol.json` version 1.1 makes the original capability task and inverse-transform
dimensions explicit, with unchanged values. The runner validates the protocol
and constructs only the models it declares. The original pilot design predates
inspection of its evaluation results.
Development seeds 0–15 calibrate a deliberately simple point-MSE alarm. Evaluation
seeds 10000–10063 are independent series. Every output mutation is paired with the
same clean and repaired ensemble. Six mutations violate the explicit output
contract; variance shrinkage and coordinate-wise sample shuffling preserve output
shapes. The shuffle preserves every empirical marginal distribution exactly.

The statistical check uses 99 independent reference ensembles and a per-case
0.05 rank-test threshold. Its null is independent sampling from the full known
reference law. It is not a test of arbitrary covariance equality, and it is not
a guarantee that a clean ensemble will pass. False alarms and Wilson confidence
intervals are retained in the paper. The MSE alarm ignores index labels and non-point
outputs.

The separate model-level fault omits inverse standardization in a historical-mean
forecast. Its repair applies the training location and scale exactly once. The
location-equivariance relation is explicitly declared for this model. The same
repair is illustrated at four ordered AirPassengers origins, without treating
them as independent replications.

Outputs:

- `results/results.json`: individual cases, scores, confidence intervals and summaries.
- `results/manifest.json`: protocol, dependency versions, base Git commit, current source
  fingerprint, results hash, and dataset hash. The source fingerprint identifies
  historical working files; the offline metadata migration preserves that evidence.
- `results/runtime.json`: thread environment for new executions; the frozen pilot
  predates this metadata, so its historical thread state is unknown.
- `results/example-*.json`: machine-readable counterexamples.
- `../paper/figures/*.png` and `../paper/figures/*.pdf`: figures rendered from the recorded results.
- `../paper/generated-results.tex`: manuscript tables derived from the same JSON.

There are no timestamps or runtime values in numerical results. A
platform-dependent tolerance may be needed on other BLAS/NumPy versions.
The CI `research-smoke` job reruns both studies and renders the paper's tables
from the fresh output, so it fails when a conclusion the paper states does not
hold there; it does not compare hashes. See [the validation record](../paper/validation.md).
Local comparisons fail on a mismatch
unless `--warn` is explicit.

Scope limits: no held-out generator families, representative real-bug corpus,
foundation models, sktime-conformance comparator, or broad real-data transfer study.

## Study 2: implementation and covariance supplement

`protocol-v2.json` keeps the supplement separate from the pilot. It declares the
four implementation faults, six model conditions, calibration/evaluation seeds,
and the permutation-invariant covariance null set. A contract violation aborts
the model study. Corrected means the clean implementation refitted; comparisons
are clean-minus-faulty, with no duplicate repaired arm. The paired loss fields
are `clean_minus_faulty_mse` and `clean_minus_faulty_reference_mse`; the
`model-repair-losses.csv` filename is retained for compatibility.

```bash
.venv/bin/python -m experiments.run_supplement --output /tmp/timegym-supplement-repeat
.venv/bin/python -m experiments.compare_results --results /tmp/timegym-supplement-repeat/results.json --manifest experiments/results-v2/manifest.json
.venv/bin/python -m experiments.render_supplement --results-dir /tmp/timegym-supplement-repeat --paper-dir /tmp/timegym-supplement-paper
```

The renderer checks the manuscript conclusions against the supplied records and
raises if they are unsupported. Small smoke runs may therefore be unsuitable for
rendering the frozen manuscript's empirical conclusions.

The `results-v2/` output contains:

- `results.json`: calibration, implementation and covariance records, paired
  loss summaries, gained/lost alarm counts, and deterministic operation counts.
- `manifest.json`: protocol, result, source, environment, and pilot-result hashes.
- `runtime.json`: observed wall time and thread state, excluded from manuscript
  text and numerical results.
- `model-ablation.csv`: clean and faulty alarm totals.
- `model-conditions.csv`: per-condition clean/faulty counts and gained/lost alarms.
- `model-repair-losses.csv`: clean-minus-faulty MSE, paired t intervals, and
  individually improved cases.
- `covariance-conditions.csv`: raw and Holm counts and Wilson intervals.

The supplement renderer also emits `paper/generated-macros.tex` for caption
numbers (`RepairDeltaMean`, `RepairDeltaLow`, `RepairDeltaHigh`, `RepairImproved`,
`CasesPerCondition`, and `CasesPerRow`). It checks these against paired raw losses.

## Exploratory follow-up to Study 2 (protocol 2.1)

`protocol-v2-followup.json` declares four analyses chosen on 13 November 2026,
after the protocol 2.0 results were known. They are exploratory, and the paper
labels every number from them as such. The runner imports the frozen Study 1
and Study 2 code, re-runs only the fits it needs, checks each re-run value
against the frozen records (within 1e-9), and never writes to `results/` or
`results-v2/`:

```bash
.venv/bin/python -m experiments.run_followup --output /tmp/timegym-followup-repeat
cmp experiments/results-v2-followup/results.json /tmp/timegym-followup-repeat/results.json
```

It takes well under a minute on one CPU thread. The record answers:

- **Answer-key alarm.** The 768 clean Study 2 calibration fits are re-run; the
  same conformal rule at 5% is applied to each case's mean squared distance
  from the answer key, and the threshold to the frozen evaluation records'
  `mean_reference_mse`. Counts appear beside the frozen loss alarm in
  column K of the complete Study 2 table and in the diagnostic-evidence figure.
- **Repair mechanism** on the random walk with drift: own-lag coefficient sums
  of the clean fit, the per-step distance from the answer key of the clean and
  stale-state branches, and the frozen paired distance from the answer key.
- **Scorecard precision:** the Figure 1 pipeline with 30 instances per cell and
  a bootstrap of ten-instance scores.
- **Fitted AR(1) on the trend** in the Study 1 capability setting: its mean
  coefficient and its mean offset from the answer key.

The `results-v2-followup/` output contains `results.json` (including the 768
stored calibration distances) and `manifest.json` (protocol, results and
frozen-input digests, source digests, Python, library and package versions).
`experiments/render_followup.py` validates the record and supplies its macros
and table column to `render_results` and `render_supplement`; both accept
`--followup-dir`. The tests recompute the thresholds and counts from the stored
distances and the frozen records without re-running any fit.

## Second exploratory record for Study 2 (protocol 2.2)

`protocol-v2-addendum.json` declares two analyses chosen on 13 November 2026,
after the protocol 2.0 results and the protocol 2.1 follow-up were known. They
are exploratory and read only values stored in `results-v2/results.json`;
nothing is refitted, and the runner never writes to `results/`, `results-v2/`
or `results-v2-followup/`:

```bash
.venv/bin/python -m experiments.run_addendum --output /tmp/timegym-addendum-repeat
cmp experiments/results-v2-addendum/results.json /tmp/timegym-addendum-repeat/results.json
```

It takes a few seconds. The record answers:

- **Suite-level multiplicity.** From the stored p-values of the four condition
  checks in each of the 64 suites, per variant: Holm at 0.05 (recomputed, and
  required to equal every stored `holm_alarm`), Hochberg's step-up procedure,
  Simes' global test and Fisher's combination (chi-square with 8 degrees of
  freedom, in closed form). It stores suites with any rejection and suites
  rejecting a true null for each method, with Wilson 95% intervals, and
  per-condition rejection counts for raw, Holm and Hochberg decisions.
- **Paired answer-key check.** For each of the 24 pairs of condition and fault,
  the paired differences of the stored `mean_reference_mse` against the clean
  branch on the same seeds, a one-sided sign-flip test with 9,999 flips (seed 0,
  declared in the protocol), and Holm across the 24 pairs. The same analysis on
  the first 30 and the first 10 evaluation seeds is a sensitivity check.

The `results-v2-addendum/` output contains `results.json` (including every
suite's p-values and decisions) and `manifest.json` (protocol, results and
frozen-input digests, source digests, Python, library and package versions).
`experiments/render_addendum.py` validates the record, checks the statements
the manuscript makes from it (`check_addendum_conclusions`) and supplies its
macros to `render_supplement`, which accepts `--addendum-dir`. The tests
recompute both analyses from the frozen records. The `research-smoke` job of
the checks workflow re-runs the follow-up and this record and checks their
conclusions on the fresh output.

### Frozen-record migration

The records were migrated twice without rerunning either study (30 October and 3 November 2026); no value changed. Each manifest keeps the original hashes under `record_migration` and `loss_key_migration`.

## Study 3 (protocol 3.0)

`protocol-v3.json` is the plan of the paper's third study. It is fixed before
anything is run (a pre-registration), so that the results cannot shape the
questions. Its field `status` stays `draft` until it is merged. The run may
not start until a later commit writes the merge commit and its date into the
file and sets the status to `registered`.

The study asks six questions:

1. A tool plants hundreds of small bugs in a correct forecasting pipeline.
   Which checks catch them?
2. Do the checks still work when the model inside the pipeline is not a
   linear regression?
3. When one statistical check runs on several settings at once, can it answer
   both "did anything fail?" and "which one failed?"
4. Can the joint-spread check also say "close enough", and not only
   "different"?
5. How close do two published estimates of predictability come to the exact
   unavoidable error, which the answer key gives?
6. What scorecards do forecasting models that people use get?

Terms used below:

- **Mutation testing** makes many small deliberate changes to a correct
  program and counts how many of them the checks notice. Each changed copy is
  a **mutant**, and a rule that makes one kind of change is an **operator**.
  An **equivalent mutant** behaves exactly like the original, so no check
  could catch it.
- The **subject** is the program under test, here one small forecasting
  pipeline.
- A **second implementation** is the same pipeline written again
  independently. Where the two disagree, one of them has a bug.
- **Calibration cases** are runs of the bug-free pipeline. They show how
  large a statistic gets by chance, so that a check can tell a bug from noise.
- The **catalogue** is this study's list of 16 hand-written time-series bugs.
- The **zoo** is the set of nine forecasting models scored for question 6.

Once the plan is registered, the run is these commands in this order. The
first uses the Cosmic Ray environment, the last three the zoo's environments,
and the others the study lock:

    python -m experiments.study3_generate --output experiments/results-v3/mutants
    python -m experiments.run_study3_mutants --part 1 --workers 4
    python -m experiments.run_study3_mutants --part 2 --workers 4
    python -m experiments.run_study3_suites
    python -m experiments.run_study3_margin --workers 4
    python -m experiments.run_study3_predictability
    python -m experiments.zoo.run_zoo
    python -m experiments.zoo.run_zoo --models chronos_small
    python -m experiments.zoo.figure

Every runner first asks `experiments/study3_common.py` whether its part may
start and stops with the list of what is missing; no flag bypasses that. With
`--plan` a runner prints what it would run and computes nothing, which also
works on the draft. Part 2 takes hours and writes its record at the end, so
`run_study3_mutants` has `--resume` for a run that a stop of the machine
interrupted: started again with the same `--output`, it keeps every unit of
work that the stopped run ended under the same specification and time limit
and runs the others. Each process runs its numerical libraries on one
thread. The units of Parts 1 and 2, the parent that computes their statistics
and the zoo's models load those libraries after the process has started, where
a limit set with threadpoolctl alone limits nothing;
`experiments/study3_threads.py` makes them start with one thread, and each of
these processes reads its thread pools back before it writes its record and
stops if one of them has more than one thread. After the merge, three commits may each change one value
of the protocol and nothing else: the one that registers it (date, merge
commit and status), the one that adds the generated mutant sources and writes
the digest of their manifest, without which Parts 1 and 2 do not start, and
the one that writes the Chronos model revision. The four files that Part 1
freezes (the subject, its second implementation, the catalogue and the
learners) carry their digests in the protocol already, and a test fails when
one of them changes.

The study runs on one numerical thread per process, on CPU only. Every Monte
Carlo rank check uses 239 calibration draws, so that p-values are multiples of
1/240 and the four Holm thresholds of a family of four at 0.05 (0.0125,
0.0167, 0.025 and 0.05) are attained exactly. Calibrated per-case alarms use
239 clean calibration cases per condition. The number of evaluation cases per
condition, 93, and the number of covariance suites, 93, are the smallest
numbers for which every Wilson 95% interval has a half-width of at most 10
percentage points. Every case seed lies in a new namespace between 7,000,000
and 15,000,000; the protocol lists the seed ranges of the earlier protocols
and of the package, and loading the protocol fails if any of its own ranges
overlap or leave that namespace. Rates carry Wilson 95% intervals, and every
case that cannot be evaluated is counted and reported.

The six parts answer the six questions, in this order:

1. **Generated mutants.** A lag-regression pipeline written for this study
   (`experiments/study3_subject.py`: lags, a seasonal lag of 12, scaling of
   inputs and target, a pluggable regressor, recursive forecasting and the
   inverse transform) is committed and hashed before any mutant exists.
   Mutants come from Cosmic Ray 8.9.0 with its standard operators and from a
   catalogue of ten kinds of time-series bug, written as 16 operators and
   defined in the protocol in a sentence or two each; the four Study 2 bugs
   are re-expressed on the same pipeline. All mutant sources are generated and
   their digests committed before any is executed. Each mutant is kept only
   if it was generated, runs, and is not equivalent to the clean pipeline on
   24 fixed inputs; the counts at each step are reported. It is then run
   through the output checks, the common and
   per-series shift checks, a second implementation, the calibrated loss
   alarm, the answer-key alarm of protocol 2.1, and a paired sign-flip check
   against the clean pipeline, in the six Study 2 model conditions. A
   deterministic check catches a mutant in a condition when it alarms on any
   case, provided it never alarms on the clean pipeline there. A calibrated
   alarm catches it when an exact beta-binomial test of its alarm count,
   Holm-adjusted across the mutants of the condition, rejects; the unadjusted
   decision is reported beside it. Detection rates are also drawn against
   false-alarm rates over all 240 attainable levels.
2. **Learners beyond Ridge.** Histogram gradient boosting on raw lags, a
   perceptron on window-standardized lags with a free training seed, and a
   perceptron on raw lags are first classified on clean pipelines as exactly
   shift equivariant, equivariant in distribution (checked with 39 refits per
   case), or not applicable (not shift equivariant even in distribution, so the
   shift checks do not apply). Each then receives the four Study 2 bugs, the 16
   catalogue mutants and 12 seeded generic mutants, checked with the checks
   that apply to its shift class.
3. **Two questions about a suite.** The covariance study of protocol 2.0 is
   repeated with 239 calibration draws: Fisher's combination answers whether
   a suite contains any failure, and Holm answers which condition failed.
4. **A joint-spread check with a margin.** An unbiased estimate of the
   squared relative Frobenius distance between the ensemble covariance and
   the reference covariance is compared, through a leave-one-out (jackknife)
   interval, with a margin of 0.25 or 0.4375 (the exact distance of the 0.75
   spread multiplier of Study 2). A simulation with 381 repetitions per cell
   measures how often each verdict occurs below, at and above the margin, with
   64 and 256 paths.
5. **Predictability estimators against the exact floor.** On 200 series of
   each of the 20 default scorecard cells, the spectral bound of Feng et al.,
   computed as their paper describes and also from the history alone, and
   weighted permutation entropy are compared with the unavoidable error that
   the answer key gives. The expectations for each row are written in the
   protocol.
6. **Scorecards of models people use.** Nine models are scored on the default
   grid with 30 instances, from `experiments/zoo/`, which has its own lock
   files: last value, seasonal naive, ETS, automatic ARIMA and Theta from
   statsforecast 2.1.1, gradient boosting and a perceptron on lags, Chronos
   (small) and context parroting. The pretrained model may run on another
   machine; the protocol states what its manifest records, and its model
   revision is written into the protocol before that run. No ranking claim is
   made.

Before registration, only the computations listed under
`pilots_before_registration` were made: installing Cosmic Ray and listing its
operators; counting, without executing any of them, the mutants it generates
on `experiments/model_faults.py` and on a copy of its clean branch; timing
clean pipelines and running the public shift check on 12 clean cases of each
learner, and of seven regressors on raw lags; installing statsforecast and
timing it on two generic series; testing which package indexes, model hosts
and literature servers this machine can reach; reading the cited papers and
the published code of the spectral bound; and checking formulas on textbook
cases. When the code was written, the clean subject alone was run on the 12
pilot cases, to confirm that its forecasts are valid, that the second
implementation agrees with it and that the shift checks are silent on it; the
catalogue was generated as text and compared with the expected text, never
executed. The protocol also says what the tests of the Study 3 code execute:
toy pipelines, toy mutants and hand-made arrays, on seeds below 1,000,000. The
protocol also lists the earlier exploratory records that bear on
the study: protocols 2.1 and 2.2, and measurements made while the scorecard's
comparison with a saved baseline and the interval beside its scores were
developed. The interval was changed on those measurements, which included
scores of the last-value forecaster, before the plan was registered. No mutant
and no Study 2 bug was executed on a new learner, the margin simulation was
not run, neither predictability estimator was run on a scorecard cell, and no
scorecard of a Part 6 model was made other than those of the two built-in
baselines, at seeds outside this protocol.

### The run and its records

The study was run on 17 November 2026 on Linux (x86-64, four cores) with
Python 3.13 and the study lock, from the protocol as registered. Each part's
record is a folder of `experiments/results-v3/`:

| Folder | Files |
|---|---|
| `mutants/` | `manifest.json` and the 136 mutant sources, committed before any was executed |
| `part-1/`, `part-2/` | `results.json`, `funnel.csv` (counts at each step), `decisions.csv` (every source, condition and check), `clean.csv` (alarms on the clean pipeline), `warnings.csv` (warnings raised while fitting and forecasting); Part 2 also `shift_classification.csv` |
| `part-3/` | `results.json`, `variants.csv`, `conditions.csv`, `suites.csv`, `p-values.csv` |
| `part-4/` | `results.json`, `cells.csv` (verdict counts of the 240 cells) |
| `part-5/` | `results.json`, `table.csv` (one row per scorecard cell), `series.csv` |
| `zoo/` | four files per model and `zoo-scorecards.pdf`, the figure drawn from them |

Every folder also has `manifest.json` (digests of the protocol, the results,
the frozen inputs and the source files, and the library versions) and
`runtime.json` (wall time and thread pools). `tests/test_study3_records.py`
checks each record against its manifest, the protocol and the lock file, and
reads back every count stated below.

What the parts show:

1. **Generated mutants (Ridge).** Cosmic Ray generated 119 mutants of the
   82-line subject. 55 raise an error on every input, 20 of the other 64 are
   equivalent on the 24 fixed inputs, and 44 are not. Of the 16 catalogue
   mutants one is equivalent, and none of the four Study 2 bugs is. The table
   counts the mutants that each check caught in at least one of the six
   conditions.

   | Check | Cosmic Ray (44) | Catalogue (15) | Study 2 bugs (4) |
   |---|---|---|---|
   | Output checks | 3 | 0 | 0 |
   | Common shift check | 22 | 2 | 1 |
   | Per-series shift check | 22 | 4 | 2 |
   | Second implementation | 44 | 15 | 4 |
   | Calibrated loss alarm | 37 | 13 | 4 |
   | Answer-key alarm | 40 | 14 | 4 |
   | Paired check against the clean pipeline | 43 | 15 | 4 |

   No deterministic check alarmed on the clean pipeline (332 cases in each
   condition). At the 5% level the loss alarm fired on 29 of the 558 clean
   evaluation cases and the answer-key alarm on 33. The equivalent mutants
   were run on all evaluation cases as well, and none differed from the clean
   pipeline there. No unit reached its time limit.
2. **Learners beyond Ridge.** The classification on the clean pipeline
   gave every learner the class that the protocol expected (16 cases in each
   of the six conditions):

   | Learner | Public shift check: PASS | Calibrated shift check: FAIL | Class |
   |---|---|---|---|
   | Ridge with global scaling (control) | 96 of 96 | not needed | exact |
   | Histogram gradient boosting | 96 of 96 | not needed | exact |
   | Window-standardized perceptron | 0 of 96 | 4 of 96 (binomial p = 0.71) | in distribution |
   | Perceptron on raw lags | 0 of 96 | 51 of 96 (p < 0.001) | not applicable |

   For each learner the first 84 Cosmic Ray mutants of the seeded order were
   screened to find 12 that run and change a forecast: 24 raise an error on
   every input and 48 are equivalent, because scaling is off for these
   learners and a mutant in the scaling code changes nothing. The same 12
   were taken for all three learners. Of the 16 catalogue mutants 4 are
   equivalent and of the four Study 2 bugs one is, as the protocol expected.
   Each cell counts the mutants that the check caught in at least one
   condition, as Cosmic Ray (of 12) / catalogue (of 12) / Study 2 bugs (of 3).

   | Check | Gradient boosting | Window-standardized perceptron | Perceptron on raw lags |
   |---|---|---|---|
   | Output checks | 3 / 0 / 0 | 3 / 0 / 0 | 3 / 0 / 0 |
   | Common shift check | 6 / 0 / 0 | not run | not run |
   | Per-series shift check | 6 / 2 / 1 | not run | not run |
   | Calibrated shift check | not run | 0 / 0 / 0 | not run |
   | Calibrated loss alarm | 8 / 11 / 3 | 5 / 8 / 3 | 8 / 6 / 3 |
   | Answer-key alarm | 9 / 12 / 3 | 5 / 7 / 3 | 8 / 7 / 3 |
   | Paired check against the clean pipeline | 11 / 12 / 3 | 6 / 9 / 3 | 11 / 12 / 3 |

   The calibrated shift check kept its level and caught no mutant. On the
   clean window-standardized pipeline it returned FAIL on 27 of the 558
   evaluation cases (4.8%, 95% interval 3.3% to 6.9%) and ERROR on none. On
   the 25 evaluated mutants it returned FAIL on 608 of the 12,648 cases it
   decided (4.8%); 5 of the 136 pairs of a mutant and a condition have an
   unadjusted p-value below 0.05, and none is left after Holm's correction.
   With this learner the paired check also caught fewer Cosmic Ray and
   catalogue mutants than with the other two: 6 and 9 of 12, against 11 and
   12 for both.

   No deterministic check alarmed on a clean pipeline (332 cases for each
   learner and condition). At the 5% level, on the 558 clean evaluation cases
   of gradient boosting, the window-standardized perceptron and the raw
   perceptron, the loss alarm fired on 30, 28 and 29 and the answer-key alarm
   on 17, 32 and 22. The equivalent catalogue and Study 2 sources were run on
   all evaluation cases as well, and none differed from the clean pipeline
   there. No unit reached its time limit or crashed, and the run of 5.5 hours
   on four workers was not interrupted.

   ConvergenceWarning is the only warning that the learners raised
   (`warnings.csv`). The window-standardized perceptron stopped at its
   iteration limit on every clean case, on all 41 fits of each evaluation
   case (22,878 warnings on 558 cases). The raw perceptron did so on 888 of
   its 1,992 clean calibration and evaluation cases. Gradient boosting and
   Ridge raised none. In Part 1 the clean pipeline raised no warning; 191 of
   the 6,528 screening records of its mutants carry a RuntimeWarning.
3. **Two questions about a suite.** Of 93 clean suites, 17 have at least one
   unadjusted rejection among their four checks, 2 have a Holm rejection and
   Fisher's combination flags 5. With the dependence between series shuffled,
   all three flag every suite; on the one condition where the shuffle changes
   nothing, the unadjusted check rejects in 4 suites and Holm in 3. With the
   spread reduced to 0.75, Fisher flags all 93 suites, while Holm rejects at
   least one condition in 68 and a given condition in 14 to 45.
4. **A joint-spread check with a margin.** With a true distance exactly on the
   margin, each one-sided verdict was meant to occur in at most about 5% of
   repetitions. FAIL does: between 0 and 29 of 381 repetitions in the 32 such
   cells, and its 95% interval lies above 5% in one cell. WITHIN_TOLERANCE
   does not: between 5 and 111 of 381, with the interval above 5% in 16 of
   the 32 cells, most of them with an inflated spread or in the two
   autocorrelated conditions; with the distance 10% beyond the margin it was
   still returned in up to 95 of 381 repetitions. The expectation written in
   the protocol for this verdict was therefore wrong, and the decision cannot
   be used to declare a spread close enough at 64 or 256 paths. At a distance
   of zero, WITHIN_TOLERANCE was returned in 670 of 1,524 repetitions at 64
   paths and 1,482 at 256 with the margin 0.25, and in 1,443 and 1,524 with
   the margin 0.4375.
5. **Predictability estimators against the exact floor.** The median ratio of
   the spectral bound to the exact unavoidable error lies between 0.86 and
   0.98 on the eight stationary rows, as expected. It is above 1 on the four
   rows with a trend (24.0 and 24.3 at low noise, 2.35 and 2.47 at high
   noise), as expected. On the seasonal rows (1.12 and 1.03) and on the
   level-shift rows (0.83) it is close to 1, where the protocol expected it
   to be off; computed from the history alone, the level-shift ratio is 81.1
   at low noise. On the random-walk rows, which have no stationary spectrum,
   it is 0.85 and 0.83, and 2.46 and 1.52 with drift. Weighted permutation
   entropy is highest on the pure-noise and fixed-level rows, and its rank
   correlation with the scale-free floor is 0.72 (0.26 with the floor
   itself).
6. **Scorecards of models people use.** The eight local models are scored; the
   Chronos scorecard needs a machine that can reach its weights. No check
   failed for any model. Each scorecard shows where its model stops, beside
   the row's attainable mark. At low noise, ETS scores 35 on irregular cycles
   (mark 93), gradient boosting on lags scores 5 on a straight-line trend
   (mark 95) and 93 on a sudden level shift (mark 97), Theta scores 22 on
   that level shift, and the perceptron on lags scores 4 on a random walk
   (mark 100). No ranking of the models is claimed.

What differs from the plan:

- **Order.** The protocol lists Parts 1 to 6 in order. They were run as
  Parts 3, 5, 4, the zoo, 1 and 2. No part reads the output of another.
- **Threads.** Three minutes into Part 2 its processes were found to use four
  threads each. The limit of one thread was set with threadpoolctl before the
  numerical libraries were loaded, and threadpoolctl limits only loaded
  libraries. The same held for the first run of Part 1 and for the gradient
  boosting model of the zoo. The run was stopped, the runners were changed
  (`experiments/study3_threads.py`), and Part 1 and all eight zoo models were
  run again under the rule. Every result file of the second runs is identical
  to the first, byte for byte. The second runs are the ones recorded, and each
  unit's last line now lists its thread pools. Parts 3, 4 and 5 kept the rule
  from the start (their `runtime.json`).
- **Continuing a stopped run.** Part 2 writes its record at the end of a run
  of several hours. Before it started, the runner got `--resume`. It was
  checked on Part 1: stopped on purpose after eight minutes and continued, it
  wrote a `results.json` identical to that of the uninterrupted run.
- **Where the mutants were run from.** Parts 1 and 2 were run from the commit
  that added the mutant sources and their digest, before that commit was
  merged. The protocol asks that the set be committed before any mutant is
  executed.
- **Warning counts.** The protocol asks that ConvergenceWarnings be counted
  and reported. Each unit wrote the count with every case, but the runner did
  not add the counts to `results.json`. `experiments/study3_warnings.py`
  tallied them afterwards from the unit records into `warnings.csv`. The unit
  records are not kept in the repository, and the manifests do not list this
  table.
- **Commit named by the zoo records.** The zoo manifests name commit 6c729e8:
  the registered commit plus the registration record, the mutant sources and
  the two runner changes above. The package (`timegym/`) is the same in both.
