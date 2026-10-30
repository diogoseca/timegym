# Reproduce the controlled pilot

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-study.txt
python -m pip install --no-build-isolation --no-deps -e .
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.run_study
MPLCONFIGDIR=/tmp/timegym-matplotlib python -m experiments.render_results
python paper/build_standalone.py
```

The exact environment lock records the tested Python 3.13 environment; supported
Python versions can instead install `pip install -e '.[dev]'`. Packages and the
public dataset are available locally after setup; the experiment itself uses no
network services or paid compute.

`protocol.json` defines the pilot before inspecting its evaluation results.
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
outputs; this illustrates a restricted numerical comparator, not superiority to
properly validated evaluation frameworks.

The separate model-level fault omits inverse standardisation in a historical-mean
forecast. Its repair applies the training location and scale exactly once. The
location-equivariance relation is explicitly declared for this model. The same
repair is illustrated at four ordered AirPassengers origins, without treating
them as independent replications. It does not make this weak baseline a good
forecasting model.

Outputs:

- `results/results.json`: individual cases, scores, confidence intervals and summaries.
- `results/manifest.json`: protocol, dependency versions, base Git commit, current source
  fingerprint, and dataset hash. The source fingerprint covers uncommitted v2 code.
- `results/example-*.json`: machine-readable counterexamples.
- `../paper/figures/*.png` and `../paper/figures/*.pdf`: figures rendered from the recorded results.
- `../paper/generated-results.tex`: manuscript tables derived from the same JSON.

For a numerical repeat, run `python -m experiments.run_study --output /tmp/timegym-repeat`
and compare `results.json`. There are no timestamps or runtime values in numerical
results. A platform-dependent tolerance may be needed on other BLAS/NumPy versions.
The exact local repeat is checked during development.

To reproduce one selected controlled counterexample without running the full
study, use `python -m experiments.reproduce_fault --mutation dependence_shuffle --seed 10000`.
This reconstructs its saved scenario, ensemble and mutation seeds, then compares
the contract and covariance decisions with the original record. Its successful
exit means reproduction matched, even when the forecast correctly receives FAIL.

Scope limits: no held-out generator families, representative real-bug corpus,
foundation models, sktime-conformance comparator, or broad real-data transfer study.
The full research roadmap is deferred, rather than implied by this pilot.
