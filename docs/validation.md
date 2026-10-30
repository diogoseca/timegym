# Validation record

Checked locally on 28 October 2026 for the unreleased 0.2.0 development artifact.
This record describes executed checks; it is not a claim of universal fault
detection or validation on every supported platform.

| Check | Observed result |
|---|---|
| Package tests, Python 3.13.2 | 506 passed |
| Package tests, Python 3.12 with older scientific dependencies | 506 passed; 13 upstream Matplotlib/pyparsing deprecation warnings |
| Fresh-kernel notebooks | 24 executed; 295 code cells and 42 figures |
| Notebook-to-package export | All 24 modules match their saved source cells |
| Historical notebook preservation | All 11 archived file hashes match the inventory |
| Documentation | Strict build passes with local links resolved; original source paths registered for live reload |
| Distributions | Wheel and source archive build; both pass metadata validation |
| Clean installed wheel | Core dependencies only; point/joint forecasts, joint controlled financial processes, reports, replay, retained runs, and CLI demo pass outside the checkout |
| Source archive | Includes all 24 active notebooks, paper, study records, citation metadata and build tooling; extracted source passes export drift check |
| Original study | Numerical JSON unchanged; historical execution evidence retained in the manifest |
| Protocol-2 study | Frozen numerical record; offline schema migration described below. Future repeats are compared by the CI research-smoke job |
| Manuscript | Generated standalone source compiles successfully |

The multivariate extension covers correlated composed processes, vector Merton
jumps, CCC-GARCH, shared hidden regimes and regime-dependent jumps. Validation
includes direct joint cumulants, independent hidden-state-path enumeration,
a multivariate forward-filter calculation, Gauss-Hermite integration of
second-step GARCH cross-moments, conditional simulation, shape and parameter
boundaries, random-stream separation, report replay and retained scalar-case
outputs. General GARCH cross-covariance is explicitly estimated; its joint
conditional sampler remains exact. Approximate regime/jump filtering stays
unsupported for the exact-reference covariance diagnostic.

All 24 notebook outputs and five main SVG/TikZ figures were refreshed. The new
joint tutorial shows four multi-target mechanisms, preserved marginals with
broken cross-series trajectory pairing, changed aggregate uncertainty, and
repair verification. Observations remain dots and point forecasts remain
unconnected crosses. These software checks and demonstrations do not expand
either original fault-study sample or establish effectiveness on real stocks.

The older compatibility environment uses NumPy 1.26.4, pandas 2.2.3,
SciPy 1.11.4, scikit-learn 1.4.2, and Matplotlib 3.8.4. It exercises an older
compatible stack, not every exact lower bound or dependency combination.
The modern scientific environment is recorded in `requirements-study.txt`
and the study manifests. The clean wheel environment contains only declared
core dependencies; its smoke check also actively rejects optional imports.

The protocol-2 numerical record has SHA256
`1b386768628dd09981ac08f1bf91bb567c678e41e483999a9ac300f7511971f7`.
The pilot numerical record has SHA256
`cb8b53b50f30c6b6bfcdf945cd84640fcbeb44ed49588e3ecf87dcc62a59b014`.
On 30 October 2026 an offline schema migration removed 384 verified duplicate
repaired records and unused alarm-difference t intervals; no study was rerun.
All retained measurements and paired loss summaries stayed unchanged. The
original supplement hash was
`ba9b2e90145ca50d4049e469869ea575a95c34d6d80fd16d0946cb8ae9395bc7`.
Both manifests retain original provenance in `record_migration`.
Its manifest fingerprints individual package, runner, fixture, and protocol
files. The original pilot records a combined source digest of
`dece174f629b2647011570775af43f8a657f6a477ae10108972ed135b56f6549`.
These historical source fingerprints identify the earlier execution, not the
current runner after migration. The `research-smoke` CI job now compares fresh
pilot and supplement JSON hashes, warns on mismatches, and retains the actual
Linux hashes in the `research-linux-hashes` artifact. It has not been executed
from this checkout, so no Linux repeatability result is claimed here.
Timing is recorded separately and is not a benchmark claim.

## Remaining release and research work

CI is configured for Python 3.10–3.13, the older compatibility stack,
installed wheels on Linux/macOS/Windows, all notebooks, documentation, and both
studies. These remote jobs have not been executed from this checkout. Local
validation was on macOS; do not infer Windows or Linux results from it.

Before publishing a first v2 release, run those jobs against the final immutable
revision and complete author review of the manuscript, attribution, examples,
and release metadata. No package release or manuscript submission was made.

For a stronger empirical paper, the next evidence should come from independently
selected defects in other forecasting implementations and additional real tasks.
The current studies support the specified workflow and expose useful blind spots;
they do not estimate general debugging effectiveness. See the
[claim ledger](../paper/claims.md) and [publication strategy](publication-strategy.md).
