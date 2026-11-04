# Repository guidance

TimeGym is notebook-first. Every Python file under `timegym/` is generated from
marked implementation cells owned by `notebooks/source-map.json`. Edit the
owning notebook, preserve cell IDs and `metadata.timegym.role`, save it, then
run `.venv/bin/python tools/export_notebooks.py` and its `--check` mode. Never
edit generated modules by hand. Read `tools/README.md` and `notebooks/README.md`
before changing library code. Paper result fragments and `paper/timegym.tex`
are generated too; edit their inputs and run their builders.

Create the environment with `python3 -m venv .venv` and
`.venv/bin/pip install -e '.[dev]'`; use `.venv/bin/python` thereafter. Add
`.[notebooks]` to execute notebooks, `.[docs]` for documentation, and `.[release]`
for distribution checks. Before handoff run the exporter check, `python -m
pytest -q`, and `python -m mkdocs build --strict` when documentation changes.
Use the venv interpreter for each command.

## Version policy

The three integers in `M.X.Y` are independent counters, not a decimal number.
The leading zero in the former `0.X.Y` was the major version; this round starts
`1.0.0`. An implemented feature increments X; a fix increments Y. Increasing X
must not reset Y: for example, a feature after `1.2.7` becomes `1.3.7`, and a fix
then becomes `1.3.8`. Only Diogo Seca decides when to change the major version.
Do not infer a major bump from another project's versioning convention.

The authoritative version string is `__version__` in notebook 12's package cell,
exported to `timegym/__init__.py`; flit reads it dynamically. Update the matching
CITATION.cff version/date and the leading CHANGELOG heading, run
`tools/sync_citation.py`, then regenerate the standalone manuscript. Never
invent a DOI. Follow `docs/release.md` for DOI registration and publication.

## Compatibility policy

Follow pandas when choosing current Python and scientific dependency support.
Prefer the latest stable Python and dependency releases supported by pandas;
when uncertain, use pandas' support policy and release timing. Keep explicit
NumPy, pandas and SciPy major-version ceilings until compatibility is reviewed.
The scheduled compatibility workflow proposes coordinated metadata, CI, Docker
and documentation updates and tests them before opening a pull request. Keep
the historical study lock and its Python version separate from current support.
A successful local check is not evidence that another OS or Python version passed.

Publication is tag-driven after owner review. Do not publish, push tags, or
change external PyPI/Zenodo settings as part of ordinary implementation work.

## Working agreements

Work on a branch and hand over a pull request; never push to `main`, never
push tags, never force-push. Before handoff run, with the venv interpreter,
`tools/export_notebooks.py --check`, `python -m pytest -q`,
`python -m mkdocs build --strict`, `paper/build_standalone.py --check`,
`tools/sync_citation.py --check`, and `notebooks/check_notebooks.py` on every
notebook you touched, with `--write` when outputs change, because the docs
site renders saved outputs. When the manuscript changes, also compile
`paper/timegym.tex` with latexmk as `paper/README.md` describes. Recount
every stated count (source notebooks, generated modules, active notebooks,
tests) when it changes. Commit messages
have an imperative subject and no trailers: no `Co-Authored-By`, no session
links, no tool attribution of any kind. Delegate to subagents on Opus; the
session's own model is for orchestration and review.
