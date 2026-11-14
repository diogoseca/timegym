# Repository guidance

TimeGym is notebook-first. Every Python file under `timegym/` is generated from
marked implementation cells owned by `source/source-map.json`. Edit the
owning notebook, preserve cell IDs and `metadata.timegym.role`, save it, then
run `.venv/bin/python tools/export_notebooks.py` and its `--check` mode. Never
edit generated modules by hand. Read `tools/README.md` and `source/README.md`
before changing library code. The 15 source notebooks in `source/` are the
library; the 12 notebooks in `tutorials/` only use it, the way a user would. Paper result fragments and `paper/timegym.tex`
are generated too; edit their inputs and run their builders.

Create the environment with `python3 -m venv .venv` and `.venv/bin/pip install -e '.[dev]'`; use `.venv/bin/python` for every command. Add `.[notebooks]` to execute notebooks, `.[docs]` for documentation, and `.[release]` for distribution checks.

## Version policy

Versions are `major.features.fixes`, as `docs/release.md` describes; a feature never resets the fix counter, and only Diogo Seca changes the major version. Do not edit `__version__` (source notebook 12), the CITATION.cff version/date or the leading CHANGELOG heading in a pull request: `tag-release.yml` writes them. Label every pull request you open `feature` or `fix` when it changes the package, or leave it unlabelled for documentation, paper and tooling work. Put user-facing notes under `## Unreleased` at the top of CHANGELOG.md. Never invent a DOI.

## Compatibility policy

Follow pandas when choosing current Python and scientific dependency support.
Prefer the latest stable Python and dependency releases supported by pandas;
when uncertain, use pandas' support policy and release timing. Keep explicit
NumPy, pandas and SciPy major-version ceilings until compatibility is reviewed.
The scheduled compatibility workflow proposes coordinated metadata, CI, Docker
and documentation updates and tests them before opening a pull request. Keep
the historical study lock and its Python version separate from current support.
A successful local check is not evidence that another OS or Python version passed.

Publication follows merges: the release workflow proposes each version as a
"Release X.Y.Z" pull request and tags and publishes it after the owner merges
it. Do not edit or label that pull request. Do not publish, push tags, or change
external PyPI/Zenodo settings as part of ordinary implementation work.

## Working agreements

Work on a branch and hand over a pull request; never push to `main`, never
push tags, never force-push. Before handoff run, with the venv interpreter,
`tools/export_notebooks.py --check`, `python -m pytest -q`,
`python -m mkdocs build --strict` followed by `tools/check_site_links.py site`,
`paper/build_standalone.py --check`,
`tools/sync_citation.py --check`, and `tools/check_notebooks.py` on every
notebook you touched, with `--write` when outputs change, because the docs
site renders saved outputs. When the manuscript changes, also compile
`paper/timegym.tex` with latexmk as `paper/README.md` describes. Recount
every stated count (source notebooks, generated modules, active notebooks,
tests) when it changes. Commit messages
have an imperative subject and no trailers: no `Co-Authored-By`, no session
links, no tool attribution of any kind. Delegate to subagents on Opus; the
session's own model is for orchestration and review.
