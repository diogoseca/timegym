# Releases and compatibility

TimeGym is a PyPI package. The Dockerfile is a development image: it installs
`requirements-study.txt` first, then the package with `--no-deps`, preserving
the recorded scientific stack. The canonical study reproduction still uses
Python 3.13 as recorded in [the paper guide](../paper/README.md). Updating the
Docker interpreter does not rewrite that historical lock or establish that a
new interpreter reproduces old numerical records.

<!-- BEGIN PYTHON SUPPORT -->
Python 3.11–3.15 (`>=3.11,<3.16`), following pandas support.
<!-- END PYTHON SUPPORT -->

Runtime ranges are NumPy `>=1.26,<3`, pandas `>=2.2,<4`, and SciPy `>=1.11,<2`.
The older-stack CI job preserves pandas 2.2 coverage. Current Python support
follows [pandas' policy](https://pandas.pydata.org/docs/development/policies.html)
and its [published package metadata](https://pypi.org/project/pandas/), rather
than raising every dependency floor to the study lock.

## Version identity

`__version__` in notebook 12 is the sole package version source; flit reads its
export in `timegym/__init__.py`. Versions are `major.features.fixes`: the second
and third numbers count merged pull requests labelled `feature` and `fix`.
Neither counter resets the other, so a feature after 1.2.7 gives 1.3.7 and a fix
after that gives 1.3.8. Only Diogo Seca starts a new major version, which
restarts both counters at 0. The root `AGENTS.md` records these rules for future
contributors.

Pull requests do not edit the version, `CITATION.cff` or the changelog heading;
the release workflow writes them. Label a pull request `feature` or `fix` when it
should produce a release, and leave documentation, paper and tooling changes
unlabelled. A pull request may add user-facing notes under a `## Unreleased`
heading at the top of `CHANGELOG.md`. When that section has text it becomes the
release notes; otherwise the notes list the labelled pull request titles. To
check the release metadata by hand:

```bash
.venv/bin/python tools/sync_citation.py --check --tag v1.1.0
.venv/bin/python paper/build_standalone.py --check
```

The citation renderer derives the README BibTeX/APA and historical paper citation
from CFF. It fails on mismatched package/changelog versions or dates. The
standalone manuscript uses the same version/date in its title block; the
manuscript title carries no version number of its own.
The release workflow commits generated sources along with their inputs before tagging.

## One-time PyPI setup

Nothing reaches PyPI until a `v*` tag is pushed: `release.yml` is the only
publishing path, and branch pushes and merges never run it. The package uses
[trusted publishing](https://docs.pypi.org/trusted-publishers/), so no API token
or repository secret is involved; an account-wide `twine` token is not needed.

1. Sign in to PyPI. While `timegym` has never been published, open
   [Your account → Publishing](https://pypi.org/manage/account/publishing/) and
   add a pending publisher with PyPI project name `timegym`. Once the project
   exists, the publisher lives on the project's **Manage → Publishing** page. A
   pending publisher does not reserve the name; the first successful upload does.
2. Choose the **GitHub** publisher with owner `diogoseca`, repository
   `timegym`, workflow name `release.yml`, and environment name `pypi`.
   The filename is not `.github/workflows/release.yml`.
3. In the GitHub repository, open [Settings → Environments → New
   environment](https://github.com/diogoseca/timegym/settings/environments/new),
   name it `pypi`, and save. Under **Deployment protection rules** tick
   **Required reviewers** and add the author. Under **Deployment branches and
   tags** choose **Selected branches and tags** and add a rule of type **Tag**
   with pattern `v*`. The workflow's publish job uses this environment and
   requests `id-token: write` for OIDC authentication.
4. Under **Settings → Actions → General**, allow the actions used by these
   workflows and enable **Allow GitHub Actions to create and approve pull
   requests** for the compatibility bot. Workflows declare their own minimal
   permissions. The `feature`, `fix` and `python-compatibility` labels are
   created the first time a pull request uses them.
5. `tag-release.yml` pushes its version commit straight to `main`. If a branch
   protection rule or ruleset on `main` requires pull requests or status checks,
   GitHub rejects that push and the job stops before anything is tagged. Let
   the workflow through by removing those requirements or giving it a bypass.

These fields must match the [PyPI trusted publisher configuration](https://docs.pypi.org/trusted-publishers/adding-a-publisher/).
For a new project, follow [PyPI's pending publisher setup](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/).

## One-time Zenodo setup and DOI lifecycle

1. Sign in to Zenodo, connect the author's GitHub account, and grant access to
   `diogoseca/timegym`. Under the profile menu choose **GitHub**, select **Sync
   now**, find `diogoseca/timegym`, and enable its slider. Do this before the
   first GitHub release. See [Zenodo's repository setup](https://help.zenodo.org/docs/github/enable-repository/).
2. Keep `CITATION.cff` at the repository root. Avoid adding `.zenodo.json` unless
   deliberately switching metadata sources: it overrides CFF. The CFF includes
   the software version/date and the 2021 arXiv manuscript as a reference, not
   as the preferred software citation. See [Zenodo's CFF handling](https://help.zenodo.org/docs/github/describe-software/citation-file/).
3. After the workflow publishes the GitHub release, wait for it to appear in
   Zenodo's GitHub page. Open the record, confirm its version/date and archived
   source, and copy its **version DOI** and **concept DOI**. Each release gets a
   version DOI; the concept DOI identifies the evolving collection. Inspect
   the record's file list; do not assume GitHub release attachments were ingested.
   The standalone PDF is attached to the GitHub release and its source is in the
   archived repository. If the record lacks the PDF, add it through Zenodo's
   supported record/file editing process or archive it separately and relate it
   to the software record. [Zenodo release processing](https://help.zenodo.org/docs/github/archive-software/github-upload/).
4. Put the actual version DOI in the root `doi:` field of `CITATION.cff` in a
   follow-up metadata PR for that same package version, rerun the citation
   renderer, and add both DOI links to the GitHub release notes. Never put the
   2021 arXiv DOI in the software DOI field. Clear the previous version's DOI
   when preparing another release; the compatibility bot does this automatically.

A DOI is minted only after the release is archived, so the first immutable tag
cannot contain its own newly minted DOI through the GitHub integration alone.
This preparation deliberately omits a software DOI until registration. The
follow-up CFF correction does not rewrite the tag or the PyPI files. If a DOI
must be inside the immutable artifacts, reserve a DOI in a manual Zenodo deposit
before building and use that deposit as the release archive instead of creating
a second archive through the GitHub integration. Do not fabricate an identifier.
Replace the arXiv manuscript manually at substantive milestones; this workflow
does not submit or replace arXiv papers.

## Automatic releases

Nobody tags by hand. Merging a pull request into `main` runs `tag-release.yml`.
It finds the latest `v*` tag and counts the pull requests labelled `feature` or
`fix` merged since then. When there is at least one, `tools/release_version.py`
writes the next version into notebook 12, `CITATION.cff` (dated the day of the
run) and `CHANGELOG.md`. The job regenerates the package module, README
citation and standalone manuscript, runs the release-metadata checks, commits
`Release X.Y.Z` to `main` with the annotated tag `vX.Y.Z`, and dispatches
`release.yml` for that tag, because a tag pushed with `GITHUB_TOKEN` starts no
workflow by itself. Unlabelled merges release nothing; their changes ship with
the next labelled one.

To release pending changes without a new merge, or to start a major version,
open **Actions → Version and tag merged changes → Run workflow** and tick
**major** for a major version. The job refuses to run when `__version__` on
`main` differs from the latest tag, which catches hand-edited versions. Several
quick merges queue behind each other; each run counts everything since the
latest tag, so later runs find nothing left and stop.

`release.yml` runs the complete reusable `checks.yml` workflow against the tag,
checks the version identity, builds sdist/wheel, and compiles the standalone
manuscript with LaTeX. The `pypi` job publishes via trusted publishing after
its environment review. Only then does the GitHub job attach the distributions,
PDF and standalone TeX to a draft release and publish it, triggering Zenodo.
The release date is the reviewed CFF date, not an unrecorded build-clock value.
All artifacts identify the tagged source; study records retain their original
execution provenance.

For an interrupted run, rerun failed jobs from the same tag. PyPI's action skips
already uploaded files, and the GitHub step reuses an existing draft/release.
Never replace an existing version's package files or move its tag. A code or
packaging correction requires a new fix counter. A manually dispatched release
must select a version tag; branch dispatches do not publish.

## Automatic Python compatibility proposals

Every Monday at 07:23 UTC, or on manual dispatch, `python-compatibility.yml`
checks stable releases listed by [Python.org](https://www.python.org/downloads/)
and the latest pandas release's Python classifiers, requirement and CPython
wheel availability. Prereleases and a new Python minor without pandas wheels
are deferred. A changed supported range updates classifiers, `requires-python`,
CI, the Docker base and the marked documentation ranges. The proposal leaves
the version alone and is labelled `fix`, so merging it releases the next fix
version through `tag-release.yml`. The study lock is retained.

The proposal runs tests, notebooks, examples, replay, export/citation/paper
checks, strict docs, distribution validation, and the pinned development image
on the newest interpreter before opening a PR. A failed dependency installation
or check leaves the scheduled run red for inspection and opens no passing PR.
A bot-created PR may not trigger another workflow using the default token;
validation therefore runs in the scheduled job itself.

Merging the proposal publishes like any other `fix` pull request, and the PyPI
environment review still applies.
