# Releases and compatibility

TimeGym is a PyPI package. The Dockerfile is a development image: it installs
`requirements-study.txt` first, then the package with `--no-deps`, preserving
the recorded scientific stack. The canonical study reproduction uses
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
and third numbers count merged pull requests labeled `feature` and `fix`.
Neither counter resets the other, so a feature after 1.2.7 gives 1.3.7 and a fix
after that gives 1.3.8. Only Diogo Seca starts a new major version, which
restarts both counters at 0.

Pull requests do not edit the version, `CITATION.cff` or the changelog heading;
the release workflow writes them in its release pull request. Label a pull
request `feature` or `fix` when it should produce a release, and leave
documentation, paper and tooling changes unlabeled. A pull request may add
user-facing notes under a `## Unreleased` heading at the top of `CHANGELOG.md`.
When that section has text it becomes the version's CHANGELOG entry; otherwise
the entry lists the labeled pull request titles. GitHub generates the
release-page notes. To check the release metadata by hand:

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

Nothing reaches PyPI without a `v*` tag: `release.yml` is the only publishing
path, and it runs only for a tag, which `tag-release.yml` creates and
dispatches when its release pull request is merged. The package uses
[trusted publishing](https://docs.pypi.org/trusted-publishers/), so no API token
or repository secret is involved; an account-wide `twine` token is not needed.

1. Sign in to PyPI and open the `timegym` project's **Manage → Publishing** page.
2. Choose the **GitHub** publisher with owner `diogoseca`, repository
   `timegym`, workflow name `release.yml`, and environment name `pypi`.
   The filename is not `.github/workflows/release.yml`.
3. In the GitHub repository, open [Settings → Environments → New
   environment](https://github.com/diogoseca/timegym/settings/environments/new),
   name it `pypi`, and save. Under **Deployment branches and tags** choose
   **Selected branches and tags** and add a rule of type **Tag** with pattern
   `v*`. Leave **Required reviewers** unticked: the `authorize` job in
   `release.yml` lets a release publish only when the repository owner merged
   or approved its release pull request, pushed its tag, or started or re-ran
   its run. Tick **Required reviewers** and add the author instead if every
   release should wait for a click; that is also the only gate that a
   collaborator with write access cannot get around by editing the workflow.
   The workflow's publish job uses this environment and requests
   `id-token: write` for OIDC authentication.
4. Under **Settings → Actions → General**, allow the actions used by these
   workflows and, under **Workflow permissions**, enable **Allow GitHub Actions
   to create and approve pull requests** for the release pull request and the
   compatibility bot. Workflows declare their own minimal permissions. The
   `feature`, `fix` and `python-compatibility` labels are created the first time
   a pull request uses them.
5. The ruleset on `main` requires pull requests, so `tag-release.yml` never
   pushes to `main`: it proposes each version as a pull request and pushes
   only the release branch and the tag, which the ruleset does not cover.

These fields must match the [PyPI trusted publisher configuration](https://docs.pypi.org/trusted-publishers/adding-a-publisher/).

## One-time Zenodo setup and DOI lifecycle

1. Sign in to Zenodo, connect the author's GitHub account, and grant access to
   `diogoseca/timegym`. Under the profile menu choose **GitHub**, select **Sync
   now**, find `diogoseca/timegym`, and enable its slider. Do this before the
   next GitHub release; v1.1.0 is already published. See [Zenodo's repository setup](https://help.zenodo.org/docs/github/enable-repository/).
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
   2021 arXiv DOI in the software DOI field. `tools/release_version.py` removes
   the previous version's DOI when `tag-release.yml` writes the next version.

A DOI is minted only after the release is archived, so the first immutable tag
cannot contain its own newly minted DOI through the GitHub integration alone.
`CITATION.cff` therefore has no software DOI until Zenodo registers one. The
follow-up CFF correction does not rewrite the tag or the PyPI files. If a DOI
must be inside the immutable artifacts, reserve a DOI in a manual Zenodo deposit
before building and use that deposit as the release archive instead of creating
a second archive through the GitHub integration. Do not fabricate an identifier.
Replace the arXiv manuscript manually at substantive milestones; this workflow
does not submit or replace arXiv papers.

## Automatic releases

Nobody tags by hand. Merging a pull request into `main` runs `tag-release.yml`.
It counts the pull requests labeled `feature` or `fix` merged since the tag of
the version on `main`. When there is at least one, `tools/release_version.py`
writes the next version into notebook 12, `CITATION.cff` (dated the day of the
run) and `CHANGELOG.md`. The job regenerates the package module, README
citation and standalone manuscript, runs the release-metadata checks, commits
`Release X.Y.Z` on the `release` branch and opens the **Release X.Y.Z** pull
request. Unlabeled merges release nothing by themselves; their changes ship
with the next release.

The release pull request stays open until you merge it. Every merge into `main`
rebuilds it from the new `main`, so its version, notes and date always count
everything merged so far; only the workflow writes the `release` branch. Leave
it unlabeled. Merging it, with any merge method, runs `tag-release.yml` again:
the job checks the release commit's metadata, tags that commit `vX.Y.Z` and
dispatches `release.yml`, because a tag pushed with `GITHUB_TOKEN` starts no
workflow. The tag marks the release commit itself, so it holds exactly what the
release counted; anything merged after the last rebuild counts toward the next
version. Because the bot opens the release pull request, GitHub holds its
checks until you click **Approve workflows to run** on the pull request, and the
Claude review skips it. Running them is optional: `release.yml` runs the full
checks against the tag before anything is published.

To rebuild the release pull request without a new merge, or to prepare a major
version, open **Actions → Version and tag merged changes → Run workflow** and
tick **major** for a major version. The job refuses to run when the version on
`main` has no matching tag, which catches hand-edited versions. Several quick
merges queue behind each other. If a merge lands while a run is preparing the
release pull request, that run stops without failing and leaves the release
pull request to the newer merge's run, which counts both. When nothing labeled
is waiting, the job closes an open release pull request. A merge from a fork
runs with a read-only token, so it cannot prepare a release; run the workflow
by hand after merging one.

`release.yml` runs the complete reusable `checks.yml` workflow against the tag,
checks the version identity, builds sdist/wheel, and compiles the standalone
manuscript with LaTeX. Its `authorize` job checks that the repository owner
merged or approved the release pull request (an approval counts only on the
release commit), pushed the tag, or started or re-ran the run. A release
pull request merged by anyone else stops there, and the owner can publish it by
re-running the failed job. The `pypi` job then publishes via trusted publishing,
so merging the release pull request yourself publishes it a few minutes later.
Only then does the GitHub job attach the distributions, PDF and standalone TeX
to a draft release and publish it, triggering Zenodo.
The release date is the `date-released` that `tag-release.yml` wrote to
`CITATION.cff` when it last rebuilt the release pull request.
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
the version alone and is labeled `fix`, so merging it adds one to the fix
counter in the next release pull request. The study lock is retained.

The proposal runs tests, notebooks, examples, replay, export/citation/paper
checks, strict docs, distribution validation, and the pinned development image
on the newest interpreter before opening a PR. A failed dependency installation
or check leaves the scheduled run red for inspection and opens no passing PR.
A bot-created PR may not trigger another workflow using the default token;
validation therefore runs in the scheduled job itself.

Like any other change, it reaches PyPI only through a release pull request
that the `authorize` job accepts.
