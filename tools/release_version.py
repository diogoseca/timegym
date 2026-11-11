"""Compute and write the next release version from merged pull-request labels.

The version is ``major.features.fixes``: each merged pull request labelled
``feature`` adds one to the feature counter and each labelled ``fix`` adds one
to the fix counter. Neither counter resets the other. Only an explicit major
release starts the counters again at ``major+1.0.0``.
"""
from __future__ import annotations

import argparse
from datetime import date
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
FEATURE, FIX = 'feature', 'fix'
VERSION = re.compile(r'__version__ = "(\d+)\.(\d+)\.(\d+)"')
UNRELEASED = re.compile(r'^## Unreleased[^\n]*\n(.*?)(?=^## |\Z)', re.M | re.S)


def labels(pr: dict) -> set[str]:
    return {label if isinstance(label, str) else label['name'] for label in pr.get('labels', [])}


def next_version(current: str, prs: list[dict], *, major: bool = False) -> str | None:
    """Return the version after ``current``, or None when nothing is releasable."""
    match = re.fullmatch(r'(\d+)\.(\d+)\.(\d+)', current)
    if not match:
        raise ValueError(f'Version {current!r} must contain three integer counters')
    counters = list(map(int, match.groups()))
    if major:
        return f'{counters[0] + 1}.0.0'
    features = sum(FEATURE in labels(pr) for pr in prs)
    fixes = sum(FIX in labels(pr) for pr in prs)
    if not features and not fixes:
        return None
    return f'{counters[0]}.{counters[1] + features}.{counters[2] + fixes}'


def package_version(root: Path = ROOT) -> str:
    found = [VERSION.search(''.join(cell['source']))
             for cell in json.loads((root / 'source/12_package.ipynb').read_text())['cells']
             if cell['cell_type'] == 'code']
    found = [match for match in found if match]
    if len(found) != 1:
        raise ValueError('Expected one notebook-owned package version')
    return '.'.join(found[0].groups())


def release_notes(changelog: str, prs: list[dict]) -> tuple[str, str]:
    """Return the changelog without its Unreleased section, and the release notes.

    Hand-written notes under ``## Unreleased`` win; otherwise each labelled pull
    request contributes its title.
    """
    match = UNRELEASED.search(changelog)
    written = match[1].strip() if match else ''
    if match:
        changelog = changelog[:match.start()] + changelog[match.end():]
    if written:
        return changelog, written
    lines = [f"- {pr['title'].strip()} (#{pr['number']})" for pr in prs
             if labels(pr) & {FEATURE, FIX}]
    return changelog, '\n'.join(lines) or '- Start a new major version.'


def apply(root: Path, version: str, released: date, prs: list[dict]) -> None:
    """Write the version into notebook 12, CITATION.cff and CHANGELOG.md."""
    notebook = root / 'source/12_package.ipynb'
    data = json.loads(notebook.read_text())
    for cell in data['cells']:
        text = ''.join(cell['source'])
        if cell['cell_type'] == 'code' and VERSION.search(text):
            text = VERSION.sub(f'__version__ = "{version}"', text)
            cell['source'] = text.splitlines(keepends=True)
    notebook.write_text(json.dumps(data, ensure_ascii=False, indent=1) + '\n')
    citation = root / 'CITATION.cff'
    text = re.sub(r'^version: .*$', f'version: {version}', citation.read_text(), flags=re.M)
    text = re.sub(r'^date-released: .*$', f'date-released: {released.isoformat()}', text, flags=re.M)
    # A previous version DOI cannot identify this new version.
    text = re.sub(r'^doi: .*\n', '', text, flags=re.M)
    citation.write_text(text)
    changelog = root / 'CHANGELOG.md'
    text, notes = release_notes(changelog.read_text(), prs)
    if not text.startswith('# Changelog\n'):
        raise ValueError('CHANGELOG.md must start with "# Changelog"')
    rest = text[len('# Changelog\n'):].lstrip('\n')
    changelog.write_text(f'# Changelog\n\n## {version} — {released.isoformat()}\n\n{notes}\n\n{rest}')


def merged_prs(since: str, repository: str) -> list[dict]:
    """Pull requests merged into the checked-out branch after the ``since`` tag."""
    commits = subprocess.run(['git', 'log', '--first-parent', '--format=%H', f'{since}..HEAD'],
                             check=True, capture_output=True, text=True).stdout.split()
    prs = {}
    for sha in commits:
        found = json.loads(subprocess.run(
            ['gh', 'api', f'repos/{repository}/commits/{sha}/pulls'],
            check=True, capture_output=True, text=True).stdout)
        for pr in found:
            if pr.get('merged_at') and pr.get('merge_commit_sha') == sha:
                prs[pr['number']] = {'number': pr['number'], 'title': pr['title'],
                                     'labels': sorted(labels(pr))}
    return [prs[number] for number in sorted(prs)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--since', required=True, help='Tag of the previous release')
    parser.add_argument('--prs', type=Path, help='JSON list of pull requests instead of asking GitHub')
    parser.add_argument('--major', action='store_true', help='Start the next major version')
    parser.add_argument('--date', type=date.fromisoformat, default=date.today())
    parser.add_argument('--apply', action='store_true', help='Write the new version into the tree')
    parser.add_argument('--github-output', type=Path)
    args = parser.parse_args()
    current = package_version()
    if 'v' + current != args.since:
        parser.exit(2, f'Package version {current} is not the released {args.since}\n')
    prs = (json.loads(args.prs.read_text()) if args.prs
           else merged_prs(args.since, os.environ['GITHUB_REPOSITORY']))
    version = next_version(current, prs, major=args.major)
    if version and args.apply:
        apply(ROOT, version, args.date, prs)
    print(json.dumps({'current': current, 'version': version, 'pull_requests': prs}, indent=1))
    if args.github_output:
        with args.github_output.open('a') as output:
            output.write(f"version={version or ''}\nrelease={str(bool(version)).lower()}\n")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
