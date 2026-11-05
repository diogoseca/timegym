"""Propose stable CPython support when pandas advertises compatible wheels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import tomllib
from urllib.request import urlopen

from packaging.specifiers import SpecifierSet
from packaging.version import Version

ROOT = Path(__file__).resolve().parents[1]


def supported_versions(downloads: str, pandas: dict) -> list[str]:
    finals = {Version(v) for v in re.findall(r'>Python (3\.\d+\.\d+)<', downloads)}
    if not finals:
        raise ValueError('No stable CPython releases found on python.org/downloads')
    latest = max(finals)
    info = pandas['info']
    requirement = SpecifierSet(info['requires_python'])
    classifiers = [c.rsplit(' :: ', 1)[-1] for c in info['classifiers']
                   if re.fullmatch(r'Programming Language :: Python :: 3\.\d+', c)]
    versions = sorted({Version(v) for v in classifiers if Version(v) in requirement
                       and Version(v) <= latest})
    latest_minor = Version(f'{latest.major}.{latest.minor}')
    # A classifier plus a CPython wheel is stronger evidence than Requires-Python alone.
    wheel_tag = f'cp{latest.major}{latest.minor}'
    if latest_minor not in versions or not any(
        f'-{wheel_tag}-' in item['filename'] and not item.get('yanked', False)
        for item in pandas['urls']
    ):
        return []
    return [str(v) for v in versions]


def update(root: Path, versions: list[str]) -> bool:
    """Rewrite the supported range; the merge workflow versions the change as a fix."""
    if not versions:
        return False
    project_path = root / 'pyproject.toml'
    source = project_path.read_text()
    project = tomllib.loads(source)['project']
    current = [c.rsplit(' :: ', 1)[-1] for c in project['classifiers']
               if re.fullmatch(r'Programming Language :: Python :: 3\.\d+', c)]
    if current == versions:
        return False
    oldest, latest = versions[0], versions[-1]
    upper = f'3.{Version(latest).minor + 1}'
    requirement = f'>={oldest},<{upper}'
    source = re.sub(r'^requires-python = .*$', f'requires-python = "{requirement}"', source, flags=re.M)
    source = re.sub(r'^    "Programming Language :: Python :: 3\.\d+",\n', '', source, flags=re.M)
    source = source.replace('    "Programming Language :: Python :: 3",\n',
                            '    "Programming Language :: Python :: 3",\n' + ''.join(
                                f'    "Programming Language :: Python :: {v}",\n' for v in versions))
    project_path.write_text(source)
    for path in (root / '.github/workflows').glob('*.yml'):
        text = path.read_text()
        text = re.sub(r"python-version: '[^']+' # current-python",
                      f"python-version: '{latest}' # current-python", text)
        if path.name == 'checks.yml':
            text = re.sub(r'python-version: \[[^\n]+\]', f'python-version: {versions!r}', text)
        path.write_text(text)
    docker = root / 'Dockerfile'
    docker.write_text(re.sub(r'^FROM python:[^\n]+', f'FROM python:{latest}-slim', docker.read_text(), flags=re.M))
    for relative in ('README.md', 'docs/release.md'):
        path = root / relative
        text = path.read_text()
        start, end = '<!-- BEGIN PYTHON SUPPORT -->', '<!-- END PYTHON SUPPORT -->'
        if text.count(start) != 1 or text.count(end) != 1:
            raise ValueError(f'Expected one Python support block in {relative}')
        before, rest = text.split(start)
        _, after = rest.split(end)
        path.write_text(before + start + f'\nPython {oldest}–{latest} (`{requirement}`), following pandas support.\n' + end + after)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true', help='Write a compatibility proposal')
    parser.add_argument('--github-output', type=Path)
    args = parser.parse_args()
    with urlopen('https://www.python.org/downloads/', timeout=30) as response:
        downloads = response.read().decode()
    with urlopen('https://pypi.org/pypi/pandas/json', timeout=30) as response:
        pandas = json.load(response)
    versions = supported_versions(downloads, pandas)
    changed = update(ROOT, versions) if args.apply else False
    print(json.dumps({'versions': versions, 'changed': changed, 'pandas': pandas['info']['version']}))
    if args.github_output:
        with args.github_output.open('a') as output:
            output.write(f'changed={str(changed).lower()}\nlatest={versions[-1] if versions else ""}\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
