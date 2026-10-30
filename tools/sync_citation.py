"""Render README citations from CITATION.cff and check release identity."""
from __future__ import annotations

import argparse
import ast
from datetime import date
from pathlib import Path
import re

import yaml

ROOT = Path(__file__).resolve().parents[1]
START = '<!-- BEGIN GENERATED CITATION -->'
END = '<!-- END GENERATED CITATION -->'


def package_version(root: Path) -> str:
    for node in ast.parse((root / 'timegym/__init__.py').read_text()).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == '__version__'
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise ValueError('Missing package __version__')


def read_metadata(root: Path = ROOT) -> dict:
    metadata = yaml.safe_load((root / 'CITATION.cff').read_text())
    version = package_version(root)
    if not re.fullmatch(r'\d+\.\d+\.\d+', version):
        raise ValueError('Package version must contain three integer counters')
    if str(metadata.get('version')) != version:
        raise ValueError('CITATION.cff version differs from package __version__')
    released = date.fromisoformat(str(metadata['date-released']))
    heading = re.search(r'^## (\S+) — (\S+)', (root / 'CHANGELOG.md').read_text(), re.M)
    if not heading or heading.groups() != (version, released.isoformat()):
        raise ValueError('Leading CHANGELOG version/date differs from CITATION.cff')
    if metadata.get('doi') and not re.fullmatch(r'10\.\d{4,9}/\S+', metadata['doi']):
        raise ValueError('CITATION.cff doi must be a registered DOI, without a URL prefix')
    return metadata


def bibtex(entry: dict, key: str, kind: str) -> str:
    authors = ' and '.join(f"{a['family-names']}, {a['given-names']}" for a in entry['authors'])
    fields = {'author': authors, 'title': '{' + entry['title'] + '}',
              'year': str(entry.get('year', str(entry.get('date-released', ''))[:4]))}
    for field in ('version', 'doi'):
        if entry.get(field):
            fields[field] = str(entry[field])
    fields['url'] = entry.get('url', entry.get('repository-code'))
    return '\n'.join([f'@{kind}{{{key},', *[
        f'  {name} = {{{value}}},' for name, value in fields.items()
    ], '}'])


def render(metadata: dict) -> str:
    year = str(metadata['date-released'])[:4]
    author = metadata['authors'][0]['family-names'].lower()
    names = []
    for person in metadata['authors']:
        initials = ' '.join(part[0] + '.' for part in person['given-names'].split())
        names.append(f"{person['family-names']}, {initials}")
    apa_names = ', '.join(names[:-1]) + ', & ' + names[-1] if len(names) > 1 else names[0]
    url = 'https://doi.org/' + metadata['doi'] if metadata.get('doi') else metadata['repository-code']
    lines = [START, '', '```bibtex', bibtex(metadata, f'{author}{year}timegym', 'software'),
             '```', '', f"{apa_names} ({year}). *{metadata['title']}*",
             f"(Version {metadata['version']}) [Computer software]. {url}", '']
    for index, reference in enumerate(metadata.get('references', [])):
        key = f"{reference['authors'][0]['family-names'].lower()}{reference['year']}timegym"
        lines.extend(['The earlier manuscript:', '', '```bibtex',
                      bibtex(reference, key if index == 0 else f'{key}{index}', 'article'), '```', ''])
    return '\n'.join([*lines, END])


def synchronize(root: Path = ROOT, *, check: bool = False, tag: str | None = None) -> bool:
    metadata = read_metadata(root)
    if tag is not None and tag != 'v' + str(metadata['version']):
        raise ValueError('Release tag must equal v' + str(metadata['version']))
    readme = root / 'README.md'
    text = readme.read_text()
    if text.count(START) != 1 or text.count(END) != 1:
        raise ValueError('README needs exactly one generated citation block')
    before, rest = text.split(START)
    _, after = rest.split(END)
    rendered = before + render(metadata) + after
    if check:
        return text == rendered
    readme.write_text(rendered)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--tag', help='Require this release tag to match the package version')
    args = parser.parse_args()
    try:
        matches = synchronize(check=args.check, tag=args.tag)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f'{error}\n')
    if not matches:
        print('README citations are stale. Run python tools/sync_citation.py')
        return 1
    print('Citation and release metadata agree.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
