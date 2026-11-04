"""Keep the four tutorial lists in the same groups and order."""
from pathlib import Path
import re

import yaml

ROOT = Path(__file__).resolve().parents[1]


class _Loader(yaml.SafeLoader):
    """Read mkdocs.yml without importing the Python objects its tags name."""


_Loader.add_multi_constructor("tag:yaml.org,2002:python/", lambda loader, suffix, node: None)


def _number(text):
    match = re.search(r"(?<![0-9])(\d\d)_[a-z]", text)
    assert match, f"no notebook number in {text!r}"
    return match.group(1)


def nav_tutorials():
    config = yaml.load((ROOT / "mkdocs.yml").read_text(encoding="utf-8"), Loader=_Loader)
    tutorials = next(item["Tutorials"] for item in config["nav"] if "Tutorials" in item)
    listed = []
    for entry in tutorials:
        (title, value), = entry.items()
        if isinstance(value, str):
            assert title == "Overview", f"ungrouped tutorial entry {title!r}"
            continue
        for page in value:
            (_, path), = page.items()
            listed.append((title, _number(path)))
    return listed


def table_tutorials(path, heading):
    lines = (ROOT / path).read_text(encoding="utf-8").splitlines()
    start = lines.index(heading)
    listed, group = [], None
    for line in lines[start + 2:]:
        if not line.startswith("|"):
            break
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        group = cells[0] or group
        listed.append((group, _number(cells[1])))
    return listed


def test_tutorial_lists_agree():
    expected = nav_tutorials()
    assert len(expected) == len(set(number for _, number in expected))
    on_disk = sorted(
        _number(path.name)
        for pattern in ("00_*.ipynb", "2[0-9]_*.ipynb", "3[0-9]_*.ipynb")
        for path in (ROOT / "notebooks").glob(pattern)
    )
    assert sorted(number for _, number in expected) == on_disk
    lists = {
        "README.md": table_tutorials("README.md", "| Group | Notebook | What it shows |"),
        "notebooks/README.md": table_tutorials(
            "notebooks/README.md", "| Group | Application | Explore |"
        ),
        "docs/notebooks.md": table_tutorials(
            "docs/notebooks.md", "| Group | Tutorial | Question it answers | Needs |"
        ),
    }
    for path, listed in lists.items():
        assert listed == expected, f"{path} disagrees with the mkdocs.yml tutorial groups"
