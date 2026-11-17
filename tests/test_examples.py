"""Keep the example notebooks, their data records and their gallery in step."""
import importlib.util
import json
from pathlib import Path

import nbformat
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
DATA = EXAMPLES / "data"
NOTEBOOKS = sorted(EXAMPLES.glob("*.ipynb"))
RECORDS = json.loads((DATA / "datasets.json").read_text(encoding="utf-8"))
SAMPLE_LIMIT = 500_000  # bytes; the samples are committed, so they stay small


def _module(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


example_data = _module(EXAMPLES / "example_data.py")
example_figures = _module(ROOT / "tools" / "example_figures.py")


class _Loader(yaml.SafeLoader):
    """Read mkdocs.yml without importing the Python objects its tags name."""


_Loader.add_multi_constructor("tag:yaml.org,2002:python/", lambda loader, suffix, node: None)


def _nav_examples():
    config = yaml.load((ROOT / "mkdocs.yml").read_text(encoding="utf-8"), Loader=_Loader)
    section = next(item["Examples"] for item in config["nav"] if "Examples" in item)
    listed = []
    for entry in section:
        (title, value), = entry.items()
        if isinstance(value, str):
            assert (title, value) == ("Overview", "docs/examples.md")
            continue
        listed += [path for page in value for path in page.values()]
    return listed


def test_examples_come_right_after_the_home_page():
    config = yaml.load((ROOT / "mkdocs.yml").read_text(encoding="utf-8"), Loader=_Loader)
    assert [next(iter(item)) for item in config["nav"][:2]] == ["Home", "Examples"]


def test_every_example_is_listed_in_the_nav_the_gallery_and_the_guide():
    assert NOTEBOOKS, "no example notebooks found"
    on_disk = sorted(f"examples/{path.name}" for path in NOTEBOOKS)
    assert sorted(_nav_examples()) == on_disk
    gallery = (ROOT / "docs" / "examples.md").read_text(encoding="utf-8")
    guide = (EXAMPLES / "README.md").read_text(encoding="utf-8")
    for path in NOTEBOOKS:
        assert f"](../examples/{path.name})" in gallery, f"{path.name} has no card on docs/examples.md"
        assert f"](images/examples/{path.stem}.png)" in gallery, f"{path.name}'s card has no picture"
        assert f"]({path.name})" in guide, f"{path.name} is missing from examples/README.md"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda path: path.stem)
def test_example_is_saved_with_its_outputs(path):
    notebook = nbformat.read(path, as_version=4)
    assert notebook.metadata.get("timegym", {}).get("role") == "example"
    assert notebook.cells[0].cell_type == "markdown" and notebook.cells[0].source.startswith("# ")
    code = [cell for cell in notebook.cells if cell.cell_type == "code" and cell.source.strip()]
    assert code and all(cell.execution_count is not None for cell in code), "execute it with --write"
    assert not any(output.output_type == "error" for cell in code for output in cell.outputs)
    source = "\n".join(cell.source for cell in code)
    assert "from example_data import" in source, "examples read their data through example_data"
    assert any(f'load("{name}"' in source for name in RECORDS), "the example loads no recorded dataset"
    assert "full=True" not in source, "an example must run on the committed sample"


def test_gallery_pictures_match_their_notebooks():
    assert example_figures.problems() == []


def test_samples_match_their_records():
    assert example_data.check() == []
    assert sorted(example_data.names()) == sorted(RECORDS)


@pytest.mark.parametrize("name", sorted(RECORDS))
def test_dataset_record_is_complete_and_documented(name):
    entry = RECORDS[name]
    assert set(entry) == {"title", "sample", "full", "source", "license", "credit", "changes"}
    assert {"file", "sha256", "holds"} <= set(entry["sample"])
    assert {"name", "publisher", "page", "download", "file", "bytes", "sha256", "retrieved"} <= set(entry["source"])
    assert set(entry["license"]) == {"name", "url"}
    assert all(str(entry[key]).strip() for key in ("title", "full", "credit", "changes"))
    assert entry["source"]["download"].startswith("https://") and len(entry["source"]["sha256"]) == 64
    sample = DATA / entry["sample"]["file"]
    assert sample.stat().st_size <= SAMPLE_LIMIT, f"{sample.name} is too large to commit"
    page = (DATA / "README.md").read_text(encoding="utf-8")
    for needed in (entry["sample"]["file"], entry["source"]["page"], entry["license"]["name"],
                   entry["license"]["url"], entry["credit"], entry["changes"]):
        assert needed in page, f"examples/data/README.md does not give {needed!r} for {name}"


def test_data_folder_holds_only_recorded_files():
    expected = {"README.md", "datasets.json", *(entry["sample"]["file"] for entry in RECORDS.values())}
    assert {path.name for path in DATA.iterdir()} == expected


def test_every_dataset_is_used_by_an_example():
    sources = "\n".join(cell.source for path in NOTEBOOKS for cell in nbformat.read(path, as_version=4).cells)
    for name in RECORDS:
        assert f'load("{name}"' in sources, f"no example loads {name}"


def test_loader_returns_regular_date_indexes():
    for name in RECORDS:
        table = example_data.load(name)
        assert len(table) > 0 and table.index.is_monotonic_increasing and table.index.is_unique
        assert table.index.freq is not None, f"{name}: TimeGym needs a regular date index"
        assert "License" in example_data.about(name)
