"""Verify the notebook build contract in disposable projects, never the checkout."""

import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

pytest.importorskip("nbdev", reason="notebook build tests require the notebooks extra")
nbformat = pytest.importorskip("nbformat")

SCRIPT = Path(__file__).resolve().parents[1] / "tools/export_notebooks.py"
spec = importlib.util.spec_from_file_location("timegym_notebook_build", SCRIPT)
exporter = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = exporter
spec.loader.exec_module(exporter)


def notebook(root, name, cells, *, role="source"):
    path = root / "notebooks" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    book = nbformat.v4.new_notebook(metadata={"timegym": {"role": role}})
    book.cells = [nbformat.v4.new_code_cell(source, id=f"cell-{index}") for index, source in enumerate(cells)]
    nbformat.write(book, path)
    return path.relative_to(root).as_posix()


def manifest(root, sources, handwritten=()):
    path = root / "notebooks/source-map.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema_version": 1, "sources": sources, "handwritten": list(handwritten)}))


def source(name, *modules):
    return {"notebook": name, "modules": list(modules)}


def test_export_builds_multiple_owned_modules_without_running_cells(tmp_path):
    name = notebook(tmp_path, "01_math.ipynb", [
        "#| default_exp core",
        "#| export\nfrom __future__ import annotations\n\ndef answer():\n    return 42",
        "#| export helpers.scaling\nfrom timegym.core import answer\n\ndef twice():\n    return 2 * answer()",
        "raise AssertionError('demonstrations must never run during export')",
    ])
    manifest(tmp_path, [source(name, "timegym/core.py", "timegym/helpers/scaling.py")])
    result = exporter.build(tmp_path)
    assert result.changed == ("timegym/core.py", "timegym/helpers/scaling.py")
    core = (tmp_path / "timegym/core.py").read_text()
    scaling = (tmp_path / "timegym/helpers/scaling.py").read_text()
    assert "def answer()" in core
    assert "demonstrations" not in core + scaling
    assert "../notebooks/01_math.ipynb" in core
    assert "../../notebooks/01_math.ipynb" in scaling
    assert str(tmp_path) not in core + scaling
    assert "from ..core import answer" in scaling
    assert exporter.build(tmp_path, check=True).changed == ()


def test_export_canonicalizes_header_and_merges_explicit_reexports(tmp_path):
    name = notebook(tmp_path, "01_core.ipynb", [
        "#| default_exp core",
        "#| export\nfrom __future__ import annotations\nfrom math import sqrt\n__all__ = ['sqrt', 'answer']",
        "#| export\ndef answer():\n    return 'forty-two'",
    ])
    manifest(tmp_path, [source(name, "timegym/core.py")])
    exporter.build(tmp_path)
    code = (tmp_path / "timegym/core.py").read_text()
    tree = ast.parse(code)
    assert all(isinstance(node, ast.ImportFrom) for node in tree.body[:2])
    assignments = [node for node in tree.body if isinstance(node, ast.Assign)]
    assert len(assignments) == 1
    assert assignments[0] is tree.body[2]
    namespace = {}
    exec(code, namespace)
    assert set(namespace["__all__"]) == {"sqrt", "answer"}
    assert namespace["answer"]() == "forty-two"
    assert 'return "forty-two"' in code
    assert "#cell-1" in code and "#cell-2" in code
    assert exporter.build(tmp_path, check=True).changed == ()


def test_check_reports_drift_without_changing_files_and_is_cwd_independent(tmp_path):
    name = notebook(tmp_path, "01_core.ipynb", ["#| default_exp core", "#| export\nVALUE = 1"])
    manifest(tmp_path, [source(name, "timegym/core.py")])
    exporter.build(tmp_path)
    module = tmp_path / "timegym/core.py"
    expected = module.read_bytes()
    unrelated = tmp_path / "elsewhere"
    unrelated.mkdir()
    clean = subprocess.run([sys.executable, str(SCRIPT), "--root", str(tmp_path), "--check"], cwd=unrelated, capture_output=True, text=True)
    assert clean.returncode == 0, clean.stderr
    module.write_bytes(expected + b"# manual drift\n")
    dirty = subprocess.run([sys.executable, str(SCRIPT), "--root", str(tmp_path), "--check"], cwd=unrelated, capture_output=True, text=True)
    assert dirty.returncode == 1
    assert "timegym/core.py" in dirty.stdout
    assert module.read_bytes() == expected + b"# manual drift\n"
    exporter.build(tmp_path)
    assert module.read_bytes() == expected


@pytest.mark.parametrize("invalid_source", ["def invalid(:", "return 42"])
def test_late_export_failure_never_replaces_earlier_module(tmp_path, invalid_source):
    first = notebook(tmp_path, "01_core.ipynb", ["#| default_exp core", "#| export\nVALUE = 1"])
    second = notebook(tmp_path, "02_other.ipynb", ["#| default_exp other", "#| export\nVALUE = 2"])
    manifest(tmp_path, [source(first, "timegym/core.py"), source(second, "timegym/other.py")])
    exporter.build(tmp_path)
    before = {path: path.read_bytes() for path in (tmp_path / "timegym").glob("*.py")}
    notebook(tmp_path, "01_core.ipynb", ["#| default_exp core", "#| export\nVALUE = 999"])
    notebook(tmp_path, "02_other.ipynb", ["#| default_exp other", "#| export\n" + invalid_source])
    with pytest.raises(exporter.BuildError, match="Export failed|does not compile"):
        exporter.build(tmp_path)
    assert {path: path.read_bytes() for path in before} == before


@pytest.mark.parametrize("destination", ["../outside", "/absolute", "wrong", "timegym.core"])
def test_undeclared_or_unsafe_export_destination_is_rejected(tmp_path, destination):
    name = notebook(tmp_path, "01_core.ipynb", [f"#| default_exp {destination}", "#| export\nVALUE = 1"])
    manifest(tmp_path, [source(name, "timegym/core.py")])
    with pytest.raises(exporter.BuildError, match="Invalid export module|Undeclared export"):
        exporter.build(tmp_path)
    assert not (tmp_path / "timegym").exists()


def test_duplicate_ownership_and_unmapped_existing_module_fail(tmp_path):
    first = notebook(tmp_path, "01_core.ipynb", ["#| default_exp core", "#| export\nVALUE = 1"])
    second = notebook(tmp_path, "02_other.ipynb", ["#| default_exp core", "#| export\nVALUE = 2"])
    manifest(tmp_path, [source(first, "timegym/core.py"), source(second, "timegym/core.py")])
    with pytest.raises(exporter.BuildError, match="Duplicate ownership"):
        exporter.build(tmp_path)
    manifest(tmp_path, [source(first, "timegym/core.py")])
    (tmp_path / "timegym").mkdir()
    (tmp_path / "timegym/unmapped.py").write_text("VALUE = 3\n")
    with pytest.raises(exporter.BuildError, match="missing from source map"):
        exporter.build(tmp_path)
    manifest(tmp_path, [source(first, "timegym/core.py")], handwritten=["timegym/unmapped.py"])
    exporter.build(tmp_path)
    assert (tmp_path / "timegym/unmapped.py").read_text() == "VALUE = 3\n"


def test_non_source_notebook_cannot_replace_package_module(tmp_path):
    name = notebook(tmp_path, "01_demo.ipynb", ["#| default_exp core", "#| export\nVALUE = 1"], role="demo")
    manifest(tmp_path, [source(name, "timegym/core.py")])
    with pytest.raises(exporter.BuildError, match="role='source'"):
        exporter.build(tmp_path)


def test_manifest_cannot_escape_project(tmp_path):
    name = notebook(tmp_path, "01_core.ipynb", ["#| default_exp core", "#| export\nVALUE = 1"])
    manifest(tmp_path, [source(name, "timegym/../../outside.py")])
    with pytest.raises(exporter.BuildError, match="project-relative"):
        exporter.build(tmp_path)


def test_generated_file_symlinks_are_rejected(tmp_path):
    name = notebook(tmp_path, "01_core.ipynb", ["#| default_exp core", "#| export\nVALUE = 1"])
    manifest(tmp_path, [source(name, "timegym/core.py")])
    (tmp_path / "timegym").mkdir()
    target = tmp_path / "do-not-touch.py"
    target.write_text("ORIGINAL = True\n")
    (tmp_path / "timegym/core.py").symlink_to(target)
    with pytest.raises(exporter.BuildError, match="symlinks"):
        exporter.build(tmp_path)
    assert target.read_text() == "ORIGINAL = True\n"


def test_export_reads_source_without_executing_exported_cells(tmp_path):
    name = notebook(tmp_path, "01_core.ipynb", [
        "#| default_exp core",
        "#| export\nraise RuntimeError('this source is intentionally not importable')",
    ])
    manifest(tmp_path, [source(name, "timegym/core.py")])
    exporter.build(tmp_path)
    assert "raise RuntimeError" in (tmp_path / "timegym/core.py").read_text()


@pytest.mark.parametrize("missing", [True, False])
def test_unstable_source_cell_ids_fail_without_silent_repair(tmp_path, missing):
    name = notebook(tmp_path, "01_core.ipynb", ["#| default_exp core", "#| export\nVALUE = 1"])
    path = tmp_path / name
    raw = json.loads(path.read_text())
    if missing:
        raw["cells"][0].pop("id")
    else:
        raw["cells"][1]["id"] = raw["cells"][0]["id"]
    path.write_text(json.dumps(raw))
    manifest(tmp_path, [source(name, "timegym/core.py")])
    with pytest.raises(exporter.BuildError, match="unique stable IDs"):
        exporter.build(tmp_path)
    assert not (tmp_path / "timegym").exists()
