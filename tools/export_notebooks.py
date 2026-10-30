#!/usr/bin/env python3
"""Build TimeGym modules from the notebooks declared in notebooks/source-map.json.

Run from any directory: ``python tools/export_notebooks.py [--check]``.
Export reads saved cells; it never executes notebook examples or package code.
All exports and syntax checks finish before any package file is replaced.
"""

from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import tempfile


class BuildError(ValueError):
    """The source map or a notebook cannot produce the declared package."""


@dataclass(frozen=True)
class Source:
    notebook: str
    modules: tuple[str, ...]


@dataclass(frozen=True)
class BuildResult:
    modules: tuple[str, ...]
    changed: tuple[str, ...]


def _relative_path(value: object, root: Path, *, notebook: bool = False) -> str:
    if not isinstance(value, str) or not value:
        raise BuildError("Manifest paths must be nonempty strings")
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value or any(
        part in {".", ".."} for part in path.parts
    ) or "\\" in value:
        raise BuildError(f"Expected a normalized project-relative path: {value!r}")
    if notebook:
        valid = len(path.parts) == 2 and path.parts[0] == "notebooks" and path.suffix == ".ipynb"
    else:
        names = (*path.parts[1:-1], path.stem)
        valid = (
            len(path.parts) >= 2
            and path.parts[0] == "timegym"
            and path.suffix == ".py"
            and all(re.fullmatch(r"[A-Za-z_]\w*", name) for name in names)
        )
    if not valid:
        expected = "notebooks/<name>.ipynb" if notebook else "timegym/<module>.py"
        raise BuildError(f"Expected {expected}, got {value!r}")
    cursor = root
    for part in path.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise BuildError(f"Source/build paths may not traverse symlinks: {value}")
    return value


def load_manifest(root: Path) -> tuple[Source, ...]:
    """Validate ownership, including explicit coverage of existing modules."""
    manifest_path = root / "notebooks/source-map.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise BuildError(f"Cannot read {manifest_path}: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise BuildError("Source map must be an object with schema_version: 1")
    entries = manifest.get("sources")
    handwritten = manifest.get("handwritten")
    if not isinstance(entries, list) or not entries or not isinstance(handwritten, list):
        raise BuildError("Source map requires a nonempty sources list and a handwritten list")
    owners: dict[str, str] = {}
    notebooks: set[str] = set()
    sources = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise BuildError("Each sources entry must be an object")
        notebook = _relative_path(entry.get("notebook"), root, notebook=True)
        if notebook in notebooks:
            raise BuildError(f"Notebook declared more than once: {notebook}")
        notebooks.add(notebook)
        if not (root / notebook).is_file():
            raise BuildError(f"Source notebook does not exist: {notebook}")
        values = entry.get("modules")
        if not isinstance(values, list) or not values:
            raise BuildError(f"Source must declare a nonempty modules list: {notebook}")
        modules = tuple(_relative_path(value, root) for value in values)
        for module in modules:
            if module in owners:
                raise BuildError(f"Duplicate ownership of {module}: {owners[module]} and {notebook}")
            owners[module] = notebook
        sources.append(Source(notebook, modules))
    for value in handwritten:
        module = _relative_path(value, root)
        if module in owners:
            raise BuildError(f"Duplicate ownership of {module}: {owners[module]} and handwritten")
        if not (root / module).is_file():
            raise BuildError(f"Handwritten module does not exist: {module}")
        owners[module] = "handwritten"
    existing = {path.relative_to(root).as_posix() for path in (root / "timegym").rglob("*.py")}
    for module in sorted(existing):
        _relative_path(module, root)
    missing = existing - owners.keys()
    if missing:
        raise BuildError("Existing modules missing from source map: " + ", ".join(sorted(missing)))
    return tuple(sources)


@contextmanager
def _working_directory(root: Path):
    previous = Path.cwd()
    os.chdir(root)
    try:
        yield
    finally:
        os.chdir(previous)


def _export_source(root: Path, stage: Path, source: Source) -> None:
    # nbdev is a development dependency; importing timegym is deliberately avoided.
    try:
        import nbformat
        from nbdev.export import nb_export
        from nbdev.maker import ModuleMaker
    except ImportError as exc:
        raise BuildError("Install the notebook build dependencies with pip install -e '.[notebooks]'") from exc

    try:
        # Check raw JSON before nbformat gets a chance to repair missing cell IDs.
        raw = json.loads((root / source.notebook).read_text(encoding="utf-8"))
        ids = [cell.get("id") for cell in raw["cells"]]
        if any(not cell_id for cell_id in ids) or len(set(ids)) != len(ids):
            raise BuildError(f"Source cells require unique stable IDs: {source.notebook}")
        notebook = nbformat.from_dict(raw)
        nbformat.validate(notebook)
    except BuildError:
        raise
    except Exception as exc:
        raise BuildError(f"Invalid source notebook {source.notebook}: {exc}") from exc
    role_metadata = notebook.metadata.get("timegym", {})
    if not isinstance(role_metadata, dict) or role_metadata.get("role") != "source":
        raise BuildError(f"Notebook must declare metadata.timegym.role='source': {source.notebook}")
    made: set[str] = set()

    class OwnedModuleMaker(ModuleMaker):
        def __init__(self, dest, name, nb_path, **kwargs):
            if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", name):
                raise BuildError(f"Invalid export module name in {source.notebook}: {name!r}")
            module = "timegym/" + name.replace(".", "/") + ".py"
            if module not in source.modules:
                raise BuildError(f"Undeclared export from {source.notebook}: {module}")
            if module in made:
                raise BuildError(f"Use one consistent export destination for {module} in {source.notebook}")
            made.add(module)
            # Each module has exactly one notebook owner. Explicit destinations
            # are complete modules too, never append operations on existing files.
            kwargs["is_new"] = True
            super().__init__(dest, name, nb_path, **kwargs)
            self.dest2nb = os.path.relpath(root / source.notebook, (root / module).parent).replace(os.sep, "/")
            self.hdr = f"# %% {self.dest2nb}"
            # The source map is sufficient provenance. Do not inject hosted-docs
            # URLs from a developer's personal nbdev configuration into modules.
            self.nb_path = None

    try:
        nb_export(str(root / source.notebook), lib_path=str(stage / "timegym"), mod_maker=OwnedModuleMaker)
    except BuildError:
        raise
    except Exception as exc:
        raise BuildError(f"Export failed for {source.notebook}: {exc}") from exc
    missing = set(source.modules) - made
    if missing:
        raise BuildError(f"No exported code for {source.notebook}: {', '.join(sorted(missing))}")


def _canonical_module(contents: str, module: str) -> bytes:
    """Keep notebook provenance while giving every module the same header."""
    try:
        import black
    except ImportError as exc:
        raise BuildError("Install the notebook build dependencies with pip install -e '.[dev]'") from exc
    try:
        tree = ast.parse(contents, filename=module)
        lines = contents.splitlines(keepends=True)
        imports = []
        exports = []
        docstring = []
        removed = {0}
        for index, node in enumerate(tree.body):
            is_import = isinstance(node, (ast.Import, ast.ImportFrom))
            is_exports = isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "__all__"
                for target in node.targets
            )
            is_docstring = index == 0 and isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
            if not (is_import or is_exports or is_docstring):
                continue
            if is_exports:
                values = ast.literal_eval(node.value)
                if not isinstance(values, (list, tuple)) or not all(isinstance(value, str) for value in values):
                    raise ValueError("__all__ must be a literal sequence of strings")
                exports.extend(value for value in values if value not in exports)
            else:
                destination = imports if is_import else docstring
                destination.append("".join(lines[node.lineno - 1:node.end_lineno]).strip())
            removed.update(range(node.lineno - 1, node.end_lineno))
        body = "".join(line for index, line in enumerate(lines) if index not in removed and not line.startswith("# %% auto "))
        header = [lines[0].strip(), *docstring, "\n".join(imports), f"__all__ = {exports!r}"]
        canonical = "\n\n".join(part for part in header if part) + "\n\n" + body.strip() + "\n"
        return black.format_file_contents(canonical, fast=False, mode=black.Mode(target_versions={black.TargetVersion.PY310})).encode("utf-8")
    except black.NothingChanged:
        return canonical.encode("utf-8")
    except (SyntaxError, ValueError, black.InvalidInput) as exc:
        raise BuildError(f"Generated module does not compile or has an invalid header: {module}: {exc}") from exc


def build(root: str | Path, *, check: bool = False) -> BuildResult:
    """Export and validate everything, then compare or replace generated files.

    A failed parse/export/syntax check never changes package files. Replacements
    are atomic per file; an OS failure during the replacement phase is reported
    rather than claiming a transaction across multiple filesystem entries.
    """
    root = Path(root).resolve()
    sources = load_manifest(root)
    declared = tuple(sorted(module for source in sources for module in source.modules))
    with tempfile.TemporaryDirectory(prefix="timegym-export-") as scratch:
        stage = Path(scratch)
        with _working_directory(root):
            for source in sources:
                _export_source(root, stage, source)
        actual = {path.relative_to(stage).as_posix() for path in stage.rglob("*") if path.is_file()}
        if actual != set(declared):
            raise BuildError(f"Export output differs from declared modules: {sorted(actual ^ set(declared))}")
        outputs = {}
        for module in declared:
            contents = _canonical_module((stage / module).read_text(encoding="utf-8"), module)
            try:
                compile(contents, module, "exec", dont_inherit=True)
            except (SyntaxError, ValueError) as exc:
                raise BuildError(f"Generated module does not compile: {module}: {exc}") from exc
            outputs[module] = contents
        changed = tuple(module for module in declared if not (root / module).is_file() or (root / module).read_bytes() != outputs[module])
        if not check:
            staged_replacements = []
            try:
                # Prepare every replacement before touching any existing module.
                for module in changed:
                    destination = root / module
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with tempfile.NamedTemporaryFile(prefix=f".{destination.name}.", dir=destination.parent, delete=False) as handle:
                        temporary = Path(handle.name)
                        staged_replacements.append((temporary, destination))
                        handle.write(outputs[module])
                        handle.flush()
                        os.fsync(handle.fileno())
                    temporary.chmod(destination.stat().st_mode & 0o777 if destination.exists() else 0o644)
                for temporary, destination in staged_replacements:
                    os.replace(temporary, destination)
            finally:
                for temporary, _ in staged_replacements:
                    temporary.unlink(missing_ok=True)
    return BuildResult(declared, changed)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="report source drift without changing files")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1], help="project root (defaults to this script's project)")
    args = parser.parse_args(argv)
    try:
        result = build(args.root, check=args.check)
    except (BuildError, OSError) as exc:
        print(f"Notebook build failed: {exc}", file=sys.stderr)
        return 2
    if args.check and result.changed:
        print("Generated modules differ from their source notebooks:")
        print("\n".join(f"  {path}" for path in result.changed))
        print("Save the notebooks, then run python tools/export_notebooks.py.")
        return 1
    verb = "Checked" if args.check else "Built"
    print(f"{verb} {len(result.modules)} notebook-owned modules; {len(result.changed)} changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
