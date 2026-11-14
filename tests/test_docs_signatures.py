"""Keep the generated API signatures in step with the public surface.

Each API page ends with `::: <import path>` directives that mkdocstrings renders
from docstrings. Members without a docstring are left out of the page, so every
rendered class, method, property and function must have one.
"""
import ast
import importlib
import inspect
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "docs" / "reference.md"
API = ROOT / "docs" / "api"
# mkdocstrings reads source statically with griffe, which does not load modules
# from a directory without __init__.py, such as timegym/data. No documented name
# lives there now; a name added there would need a hand-written description.
STATIC_LOADER_GAPS = set()


def public_surface():
    """Map each public name's full import path to the API pages its row links to."""
    section = REFERENCE.read_text(encoding="utf-8").split("\n## Public surface\n")[1]
    rows = [[cell.strip() for cell in line.strip().strip("|").split("|")]
            for line in section.split("\n#")[0].splitlines() if line.startswith("|")]
    header, _, *body = rows
    names, module, page = (header.index(h) for h in ("Name", "Import path", "Page"))
    surface = {}
    for row in body:
        pages = {Path(link).stem for link in re.findall(r"\]\(api/([^)#]+)", row[page])}
        for name in re.findall(r"`([^`]+)`", row[names]):
            surface[f"{row[module].strip('`')}.{name}"] = pages
    return surface


def directives():
    """Map each `:::` identifier on the API pages to the pages that render it."""
    found = {}
    for page in sorted(API.glob("*.md")):
        for identifier in re.findall(r"^::: (\S+)$", page.read_text(encoding="utf-8"), re.M):
            found.setdefault(identifier, []).append(page.stem)
    return found


def _object(path):
    module, _, name = path.rpartition(".")
    return getattr(importlib.import_module(module), name)


def rendered(path):
    """A class or function defined under its public name; constants and aliases stay prose."""
    obj = _object(path)
    return (inspect.isclass(obj) or inspect.isfunction(obj)) and obj.__name__ == path.rpartition(".")[2]


def test_every_public_class_and_function_has_one_signature():
    surface, found = public_surface(), directives()
    expected = {path for path in surface if rendered(path)} - STATIC_LOADER_GAPS
    missing, extra = sorted(expected - found.keys()), sorted(found.keys() - expected)
    assert not missing and not extra, (
        f"API pages lack ::: directives for {missing}; they render names that are not "
        f"public classes or functions in docs/reference.md: {extra}")
    for path, pages in found.items():
        assert len(pages) == 1, f"{path} is rendered on several pages: {pages}"
        assert pages[0] in surface[path], (
            f"{path} is rendered on {pages[0]}.md, but docs/reference.md links it to {sorted(surface[path])}")


def test_static_loader_gaps_still_apply():
    assert not (ROOT / "timegym" / "data" / "__init__.py").exists(), (
        "timegym.data is a regular package now: render STATIC_LOADER_GAPS on their API "
        "pages and empty the set")
    surface = public_surface()
    for path in STATIC_LOADER_GAPS:
        assert path in surface and rendered(path), path


def _definition(obj):
    tree = ast.parse(Path(inspect.getsourcefile(obj)).read_text(encoding="utf-8"))
    kinds = (ast.ClassDef, ast.FunctionDef)
    matches = [node for node in tree.body if isinstance(node, kinds) and node.name == obj.__qualname__]
    assert len(matches) == 1, obj.__qualname__
    return matches[0]


def test_rendered_names_and_public_members_have_docstrings():
    undocumented = []
    for path in sorted(directives()):
        node = _definition(_object(path))
        if not ast.get_docstring(node):
            undocumented.append(path)
        if isinstance(node, ast.ClassDef):
            undocumented += [f"{path}.{member.name}" for member in node.body
                             if isinstance(member, ast.FunctionDef)
                             and not member.name.startswith("_")
                             and not ast.get_docstring(member)]
    assert not undocumented, (
        f"Add docstrings in the source notebooks; mkdocstrings omits {undocumented}")
