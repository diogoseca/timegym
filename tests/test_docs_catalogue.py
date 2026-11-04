"""Keep the diagnostics catalogue in step with the diagnostics the package emits."""
import ast
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
CATALOGUE = ROOT / "docs" / "api" / "diagnostics.md"
STATUSES = r"\b(PASS|FAIL|INCONCLUSIVE|UNSUPPORTED|ERROR)\b"
DYNAMIC = "<*>"  # A name part filled in at run time, such as metric_<name>.


def _values(node, tree, parents):
    """Every string an argument can take, resolved statically."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return {node.value}
    if isinstance(node, ast.IfExp):
        return _values(node.body, tree, parents) | _values(node.orelse, tree, parents)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return {prefix + DYNAMIC for prefix in _values(node.left, tree, parents)}
    if isinstance(node, ast.JoinedStr) and isinstance(node.values[0], ast.Constant):
        return {node.values[0].value + DYNAMIC}
    if isinstance(node, ast.Name):
        function = parents[node]
        while not isinstance(function, (ast.FunctionDef, ast.Module)):
            function = parents[function]
        found = set()
        for item in ast.walk(function):  # Local constants, including tuple unpacking.
            if isinstance(item, ast.Assign):
                for target in item.targets:
                    pairs = (zip(target.elts, item.value.elts)
                             if isinstance(target, ast.Tuple) and isinstance(item.value, ast.Tuple)
                             else [(target, item.value)])
                    found |= {v for t, value in pairs if getattr(t, "id", None) == node.id
                              for v in _values(value, tree, parents)}
        parameters = [a.arg for a in getattr(function, "args", ast.arguments()).args]
        if node.id in parameters:  # A helper parameter: resolve every call site.
            position = parameters.index(node.id)
            for call in ast.walk(tree):
                if isinstance(call, ast.Call) and getattr(call.func, "id", None) == function.name:
                    passed = call.args[position:position + 1] or [
                        k.value for k in call.keywords if k.arg == node.id]
                    found |= {v for value in passed for v in _values(value, tree, parents)}
        if found:
            return found
    raise AssertionError(f"Cannot resolve a DiagnosticResult field at line {node.lineno}: "
                         f"{ast.unparse(node)}; extend tests/test_docs_catalogue.py")


def emitted():
    """Map each diagnostic name the package can emit to its statuses and categories."""
    found = {}
    for path in sorted((ROOT / "timegym").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
        for call in ast.walk(tree):
            func = getattr(call, "func", None)
            if getattr(func, "id", getattr(func, "attr", None)) != "DiagnosticResult":
                continue
            fields = dict(zip(("name", "status", "category"), call.args))
            fields |= {k.arg: k.value for k in call.keywords if k.arg}
            if "name" not in fields:
                # DiagnosticResult(**stored) reloads a saved report; it emits nothing new.
                assert any(k.arg is None for k in call.keywords), f"{path}:{call.lineno}"
                continue
            for name in _values(fields["name"], tree, parents):
                statuses, categories = found.setdefault(name, (set(), set()))
                statuses |= _values(fields["status"], tree, parents)
                categories |= _values(fields["category"], tree, parents)
    return found


def catalogue():
    """Map each catalogue row's diagnostic name to its listed statuses and category."""
    section = CATALOGUE.read_text(encoding="utf-8").split("\n## Diagnostics catalogue\n")[1]
    rows = [[cell.strip() for cell in line.strip().strip("|").split("|")]
            for line in section.split("\n#")[0].splitlines() if line.startswith("|")]
    header, _, *body = rows
    name, category, status = (header.index(column) for column in
                              ("Diagnostic", "Category", "Statuses it can return"))
    return {re.sub(r"<[^>]+>", DYNAMIC, row[name].strip("`")):
            (set(re.findall(STATUSES, row[status])), {row[category].strip("`")})
            for row in body}


def test_catalogue_lists_exactly_the_emitted_diagnostics():
    code, documented = emitted(), catalogue()
    missing, stale = sorted(code.keys() - documented.keys()), sorted(documented.keys() - code.keys())
    assert not missing and not stale, (
        f"docs/api/diagnostics.md catalogue lacks rows for {missing}; "
        f"it lists names the package no longer emits: {stale}")


def test_catalogue_statuses_and_categories_match_the_code():
    code, documented = emitted(), catalogue()
    differing = {name: {"code": code[name], "catalogue": documented[name]}
                 for name in sorted(code.keys() & documented.keys()) if code[name] != documented[name]}
    assert not differing, f"Catalogue statuses or categories differ from the code: {differing}"
