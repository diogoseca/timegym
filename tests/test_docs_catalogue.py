"""Keep the diagnostics catalogue in step with the diagnostics the package emits."""
import ast
import json
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


# The capability map crosses capabilities with the oracle types of the catalogue.
CAPABILITY_MAP = ROOT / "docs" / "concepts" / "capability-map.md"
# A tutorial shows a diagnostic by name, or through a score or call its catalogue entry names.
SHOWN_BY = {
    "point_accuracy": {"mean_reference_mse"},
    "quantile_accuracy": {"pinball"},
    "sample_scores": {"crps", "energy_score", "variogram_score"},
    "reference_covariance": {"covariance_dependence_check", "check_dependence=True"},
    "shift_equivariance": {"check_shift_equivariance"},
}


def _table(path, heading):
    """The first Markdown table under a level-two heading, as lists of cells."""
    section = path.read_text(encoding="utf-8").split(f"\n## {heading}\n")[1].split("\n## ")[0]
    rows = [[cell.strip() for cell in line.strip().strip("|").split("|")]
            for line in section.splitlines() if line.startswith("|")]
    return rows[0], rows[2:]


def _key(name):
    return re.sub(r"<[^>]+>", DYNAMIC, name)


def catalogue_entries():
    """Map each catalogue name to its raw name, oracle type and detail section."""
    header, body = _table(CATALOGUE, "Diagnostics catalogue")
    name, oracle = header.index("Diagnostic"), header.index("Oracle type")
    text = CATALOGUE.read_text(encoding="utf-8")
    entries = {}
    for row in body:
        raw = row[name].strip("`")
        detail = text.split(f"\n### `{raw}`\n")[1].split("\n#")[0]
        entries[_key(raw)] = {"raw": raw, "oracle": row[oracle], "detail": detail}
    return entries


def _catalogue_key(name, entries):
    matches = [key for key in entries
               if re.fullmatch(re.escape(key).replace(re.escape(DYNAMIC), "[a-z_]+"), name)]
    assert len(matches) == 1, f"capability map names {name!r}, which the catalogue does not list"
    return matches[0]


def evidence_columns():
    """Map each evidence column of the capability map to its oracle type and diagnostics."""
    header, body = _table(CAPABILITY_MAP, "The four kinds of evidence")
    column, oracle, names = (header.index(h) for h in ("Column", "Oracle type", "Diagnostics"))
    return {row[column]: (row[oracle], {_key(n) for n in re.findall(r"`([^`]+)`", row[names])})
            for row in body}


def scenario_kinds():
    from timegym.processes import PROCESS_KINDS
    from timegym.scenarios import CORE_KINDS
    return set(CORE_KINDS) | set(PROCESS_KINDS)


def test_capability_map_columns_follow_the_catalogue_oracle_types():
    entries, columns = catalogue_entries(), evidence_columns()
    for column, (oracle, names) in columns.items():
        expected = {key for key, entry in entries.items() if entry["oracle"] == oracle}
        assert expected, f"{column!r}: no catalogue diagnostic has oracle type {oracle!r}"
        assert names == expected, (f"{column!r} lists {sorted(names)}; the catalogue gives "
                                   f"{sorted(expected)} the oracle type {oracle!r}")
    for key, token in ((key, token) for key, tokens in SHOWN_BY.items() for token in tokens):
        assert f"`{token}`" in entries[key]["detail"], (
            f"SHOWN_BY lets {token!r} show {key!r}, but its catalogue entry does not name it")


def test_capability_map_rows_cover_every_scenario_family():
    header, body = _table(CAPABILITY_MAP, "The map")
    assert header[0] == "Capability and scenario families"
    assert header[1:] == list(evidence_columns()), "map columns differ from the evidence table"
    listed = {name for row in body for name in re.findall(r"`([^`]+)`", row[0])} - {"n_targets"}
    kinds = scenario_kinds()
    assert listed <= kinds, f"capability map lists unknown scenario families {sorted(listed - kinds)}"
    assert listed == kinds, f"no capability row lists {sorted(kinds - listed)}"


def _tutorial_text(path):
    cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
    code = "\n".join("".join(cell["source"]) for cell in cells if cell["cell_type"] == "code")
    shown = "\n".join(
        "".join(cell["source"]) + "".join(
            "".join(output.get("text", "")) + "".join(output.get("data", {}).get("text/plain", ""))
            for output in cell.get("outputs", []))
        for cell in cells)
    return code, shown


def test_capability_map_cells_link_real_pairs():
    entries, columns = catalogue_entries(), evidence_columns()
    header, body = _table(CAPABILITY_MAP, "The map")
    kinds, filled = scenario_kinds(), 0
    for row in body:
        capability = re.match(r"\*\*(.+?)\*\*<br>", row[0])[1]
        for column, cell in zip(header[1:], row[1:]):
            for entry in filter(None, cell.split("<br>")):
                filled += 1
                families, diagnostic, tutorial = entry.split(" · ")
                families = re.findall(r"`([^`]+)`", families)
                name, anchor = re.fullmatch(
                    r"\[`([^`]+)`\]\(\.\./api/diagnostics\.md#([a-z_]+)\)", diagnostic).groups()
                number, notebook = re.fullmatch(
                    r"\[(\d\d)\]\(\.\./\.\./tutorials/(\d\d_[a-z0-9_]+\.ipynb)\)", tutorial).groups()
                where = f"{capability} × {column}: {entry}"
                assert families and set(families) <= kinds, f"{where}: unknown family"
                key = _catalogue_key(name, entries)
                assert entries[key]["oracle"] == columns[column][0], f"{where}: wrong column"
                assert anchor == re.sub(r"[<>]", "", entries[key]["raw"]), f"{where}: wrong anchor"
                path = ROOT / "tutorials" / notebook
                assert notebook.startswith(number) and path.exists(), f"{where}: no such tutorial"
                code, shown = _tutorial_text(path)
                for family in families:
                    assert re.search(rf"""['"]{family}['"]""", code), (
                        f"{where}: tutorial {number} does not generate {family!r}")
                assert any(token in shown for token in {name} | SHOWN_BY.get(key, set())), (
                    f"{where}: tutorial {number} does not show {name!r}")
    assert filled, "the capability map has no filled cells"
