"""The catalogue of Study 3, Part 1 (protocol 3.0): 16 time-series bug operators and the 4 Study 2 bugs.

Each operator is a source-to-source transformation of experiments/study3_subject.py, as defined in
part_1_generated_mutants.sources.catalogue.operators and sources.study2_bugs.bugs. An operator finds its site by the
trailing "# site: <name>" comment (located with tokenize, because comments are not in the AST), takes the one
statement that starts on that line, checks that the statement and its expression have exactly the expected form,
and replaces it; anything else raises CatalogueError. Every source is produced by ast.unparse, so it carries no
comments. The two scaler-reuse operators keep the statistics of their first call in a module-level dictionary of
the mutated source, first_scaler, through dict.setdefault, which returns the stored value on every later call.

Standard library only, so that it runs in the Cosmic Ray environment. Frozen with the protocol.
"""
from __future__ import annotations

import ast
import io
import re
import tokenize

SITES = ("T-lag", "T-season", "T-target", "P-lag", "P-season", "S-x", "S-y", "H", "I", "R")
MARKER = re.compile(r"#\s*site:\s*(\S+)\s*")
MEMORY = "first_scaler"

# (operator id, site, expected expression, replacement expression) for operators that rewrite the value of the
# assignment on their site line.
VALUE_OPERATORS = (
    ("lag_window_off_by_one_training", "T-lag", "values[t - window:t]", "values[t - window - 1:t - 1]"),
    ("lag_window_off_by_one_prediction", "P-lag", "history[-window:]", "history[-window - 1:-1]"),
    ("lag_window_includes_target_training", "T-lag", "values[t - window:t]", "values[t - window + 1:t + 1]"),
    ("lag_order_reversed_training", "T-lag", "values[t - window:t]", "values[t - window:t][::-1]"),
    ("lag_order_reversed_prediction", "P-lag", "history[-window:]", "history[-window:][::-1]"),
    ("target_columns_swapped", "T-target", "values[start:]", "values[start:][:, ::-1]"),
    ("series_columns_swapped_prediction", "P-lag", "history[-window:]", "history[-window:][:, ::-1]"),
    ("feature_scaler_reused_across_contexts", "S-x", "fit_scaler(features)",
     f"{MEMORY}.setdefault('features', fit_scaler(features))"),
    ("target_scaler_reused_across_contexts", "S-y", "fit_scaler(targets)",
     f"{MEMORY}.setdefault('targets', fit_scaler(targets))"),
    ("inverse_transform_skipped", "I", "invert_scaler(output, self.target_center, self.target_scale)", "output"),
    ("inverse_transform_applied_twice", "I", "invert_scaler(output, self.target_center, self.target_scale)",
     "invert_scaler(invert_scaler(output, self.target_center, self.target_scale), self.target_center, "
     "self.target_scale)"),
    ("seasonal_period_off_by_one_training", "T-season", "values[t - period]", "values[t - period - 1]"),
    ("seasonal_period_off_by_one_prediction", "P-season", "history[-period]", "history[-period - 1]"),
    ("training_rows_dropped_from_wrong_end", "T-target", "values[start:]", "values[:len(values) - start]"),
)
OPERATORS = (
    "lag_window_off_by_one_training", "lag_window_off_by_one_prediction", "lag_window_includes_target_training",
    "lag_order_reversed_training", "lag_order_reversed_prediction", "recursion_state_not_updated",
    "horizon_index_off_by_one", "target_columns_swapped", "series_columns_swapped_prediction",
    "feature_scaler_reused_across_contexts", "target_scaler_reused_across_contexts", "inverse_transform_skipped",
    "inverse_transform_applied_twice", "seasonal_period_off_by_one_training",
    "seasonal_period_off_by_one_prediction", "training_rows_dropped_from_wrong_end",
)
OPERATOR_SITES = {**{name: site for name, site, _, _ in VALUE_OPERATORS},
                  "recursion_state_not_updated": "R", "horizon_index_off_by_one": "H"}
STUDY2_BUGS = ("reverse_prediction_lags", "stale_recursive_state", "omit_inverse_target_scaling",
               "swap_recursive_targets")
STUDY2_BUG_SAME_SOURCE = {"reverse_prediction_lags": "lag_order_reversed_prediction",
                          "stale_recursive_state": "recursion_state_not_updated",
                          "omit_inverse_target_scaling": "inverse_transform_skipped"}


class CatalogueError(ValueError):
    """The subject does not have the form an operator expects."""


def site_lines(source: str) -> dict[str, int]:
    """Line number of every site marker; each registered site must appear exactly once and nothing else."""
    lines: dict[str, int] = {}
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type != tokenize.COMMENT:
            continue
        match = MARKER.fullmatch(token.string)
        if match is None:
            continue
        name = match.group(1)
        if name not in SITES:
            raise CatalogueError(f"Unknown site marker {name!r} on line {token.start[0]}")
        if name in lines:
            raise CatalogueError(f"Site {name} is marked on lines {lines[name]} and {token.start[0]}")
        lines[name] = token.start[0]
    missing = [name for name in SITES if name not in lines]
    if missing:
        raise CatalogueError(f"Site markers missing: {missing}")
    return lines


def _expression(text: str) -> ast.expr:
    return ast.parse(text, mode="eval").body


def _same_expression(node: ast.AST, text: str) -> bool:
    return ast.dump(node) == ast.dump(_expression(text))


def _same_statement(node: ast.AST, text: str) -> bool:
    return ast.dump(node) == ast.dump(ast.parse(text).body[0])


def _bodies(tree: ast.AST):
    """Every statement list of the tree, with its owner."""
    for owner in ast.walk(tree):
        for field in ("body", "orelse", "finalbody"):
            body = getattr(owner, field, None)
            if isinstance(body, list) and body and isinstance(body[0], ast.stmt):
                yield owner, body


def _statement_at(tree: ast.Module, line: int) -> tuple[ast.stmt, list, ast.AST]:
    """The one statement that starts on ``line``, the list that holds it and that list's owner."""
    found = [(statement, body, owner) for owner, body in _bodies(tree) for statement in body
             if statement.lineno == line]
    if len(found) != 1:
        raise CatalogueError(f"Expected one statement starting on line {line}, found {len(found)}")
    return found[0]


def _function(tree: ast.Module, statement: ast.stmt) -> ast.FunctionDef:
    """The innermost function definition that contains ``statement``."""
    holders = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
               and any(child is statement for child in ast.walk(node))]
    if not holders:
        raise CatalogueError("The site is not inside a function")
    return min(holders, key=lambda node: node.end_lineno - node.lineno)


def _method(tree: ast.Module, class_name: str, name: str) -> ast.FunctionDef:
    found = [item for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name
             for item in node.body if isinstance(item, ast.FunctionDef) and item.name == name]
    if len(found) != 1:
        raise CatalogueError(f"Expected one method {class_name}.{name}, found {len(found)}")
    return found[0]


def _site(source: str, tree: ast.Module, site: str) -> tuple[ast.stmt, list, ast.AST]:
    return _statement_at(tree, site_lines(source)[site])


def _replace_value(source: str, tree: ast.Module, site: str, expected: str, replacement: str) -> None:
    statement, _, _ = _site(source, tree, site)
    if not isinstance(statement, ast.Assign) or statement.end_lineno != statement.lineno:
        raise CatalogueError(f"Site {site} must be an assignment on one line")
    if not _same_expression(statement.value, expected):
        raise CatalogueError(f"Site {site} does not assign {expected}: {ast.unparse(statement.value)}")
    statement.value = _expression(replacement)


def _insert_memory(tree: ast.Module) -> None:
    """Add ``first_scaler = {}`` after the module's imports."""
    if any(isinstance(node, ast.Name) and node.id == MEMORY for node in ast.walk(tree)):
        raise CatalogueError(f"The subject already uses the name {MEMORY}")
    position = max(index for index, node in enumerate(tree.body) if isinstance(node, (ast.Import, ast.ImportFrom)))
    tree.body.insert(position + 1, ast.parse(f"{MEMORY} = {{}}").body[0])


def _remove_recursion_update(source: str, tree: ast.Module) -> None:
    statement, body, _ = _site(source, tree, "R")
    if not _same_statement(statement, "history = np.vstack([history, value])"):
        raise CatalogueError(f"Site R does not append the value to the history: {ast.unparse(statement)}")
    if len(body) == 1:
        raise CatalogueError("Removing site R would leave an empty block")
    body.remove(statement)


def _horizon_off_by_one(source: str, tree: ast.Module) -> None:
    statement, _, _ = _site(source, tree, "H")
    if not isinstance(statement, ast.For) or not _same_expression(statement.iter, "range(len(future_index))"):
        raise CatalogueError("Site H must be the loop for step in range(len(future_index))")
    function = _function(tree, statement)
    returns = [node for node in ast.walk(function) if isinstance(node, ast.Return)]
    if len(returns) != 1 or not isinstance(returns[0].value, ast.Call):
        raise CatalogueError("The forecasting loop's function must return one call")
    points = [keyword for keyword in returns[0].value.keywords
              if keyword.arg == "point" and _same_expression(keyword.value, "np.asarray(points)")]
    if len(points) != 1:
        raise CatalogueError("The returned forecast must take point=np.asarray(points)")
    statement.iter = _expression("range(len(future_index) + 1)")
    points[0].value = _expression("np.asarray(points[1:])")


def _swap_recursive_targets(source: str, tree: ast.Module) -> None:
    site_lines(source)
    predict = _method(tree, "SubjectForecaster", "predict")
    found = [(statement, body) for owner, body in _bodies(predict) for statement in body
             if _same_statement(statement, "value = output[0]")]
    if len(found) != 1:
        raise CatalogueError(f"Expected one statement value = output[0] in predict, found {len(found)}")
    statement, body = found[0]
    body.insert(body.index(statement) + 1, ast.parse("value = value[::-1].copy()").body[0])


def _transformed(source: str, change) -> str:
    tree = ast.parse(source)
    change(tree)
    return ast.unparse(ast.fix_missing_locations(tree)) + "\n"


def apply_operator(subject_source: str, operator: str) -> str:
    """The mutant source of one catalogue operator."""
    for name, site, expected, replacement in VALUE_OPERATORS:
        if name == operator:
            def change(tree, site=site, expected=expected, replacement=replacement):
                if site in ("S-x", "S-y"):
                    _insert_memory(tree)
                _replace_value(subject_source, tree, site, expected, replacement)
            return _transformed(subject_source, change)
    if operator == "recursion_state_not_updated":
        return _transformed(subject_source, lambda tree: _remove_recursion_update(subject_source, tree))
    if operator == "horizon_index_off_by_one":
        return _transformed(subject_source, lambda tree: _horizon_off_by_one(subject_source, tree))
    raise KeyError(f"No catalogue operator named {operator!r}")


def catalogue_sources(subject_source: str) -> dict[str, str]:
    """The 16 catalogue mutant sources, keyed by operator id in protocol order."""
    return {operator: apply_operator(subject_source, operator) for operator in OPERATORS}


def study2_bug_sources(subject_source: str) -> dict[str, str]:
    """The 4 Study 2 bugs on the subject; three share their source with a catalogue operator."""
    sources = {bug: apply_operator(subject_source, STUDY2_BUG_SAME_SOURCE[bug]) for bug in STUDY2_BUGS[:3]}
    sources["swap_recursive_targets"] = _transformed(subject_source,
                                                     lambda tree: _swap_recursive_targets(subject_source, tree))
    return {bug: sources[bug] for bug in STUDY2_BUGS}

