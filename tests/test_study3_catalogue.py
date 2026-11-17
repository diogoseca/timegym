"""The Study 3 catalogue, tested statically: transformed source text against expected text. Nothing generated here is
imported or executed; every source is only parsed and compiled."""
import ast
import json

import pytest

from experiments import study3_catalogue as catalogue
from experiments import study3_common as common

SUBJECT = (common.ROOT / "experiments/study3_subject.py").read_text()

CLEAN = {
    "training_matrix": """def training_matrix(values, window, period):
    start = max(window, period)
    rows = []
    for t in range(start, len(values)):
        lags = values[t - window:t]
        seasonal = values[t - period]
        rows.append(np.concatenate([lags.ravel(), seasonal]))
    targets = values[start:]
    return (np.asarray(rows), targets)""",
    "prediction_row": """def prediction_row(history, window, period):
    lags = history[-window:]
    seasonal = history[-period]
    return np.concatenate([lags.ravel(), seasonal])[None, :]""",
    "SubjectForecaster.fit": """def fit(self, context):
    values = context.to_numpy(dtype=float)
    features, targets = training_matrix(values, self.window, self.period)
    if self.scaling == 'global':
        self.feature_center, self.feature_scale = fit_scaler(features)
        self.target_center, self.target_scale = fit_scaler(targets)
        features = apply_scaler(features, self.feature_center, self.feature_scale)
        targets = apply_scaler(targets, self.target_center, self.target_scale)
    self.regressor = self.make_regressor().fit(features, targets)
    self.history = values
    self.names = tuple(context.columns)
    return self""",
    "SubjectForecaster.predict": """def predict(self, future_index):
    history = self.history
    points = []
    for step in range(len(future_index)):
        row = prediction_row(history, self.window, self.period)
        if self.scaling == 'global':
            row = apply_scaler(row, self.feature_center, self.feature_scale)
        output = self.regressor.predict(row)
        if self.scaling == 'global':
            output = invert_scaler(output, self.target_center, self.target_scale)
        value = output[0]
        points.append(value)
        history = np.vstack([history, value])
    return Forecast(index=future_index.copy(), point=np.asarray(points), target_names=self.names)""",
}

# operator: (definition, [(clean line, mutated line or None to delete), ...], module statement added or None)
EXPECTED = {
    "lag_window_off_by_one_training": ("training_matrix", [
        ("        lags = values[t - window:t]\n", "        lags = values[t - window - 1:t - 1]\n")], None),
    "lag_window_off_by_one_prediction": ("prediction_row", [
        ("    lags = history[-window:]\n", "    lags = history[-window - 1:-1]\n")], None),
    "lag_window_includes_target_training": ("training_matrix", [
        ("        lags = values[t - window:t]\n", "        lags = values[t - window + 1:t + 1]\n")], None),
    "lag_order_reversed_training": ("training_matrix", [
        ("        lags = values[t - window:t]\n", "        lags = values[t - window:t][::-1]\n")], None),
    "lag_order_reversed_prediction": ("prediction_row", [
        ("    lags = history[-window:]\n", "    lags = history[-window:][::-1]\n")], None),
    "recursion_state_not_updated": ("SubjectForecaster.predict", [
        ("        history = np.vstack([history, value])\n", "")], None),
    "horizon_index_off_by_one": ("SubjectForecaster.predict", [
        ("    for step in range(len(future_index)):\n", "    for step in range(len(future_index) + 1):\n"),
        ("    return Forecast(index=future_index.copy(), point=np.asarray(points), target_names=self.names)",
         "    return Forecast(index=future_index.copy(), point=np.asarray(points[1:]), target_names=self.names)")],
        None),
    "target_columns_swapped": ("training_matrix", [
        ("    targets = values[start:]\n", "    targets = values[start:][:, ::-1]\n")], None),
    "series_columns_swapped_prediction": ("prediction_row", [
        ("    lags = history[-window:]\n", "    lags = history[-window:][:, ::-1]\n")], None),
    "feature_scaler_reused_across_contexts": ("SubjectForecaster.fit", [
        ("        self.feature_center, self.feature_scale = fit_scaler(features)\n",
         "        self.feature_center, self.feature_scale = "
         "first_scaler.setdefault('features', fit_scaler(features))\n")],
        "first_scaler = {}"),
    "target_scaler_reused_across_contexts": ("SubjectForecaster.fit", [
        ("        self.target_center, self.target_scale = fit_scaler(targets)\n",
         "        self.target_center, self.target_scale = "
         "first_scaler.setdefault('targets', fit_scaler(targets))\n")],
        "first_scaler = {}"),
    "inverse_transform_skipped": ("SubjectForecaster.predict", [
        ("            output = invert_scaler(output, self.target_center, self.target_scale)\n",
         "            output = output\n")], None),
    "inverse_transform_applied_twice": ("SubjectForecaster.predict", [
        ("            output = invert_scaler(output, self.target_center, self.target_scale)\n",
         "            output = invert_scaler(invert_scaler(output, self.target_center, self.target_scale), "
         "self.target_center, self.target_scale)\n")], None),
    "seasonal_period_off_by_one_training": ("training_matrix", [
        ("        seasonal = values[t - period]\n", "        seasonal = values[t - period - 1]\n")], None),
    "seasonal_period_off_by_one_prediction": ("prediction_row", [
        ("    seasonal = history[-period]\n", "    seasonal = history[-period - 1]\n")], None),
    "training_rows_dropped_from_wrong_end": ("training_matrix", [
        ("    targets = values[start:]\n", "    targets = values[:len(values) - start]\n")], None),
    "swap_recursive_targets": ("SubjectForecaster.predict", [
        ("        value = output[0]\n", "        value = output[0]\n        value = value[::-1].copy()\n")], None),
}


def parts(source):
    """Module statements, functions, and the class's other statements and methods, each as unparsed text."""
    tree = ast.parse(source)
    definitions = (ast.FunctionDef, ast.ClassDef)
    out = {"module": [ast.unparse(node) for node in tree.body if not isinstance(node, definitions)]}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            out[node.name] = ast.unparse(node)
        elif isinstance(node, ast.ClassDef):
            out[node.name] = [ast.unparse(item) for item in node.body if not isinstance(item, ast.FunctionDef)]
            out.update({f"{node.name}.{item.name}": ast.unparse(item) for item in node.body
                        if isinstance(item, ast.FunctionDef)})
    return out


def expected_parts(name):
    definition, edits, added = EXPECTED[name]
    out = parts(SUBJECT)
    text = CLEAN[definition] + "\n"
    for old, new in edits:
        assert text.count(old) == 1, old
        text = text.replace(old, new)
    out[definition] = text.rstrip("\n")
    if added:
        after = 1 + max(i for i, statement in enumerate(out["module"]) if statement.startswith(("import ", "from ")))
        out["module"] = out["module"][:after] + [added] + out["module"][after:]
    return out


def test_the_clean_definitions_are_the_frozen_structure():
    clean = parts(SUBJECT)
    for name, text in CLEAN.items():
        assert clean[name] == text, name
    assert clean["module"][1:] == ["import numpy as np", "from timegym import Forecast"]
    assert sorted(catalogue.site_lines(SUBJECT)) == sorted(catalogue.SITES)


def test_the_catalogue_matches_the_protocol():
    protocol = json.loads(common.PROTOCOL.read_text())["part_1_generated_mutants"]["sources"]
    registered = protocol["catalogue"]["operators"]
    assert list(catalogue.OPERATORS) == [item["id"] for item in registered]
    assert catalogue.OPERATOR_SITES == {item["id"]: item["site"] for item in registered}
    assert list(catalogue.STUDY2_BUGS) == [item["id"] for item in protocol["study2_bugs"]["bugs"]]
    assert catalogue.STUDY2_BUG_SAME_SOURCE == {item["id"]: item["same_source_as"]
                                                for item in protocol["study2_bugs"]["bugs"] if "same_source_as" in item}
    assert sorted(EXPECTED) == sorted(catalogue.OPERATORS + ("swap_recursive_targets",))


@pytest.mark.parametrize("name", catalogue.OPERATORS)
def test_each_operator_changes_exactly_its_site(name):
    source = catalogue.catalogue_sources(SUBJECT)[name]
    compile(source, name, "exec")
    assert parts(source) == expected_parts(name)


def test_the_study2_bugs():
    study2, sources = catalogue.study2_bug_sources(SUBJECT), catalogue.catalogue_sources(SUBJECT)
    for bug, operator in catalogue.STUDY2_BUG_SAME_SOURCE.items():
        assert study2[bug] == sources[operator], bug
    compile(study2["swap_recursive_targets"], "swap_recursive_targets", "exec")
    assert parts(study2["swap_recursive_targets"]) == expected_parts("swap_recursive_targets")
    distinct = {ast.unparse(ast.parse(source)) for source in list(study2.values()) + list(sources.values())}
    assert len(distinct) == 17 and ast.unparse(ast.parse(SUBJECT)) not in distinct


@pytest.mark.parametrize("edit, message", [
    (("lags = values[t - window:t]  # site: T-lag", "lags = values[t - window:t + 0]  # site: T-lag"),
     "does not assign"),
    (("  # site: H", ""), "missing"),
    (("  # site: S-y", "  # site: S-x"), "marked on lines"),
    (("  # site: R", "  # site: Q"), "Unknown site"),
    (("history = np.vstack([history, value])  # site: R", "history = np.vstack((history, value))  # site: R"),
     "does not append"),
    (("value = output[0]\n", "value = output[-1]\n"), "value = output"),
])
def test_an_unexpected_subject_is_refused(edit, message):
    old, new = edit
    assert SUBJECT.count(old) == 1
    changed = SUBJECT.replace(old, new)
    with pytest.raises(catalogue.CatalogueError, match=message):
        catalogue.catalogue_sources(changed)
        catalogue.study2_bug_sources(changed)


def test_the_catalogue_uses_the_standard_library_only():
    tree = ast.parse((common.ROOT / "experiments/study3_catalogue.py").read_text())
    imported = {alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import)
                for alias in node.names}
    imported |= {node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert imported <= {"__future__", "ast", "io", "re", "tokenize"}
