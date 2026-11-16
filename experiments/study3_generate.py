"""Write the mutant set of Study 3, Part 1 (protocol 3.0) and its manifest.

    python -m experiments.study3_generate --output experiments/results-v3/mutants

Run with the Cosmic Ray interpreter (cosmic-ray 8.9.0; standard library plus cosmic_ray, no NumPy). It refuses
when the protocol's start conditions for the generation step are not met. The commit that adds its output writes
the manifest's SHA-256 into part_1_generated_mutants.sources.manifest; Parts 1 and 2 refuse to start without it. It enumerates Cosmic Ray's mutants of the subject as
'cosmic-ray init' does (every operator that 'cosmic-ray operators' lists, each instantiated as init instantiates
it with no operator configuration; an operator that requires arguments yields no mutant there and is recorded as
such), applies each (operator, occurrence) with cosmic_ray.mutating.mutate_code, adds the 16 catalogue sources
and the 4 Study 2 bug sources of experiments/study3_catalogue.py, and deduplicates by ast.unparse(ast.parse(source)),
keeping every label that produced a source. No Cosmic Ray test command is used and no source is executed.

It writes the complete enumerated set: the stratified sample of sources.cosmic_ray.sampling_rule needs NumPy's
generator and is drawn by experiments/run_study3_mutants.py. Output: sources/<sha256>.py, one file per distinct
source (the text of the first label that produced it), and manifest.json (tool version and environment, the
operator list, the subject's digest, and per source its labels, origin, operator, family, occurrence or site,
definition and SHA-256).
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

ROOT = Path(__file__).resolve().parents[1]
SUBJECT = ROOT / "experiments/study3_subject.py"
CATALOGUE = ROOT / "experiments/study3_catalogue.py"


def normalized(source: str) -> str:
    return ast.unparse(ast.parse(source))


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def operator_names() -> tuple[str, ...]:
    """Every operator name that 'cosmic-ray operators' lists, in its order."""
    import cosmic_ray.plugins

    return tuple(cosmic_ray.plugins.operator_names())


def family(name: str) -> str:
    """The operator family: 'core/ReplaceBinaryOperator_Add_Sub' gives 'ReplaceBinaryOperator'."""
    return name.split("/", 1)[1].split("_", 1)[0]


def cosmic_ray_mutants(module_path: Path) -> tuple[list[dict], list[str]]:
    """Every (operator, occurrence) of the module with its mutated source, in the order of 'cosmic-ray init'.

    Returns the mutants and the operators skipped because they take arguments that init would need from the
    configuration.
    """
    import cosmic_ray.plugins
    from cosmic_ray.ast import ast_nodes, get_ast
    from cosmic_ray.ast.ast_query import ASTQuery
    from cosmic_ray.mutating import mutate_code

    source = Path(module_path).read_text(encoding="utf-8")
    module_ast = get_ast(source)
    mutants, skipped = [], []
    for name in operator_names():
        operator_class = cosmic_ray.plugins.get_operator(name)
        if operator_class.arguments():
            skipped.append(name)
            continue
        operator = operator_class()
        positions = [(node, start, end) for node in ast_nodes(module_ast)
                     for start, end in operator.mutation_positions(node)]
        for occurrence, (node, start, end) in enumerate(positions):
            mutated = mutate_code(source, operator_class(), occurrence)
            if mutated is None:
                raise RuntimeError(f"{name} occurrence {occurrence} was enumerated but applies no mutation")
            mutants.append({"origin": "cosmic_ray", "label": f"{name}#{occurrence}", "operator": name,
                            "family": family(name), "occurrence": occurrence, "start": list(start),
                            "end": list(end), "definition": ASTQuery(node).get_definition_name(),
                            "source": mutated})
    return mutants, skipped


def catalogue_mutants(subject_source: str, catalogue_path: Path = CATALOGUE) -> list[dict]:
    """The catalogue sources and the Study 2 bug sources, loaded from experiments/study3_catalogue.py by path."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("study3_catalogue_for_generation", catalogue_path)
    catalogue = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(catalogue)
    out = [{"origin": "catalogue", "label": name, "operator": name, "family": None,
            "site": catalogue.OPERATOR_SITES[name], "source": source}
           for name, source in catalogue.catalogue_sources(subject_source).items()]
    out += [{"origin": "study2", "label": name, "operator": name, "family": None,
             "same_source_as": catalogue.STUDY2_BUG_SAME_SOURCE.get(name), "source": source}
            for name, source in catalogue.study2_bug_sources(subject_source).items()]
    return out


def deduplicate(subject_source: str, mutants: list[dict]) -> list[dict]:
    """Distinct sources by normalized text, in order of first appearance, with every label that produced each."""
    clean = normalized(subject_source)
    distinct: dict[str, dict] = {}
    for mutant in mutants:
        key = normalized(mutant["source"])
        label = {key_: value for key_, value in mutant.items() if key_ != "source"}
        label["kind"] = label.pop("origin")
        entry = distinct.setdefault(key, {"source": mutant["source"], "labels": [],
                                          "identical_to_subject": key == clean})
        entry["labels"].append(label)
    return list(distinct.values())


def environment() -> dict:
    """Interpreter, platform and every installed distribution (the environment's pip freeze)."""
    installed = sorted(f"{dist.metadata['Name']}=={dist.version}" for dist in importlib.metadata.distributions())
    return {"python": platform.python_version(), "platform": platform.platform(), "installed": installed}


def generate(module_path: Path, output: Path, extra: list[dict] | None = None) -> dict:
    """Enumerate, deduplicate and write the sources of ``module_path`` and ``extra``; return the manifest."""
    output = Path(output)
    if (output / "manifest.json").exists():
        raise FileExistsError(f"{output / 'manifest.json'} already exists")
    subject_source = Path(module_path).read_text(encoding="utf-8")
    generated, skipped = cosmic_ray_mutants(module_path)
    distinct = deduplicate(subject_source, generated + list(extra or []))
    (output / "sources").mkdir(parents=True, exist_ok=True)
    entries = []
    for item in distinct:
        sha = digest(item["source"])
        name = f"sources/{sha}.py"
        (output / name).write_bytes(item["source"].encode("utf-8"))
        entries.append({"file": name, "sha256": sha, "labels": item["labels"],
                        "identical_to_subject": item["identical_to_subject"]})
    labels = [label for entry in entries for label in entry["labels"]]
    manifest = {
        "tool": {"name": "cosmic-ray", "version": importlib.metadata.version("cosmic-ray"),
                 "operators": list(operator_names()), "operators_without_mutants_needing_arguments": skipped,
                 "environment": environment()},
        "subject": {"path": str(Path(module_path).name), "sha256": hashlib.sha256(Path(module_path).read_bytes())
                    .hexdigest()},
        "counts": {"labels": len(labels), "distinct_sources": len(entries),
                   "by_kind": {kind: sum(label["kind"] == kind for label in labels)
                               for kind in ("cosmic_ray", "catalogue", "study2")}},
        "sources": entries,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def _require_start() -> None:
    """The start check of the generation step, from experiments/study3_common.py (standard library only)."""
    from . import study3_common

    study3_common.require_start(study3_common.load_protocol(), "generation")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    _require_start()
    subject_source = SUBJECT.read_text(encoding="utf-8")
    manifest = generate(SUBJECT, args.output, catalogue_mutants(subject_source))
    manifest["subject"]["path"] = str(SUBJECT.relative_to(ROOT))
    manifest["catalogue"] = {"path": str(CATALOGUE.relative_to(ROOT)),
                             "sha256": hashlib.sha256(CATALOGUE.read_bytes()).hexdigest()}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), **manifest["counts"]}, indent=2))


if __name__ == "__main__":
    main()
