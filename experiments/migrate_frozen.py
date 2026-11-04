"""Apply recorded offline migrations to frozen records without running models.

``schema-reduction`` applies the EX4/EX17 schema reduction. Use --root with a
checkout containing the original records and updated protocols. The original
manifest evidence is retained under record_migration; source hashes continue to
identify the historical execution, not this record transformation.

``loss-keys`` renames the supplement's paired loss fields from repair_minus_faulty_*
to clean_minus_faulty_* and logs the rename under loss_key_migration. Field order
follows the sorted serialization the runner writes, and no value changes.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

from .common import sha256
from .run_supplement import export_tables


def reduce_supplement(results: dict) -> dict:
    results = deepcopy(results)
    model = results["model"]
    clean = {(row["condition"], row["seed"]): row for row in model["records"] if row["branch"] == "clean"}
    repaired = [row for row in model["records"] if row["branch"] == "repaired"]
    if len(repaired) != len(clean):
        raise ValueError("Expected one duplicate repaired record for every clean case")
    for row in repaired:
        if {**row, "branch": "clean"} != clean[(row["condition"], row["seed"])]:
            raise ValueError("Cannot remove a repaired record that differs from its clean pair")
    model["records"] = [row for row in model["records"] if row["branch"] != "repaired"]
    for row in model["condition_summaries"]:
        if row.pop("repaired") != row["clean"]:
            raise ValueError("Repaired summary differs from clean summary")
        del row["paired_alarm_change"]
    for row in model["ablation_counts"]:
        if row.pop("repair_alarms") != row["clean_alarms"] or row.pop("repair_n") != row["clean_n"]:
            raise ValueError("Repaired ablation differs from clean ablation")
    for example in model["examples"]:
        if example["branch_points"].pop("repaired") != example["branch_points"]["clean"]:
            raise ValueError("Repaired example differs from clean example")
    model["cost_counts"]["sklearn_fits"] -= 4 * len(repaired)
    return results


def migrate(root: Path) -> None:
    for folder, protocol_name in (("results", "protocol.json"), ("results-v2", "protocol-v2.json")):
        directory = root / "experiments" / folder
        path, manifest_path = directory / "results.json", directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if "record_migration" in manifest:
            raise ValueError(f"{folder} has already been migrated")
        before = sha256(path)
        if manifest.get("results_sha256", before) != before:
            raise ValueError("Original results do not match their manifest")
        original = deepcopy(manifest)
        original_cost_counts = None
        protocol_path = root / "experiments" / protocol_name
        protocol = json.loads(protocol_path.read_text())
        manifest.update(protocol=protocol, protocol_sha256=sha256(protocol_path))
        if folder == "results-v2":
            previous = json.loads(path.read_text())
            original_cost_counts = deepcopy(previous["model"]["cost_counts"])
            results = reduce_supplement(previous)
            results["protocol_sha256"] = manifest["protocol_sha256"]
            path.write_text(json.dumps(results, indent=2, sort_keys=True, allow_nan=False)+"\n")
            export_tables(results, directory)
        else:
            manifest["versions"].pop("scikit-learn", None)
        manifest["results_sha256"] = sha256(path)
        manifest["record_migration"] = {
            "decisions": ["EX1", "EX3"] if folder == "results" else ["EX4", "EX15", "EX16", "EX17"],
            "method": "Offline metadata/schema migration; no study rerun and no retained measurements changed.",
            "original_results_sha256": before,
            "original_manifest": original,
            **({"original_model_cost_counts": original_cost_counts} if original_cost_counts is not None else {}),
            "migration_script_sha256": sha256(Path(__file__)),
            "cost_note": "Operation counts now describe retained branches; original executed counts and timing remain historical."
                         if folder == "results-v2" else "Pilot numerical results are byte-identical.",
        }
        manifest["source_note"] = ("Source fingerprints and environment describe the historical execution. "
                                   "record_migration retains original provenance; updated protocol/schema were applied offline.")
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False)+"\n")


LOSS_KEYS = {
    "repair_minus_faulty_mse": "clean_minus_faulty_mse",
    "repair_minus_faulty_reference_mse": "clean_minus_faulty_reference_mse",
}


def rename_loss_keys(results: dict) -> dict:
    results = deepcopy(results)
    renamed = 0
    for row in results["model"]["loss_summaries"]:
        for old, new in LOSS_KEYS.items():
            if new in row:
                raise ValueError(f"Loss summary already has {new}")
            row[new] = row.pop(old)
            renamed += 1
    if renamed != 2 * len(results["model"]["loss_summaries"]) or not renamed:
        raise ValueError("Expected both paired loss fields in every loss summary")
    return results


def migrate_loss_keys(root: Path) -> None:
    directory = root / "experiments" / "results-v2"
    path, manifest_path = directory / "results.json", directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if "loss_key_migration" in manifest:
        raise ValueError("results-v2 loss keys have already been migrated")
    before = sha256(path)
    if manifest["results_sha256"] != before:
        raise ValueError("Results do not match their manifest")
    previous = json.loads(path.read_text())
    results = rename_loss_keys(previous)
    if {**results["model"], "loss_summaries": None} != {**previous["model"], "loss_summaries": None} \
            or {key: value for key, value in results.items() if key != "model"} != \
            {key: value for key, value in previous.items() if key != "model"}:
        raise ValueError("Loss-key migration changed fields outside the loss summaries")
    for new_row, old_row in zip(results["model"]["loss_summaries"], previous["model"]["loss_summaries"]):
        if {LOSS_KEYS.get(key, key): value for key, value in old_row.items()} != new_row:
            raise ValueError("Loss-key migration changed a value")
    path.write_text(json.dumps(results, indent=2, sort_keys=True, allow_nan=False)+"\n")
    export_tables(results, directory)
    manifest["results_sha256"] = sha256(path)
    manifest["loss_key_migration"] = {
        "decision": "Owner decision, 3 November 2026",
        "method": "Offline field rename; no study rerun and no numerical value changed.",
        "renamed_fields": LOSS_KEYS,
        "original_results_sha256": before,
        "migration_script_sha256": sha256(Path(__file__)),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--step", choices=("schema-reduction", "loss-keys"), default="schema-reduction")
    args = parser.parse_args()
    if args.step == "loss-keys":
        migrate_loss_keys(args.root)
    else:
        migrate(args.root)


if __name__ == "__main__":
    main()
