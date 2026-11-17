"""Protocol 3.0 pieces shared by the Study 3 runners: the protocol file, its seeds, the start conditions, the record.

Nothing in this module computes a study outcome.
"""
from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
import platform

from .common import protect_output, sha256

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "experiments/protocol-v3.json"
RESULTS = ROOT / "experiments/results-v3"
EARLIER_PROTOCOLS = ("protocol.json", "protocol-v2.json", "protocol-v2-followup.json", "protocol-v2-addendum.json")
SEED_FLOOR, SEED_CEILING = 7_000_000, 15_000_000
PARTS = ("generation", "part-1", "part-2", "part-3", "part-4", "part-5", "zoo")


def load_protocol(path: Path = PROTOCOL) -> dict:
    """Read the protocol and check its seed table."""
    protocol = json.loads(Path(path).read_text())
    if protocol.get("protocol_version") != "3.0":
        raise ValueError("Not a protocol 3.0 file")
    check_seed_table(protocol)
    return protocol


def _block(protocol: dict, namespace: str) -> tuple[int, int, int, int]:
    blocks = protocol["seeds"]["case_seed_blocks"]
    if namespace == "format" or namespace not in blocks:
        raise KeyError(f"No case seed block named {namespace!r}")
    base, stride, n_blocks, count = blocks[namespace]
    return int(base), int(stride), int(n_blocks), int(count)


def case_seed(protocol: dict, namespace: str, index: int, block: int = 0) -> int:
    """The seed of case ``index`` in block ``block`` (a condition or a suite) of a registered namespace."""
    base, stride, n_blocks, count = _block(protocol, namespace)
    if not 0 <= block < n_blocks or not 0 <= index < count:
        raise IndexError(f"{namespace} has {n_blocks} blocks of {count} seeds; got block {block}, index {index}")
    return base + stride * block + index


def case_seeds(protocol: dict, namespace: str, block: int = 0, count: int | None = None) -> list[int]:
    """The first ``count`` seeds (all of them by default) of one block of a registered namespace."""
    size = _block(protocol, namespace)[3]
    return [case_seed(protocol, namespace, index, block) for index in range(size if count is None else count)]


def generator_tag(protocol: dict, name: str) -> int:
    tags = protocol["seeds"]["generator_tags"]
    if name == "format" or name not in tags:
        raise KeyError(f"No generator tag named {name!r}")
    return int(tags[name])


def check_seed_table(protocol: dict) -> None:
    """Refuse a seed table whose blocks overlap, leave the Study 3 range, or contain a generator tag."""
    spans = []
    for namespace in protocol["seeds"]["case_seed_blocks"]:
        if namespace == "format":
            continue
        base, stride, n_blocks, count = _block(protocol, namespace)
        if min(stride, n_blocks, count) < 1 or (n_blocks > 1 and count > stride):
            raise ValueError(f"Seed block {namespace} overlaps itself")
        for block in range(n_blocks):
            spans.append((base + stride * block, base + stride * block + count - 1, namespace))
    spans.sort()
    if not spans or spans[0][0] < SEED_FLOOR or spans[-1][1] >= SEED_CEILING:
        raise ValueError(f"Study 3 case seeds must lie in [{SEED_FLOOR}, {SEED_CEILING})")
    for (_, stop, first), (start, _, second) in zip(spans, spans[1:]):
        if start <= stop:
            raise ValueError(f"Seed blocks {first} and {second} overlap")
    tags = [(name, int(value)) for name, value in protocol["seeds"]["generator_tags"].items() if name != "format"]
    if len({value for _, value in tags}) != len(tags):
        raise ValueError("Generator tags repeat")
    for name, value in tags:
        if any(start <= value <= stop for start, stop, _ in spans):
            raise ValueError(f"Generator tag {name} falls inside a case seed block")


def frozen_file_problems(protocol: dict) -> list[str]:
    """Frozen files of Part 1 that are missing, unhashed or changed."""
    subject = protocol["part_1_generated_mutants"]["subject"]
    recorded, problems = subject["frozen_files_sha256"], []
    for name in subject["frozen_files"]:
        path = ROOT / name
        if not path.is_file():
            problems.append(f"{name} does not exist")
        elif name not in recorded:
            problems.append(f"{name} has no digest in frozen_files_sha256")
        elif sha256(path) != recorded[name]:
            problems.append(f"{name} differs from its registered digest")
    return problems


def manifest_problems(protocol: dict) -> list[str]:
    """Why the mutant manifest may not be used: its digest is not registered, it is missing or it has changed."""
    registered = protocol["part_1_generated_mutants"]["sources"]["manifest"]
    path = ROOT / registered["path"]
    if not registered.get("sha256"):
        return ["part_1_generated_mutants.sources.manifest.sha256 is not filled in"]
    if not path.is_file():
        return [f"{registered['path']} does not exist"]
    if sha256(path) != registered["sha256"]:
        return [f"{registered['path']} differs from its registered digest"]
    return []


def unmet_start_conditions(protocol: dict, part: str) -> list[str]:
    """Why ``part`` may not run yet; an empty list means it may.

    "generation" writes the mutant sources and their manifest; Parts 1 and 2 execute them, so they also need the
    manifest's digest in the protocol.
    """
    if part not in PARTS:
        raise ValueError(f"part must be one of {PARTS}")
    registration, unmet = protocol["registration"], []
    if protocol.get("status") != "registered":
        unmet.append(f"status is {protocol.get('status')!r}, not 'registered'")
    if not registration.get("merge_commit") or not registration.get("registered_on"):
        unmet.append("registration.merge_commit and registration.registered_on are not filled in")
    if part in ("generation", "part-1", "part-2", "zoo"):
        unmet.extend(frozen_file_problems(protocol))
    if part in ("part-1", "part-2"):
        unmet.extend(manifest_problems(protocol))
    return unmet


def require_start(protocol: dict, part: str) -> None:
    """Raise unless every start condition of ``part`` holds. The runners call this before any computation."""
    unmet = unmet_start_conditions(protocol, part)
    if unmet:
        raise RuntimeError(f"Study 3 {part} may not start: " + "; ".join(unmet))


def environment(packages: tuple[str, ...] = ("numpy", "scipy", "pandas", "scikit-learn", "threadpoolctl")) -> dict:
    versions = {}
    for name in packages:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": platform.python_version(), "platform": platform.platform(), "versions": versions}


def write_record(output: Path, results: dict, *, protocol_path: Path = PROTOCOL, sources: tuple[Path, ...] = (),
                 runtime: dict | None = None, extra: dict | None = None, force: bool = False) -> dict:
    """Write results.json, manifest.json and runtime.json as the earlier runners do, and return the manifest.

    ``results`` holds numbers only. Wall times and thread pools go in ``runtime`` and never enter results.json.
    """
    output = Path(output)
    protect_output(output, force=force)
    output.mkdir(parents=True, exist_ok=True)
    numerical = output / "results.json"
    numerical.write_text(json.dumps(results, indent=2, sort_keys=True, allow_nan=False) + "\n")
    protocol = json.loads(Path(protocol_path).read_text())
    manifest = {"protocol_sha256": sha256(protocol_path), "results_sha256": sha256(numerical),
                "frozen_inputs_sha256": {item["path"]: sha256(ROOT / item["path"])
                                         for item in protocol["frozen_inputs"].values()},
                "source_files": {str(Path(path).resolve().relative_to(ROOT)): sha256(path) for path in sources},
                **environment(), **(extra or {})}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
    if runtime is not None:
        (output / "runtime.json").write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n")
    return manifest
