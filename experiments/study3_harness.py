"""One unit of work of Study 3, Parts 1 and 2 (protocol 3.0, shared.unit_of_work), and the parent that runs units.

A unit is one source (the clean subject, a Study 2 bug or a mutant) with one learner in one condition. The parent
starts each unit in a fresh spawned process and enforces its time limit; the child loads the source as a module
once, fits and forecasts once on the warm-up context (the condition's first equivalence input) and discards the
result, then takes its stages in the order given (equivalence inputs, the calibration cases of the clean subject,
evaluation cases) and, within a case, the fixed order: forecast, output checks, scalar shift check, per-target
shift check, second implementation, calibrated shift check. The output checks run on every case; the scores
(squared error from the observed future and from the answer key) and the other checks only where the spec names
them, never on the equivalence inputs. It writes one JSON line per case as soon as the case is finished, so a
unit stopped at its time limit leaves every case it finished.

Everything a unit computes depends only on its specification: the source text, the learner module (its build and
SeedStream), the condition, the seeds and the settings. The training seed stream of a case is
SeedStream([learner_training tag, learner index, case seed]); the forecast and the calibrated shift check share it
(draws 1 and 2-41), and the scalar and per-target shift checks each get a fresh copy of it, so that their refits do
not move the calibrated check's draws. Numerical libraries run on one thread: the child loads them after it
starts, so it first makes them start with one thread (experiments/study3_threads.py), limits them with
threadpoolctl once the source and the learner are loaded, and before its last line reads every loaded thread pool
back; a pool with more than one thread stops the run, and the last line lists the pools. A unit never reads
the state of another unit, so results do not depend on the number of workers. Wall times are written in the
"seconds" fields only, which the runner keeps out of numerical records.

Beside a unit's JSON lines the parent writes how the unit ended (<output>.outcome.json: finished, stopped at its
limit, crashed, its wall time and limit, a digest of its specification and limit, and a digest of its JSON lines).
A run that was stopped (a machine that shut down) can then be continued: with resume=True, run_unit returns the
stored result of a unit that ended under the same specification and limit, and whose JSON lines are unchanged,
instead of running it again. A unit that crashed is always run again, because a machine that shuts down also ends
a unit's process without a time-out.

Spec keys: output (JSON lines path), source, module_name, learner {module, name, index, tag}, condition (a
Scenario definition with an id), warmup_seed, stages [[stage name, [seeds]], ...], checks (a subset of CHECKS),
reference ("module:function" or None), settings {atol, rtol, scalar_shift, target_shifts, refits, alpha},
mode ("cases", "classification_public" or "classification_calibrated") and, optionally, stage_checks
{stage: checks}, which replaces checks for the named stages.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib
import importlib.util
import json
import math
import multiprocessing
import os
from pathlib import Path
import sys
import time
import traceback
import types
import warnings

from .common import scenario_settings
from .study3_threads import require_one_thread, start_with_one_thread

CHECKS = ("scores", "scalar_shift", "target_shift", "second_implementation", "calibrated_shift")
MODES = ("cases", "classification_public", "classification_calibrated")


def resolve(reference: str):
    """``package.module:name`` or ``/path/to/file.py:name`` to the named object."""
    location, _, name = reference.rpartition(":")
    if location.endswith(".py"):
        module_name = "study3_loaded_" + Path(location).stem
        module = sys.modules.get(module_name)
        if module is None:
            spec = importlib.util.spec_from_file_location(module_name, location)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
    else:
        module = importlib.import_module(location)
    return getattr(module, name) if name else module


def load_source(source: str, module_name: str) -> types.ModuleType:
    """Compile and execute ``source`` as a new module registered under ``module_name``."""
    code = compile(source, f"<{module_name}>", "exec")
    module = types.ModuleType(module_name)
    module.__file__ = f"<{module_name}>"
    sys.modules[module_name] = module
    exec(code, module.__dict__)
    return module


def _error(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"[:500]


def _json_value(value):
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    return value


def describe_forecast(forecast) -> dict:
    """Index, target names, shape and finite points of a forecast, as JSON values."""
    import numpy as np

    try:
        point = None if forecast.point is None else np.asarray(forecast.point, dtype=float)
    except (TypeError, ValueError):
        point = None
    finite = point is not None and bool(np.isfinite(point).all())
    try:
        index = [_json_value(label.item() if hasattr(label, "item") else label) for label in forecast.index]
    except TypeError:
        index = None
    names = None if forecast.target_names is None else [str(name) for name in forecast.target_names]
    return {"index": index, "target_names": names, "shape": None if point is None else list(point.shape),
            "finite": finite, "point": point.tolist() if finite else None}


class _Unit:
    """The state of one running unit in the child process."""

    def __init__(self, spec: dict, module: types.ModuleType):
        from timegym import Scenario

        self.spec, self.module = spec, module
        self.settings = spec.get("settings", {})
        learner = spec["learner"]
        learner_module = resolve(learner["module"] + ":")
        self.build, self.stream = learner_module.build, learner_module.SeedStream
        self.learner_name, self.learner_index, self.tag = learner["name"], learner["index"], learner["tag"]
        condition = {key: value for key, value in spec["condition"].items() if key != "id"}
        self.scenario = Scenario(**scenario_settings(condition))
        self.n_targets = int(condition.get("n_targets", 1))
        self.reference = resolve(spec["reference"]) if spec.get("reference") else None

    def pipeline(self, seed: int):
        from timegym import Pipeline

        stream = self.stream([self.tag, self.learner_index, seed])
        return Pipeline(self.build(self.learner_name, stream=stream, n_targets=self.n_targets, subject=self.module))

    def warm_up(self) -> dict:
        case = self.scenario.generate(seed=self.spec["warmup_seed"])
        started = time.perf_counter()
        try:
            self.pipeline(self.spec["warmup_seed"]).predict(case.task.history, case.task.future_index)
            error = None
        except Exception as exc:  # the warm-up result is discarded; an error is recorded
            error = _error(exc)
        return {"kind": "warmup", "seed": self.spec["warmup_seed"], "error": error,
                "seconds": time.perf_counter() - started}

    def case(self, stage: str, seed: int, checks: tuple[str, ...]) -> dict:
        import numpy as np
        from timegym.checks import check_shift_equivariance
        from timegym.metrics import mse

        case = self.scenario.generate(seed=seed)
        context, future = case.task.history, case.task.future_index
        settings = self.settings
        record = {"kind": "case", "stage": stage, "seed": seed, "raised": False, "error": None}
        started = time.perf_counter()
        pipeline = self.pipeline(seed)
        try:
            forecast = pipeline.predict(context, future)
        except Exception as exc:  # a raising mutant is an outcome, recorded in its own column
            record.update(raised=True, error=_error(exc), forecast_seconds=time.perf_counter() - started)
            return record
        record["forecast_seconds"] = time.perf_counter() - started
        record["forecast"] = describe_forecast(forecast)
        try:
            violations = sorted({item["code"] for item in forecast.validate(case.task)})
        except Exception as exc:  # a forecast so malformed that validation fails is an output alarm
            violations = [f"validation_failed: {_error(exc)}"]
        record["violations"] = violations
        valid = not violations and record["forecast"]["finite"]
        record["output_alarm"] = not valid
        point = np.asarray(record["forecast"]["point"], dtype=float) if valid else None
        finite_point = (np.asarray(record["forecast"]["point"], dtype=float) if record["forecast"]["finite"]
                        else np.full(1, np.nan))
        if "scores" in checks:
            record["scores"] = ({"loss": mse(case.observed.to_numpy(), point),
                                 "answer_key": mse(case.reference.mean, point)} if valid else None)
        if "scalar_shift" in checks:
            result = check_shift_equivariance(self.pipeline(seed), context, future, shift=settings["scalar_shift"],
                                              atol=settings["atol"], rtol=settings["rtol"])
            record["scalar_shift"] = {"status": result.status,
                                      "max_absolute_error": _json_value(result.evidence.get("max_absolute_error")),
                                      "error": result.message if result.status == "ERROR" else None}
        if "target_shift" in checks:
            from .model_faults import target_shift_agreement

            try:
                result = target_shift_agreement(self.pipeline(seed), context, future, forecast,
                                                shifts=settings["target_shifts"], atol=settings["atol"],
                                                rtol=settings["rtol"])
                record["target_shift"] = {"alarm": bool(result["alarm"]), "error": None,
                                          "max_absolute_error": _json_value(result["max_absolute_error"])}
            except Exception as exc:  # an exception on the shifted context is an alarm, reported with its error
                record["target_shift"] = {"alarm": True, "error": _error(exc), "max_absolute_error": None}
        if "second_implementation" in checks:
            from .model_faults import point_agreement

            expected = self.reference(context, future)
            result = point_agreement(finite_point, expected.point, atol=settings["atol"], rtol=settings["rtol"])
            record["second_implementation"] = {"alarm": bool(result["alarm"]),
                                               "max_absolute_error": _json_value(result["max_absolute_error"])}
        if "calibrated_shift" in checks:
            from .study3_shift import calibrated_shift_check

            record["calibrated_shift"] = calibrated_shift_check(
                pipeline, context, future, forecast=forecast, refits=settings["refits"],
                shift=settings["scalar_shift"], alpha=settings["alpha"], atol=settings["atol"],
                rtol=settings["rtol"])
        record["seconds"] = time.perf_counter() - started
        return record

    def classification_case(self, seed: int, mode: str) -> dict:
        from timegym.checks import check_shift_equivariance

        from .study3_shift import calibrated_shift_check

        case = self.scenario.generate(seed=seed)
        context, future = case.task.history, case.task.future_index
        settings, started = self.settings, time.perf_counter()
        if mode == "classification_public":
            result = check_shift_equivariance(self.pipeline(seed), context, future, shift=settings["scalar_shift"],
                                              atol=settings["atol"], rtol=settings["rtol"])
            check = {"status": result.status, "error": result.message if result.status == "ERROR" else None,
                     "max_absolute_error": _json_value(result.evidence.get("max_absolute_error"))}
        else:
            check = calibrated_shift_check(self.pipeline(seed), context, future, refits=settings["refits"],
                                           shift=settings["scalar_shift"], alpha=settings["alpha"],
                                           atol=settings["atol"], rtol=settings["rtol"])
        return {"kind": "case", "stage": "classification", "seed": seed, "check": mode, **check,
                "seconds": time.perf_counter() - started}


def unit_main(spec: dict) -> None:
    """The child process: run one unit and write its JSON lines to spec['output']."""
    started = time.perf_counter()
    # No numerical library is loaded yet, so a threadpoolctl limit set here alone would limit nothing.
    start_with_one_thread()
    with open(spec["output"], "w", encoding="utf-8") as handle:
        def emit(record):
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()

        from threadpoolctl import threadpool_limits

        with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                module = load_source(spec["source"], spec["module_name"])
            except SyntaxError as exc:
                emit({"kind": "load", "ok": False, "reason": "syntax", "error": _error(exc)})
                return
            except Exception as exc:  # any failure to import the mutated module makes it not runnable
                emit({"kind": "load", "ok": False, "reason": "import", "error": _error(exc)})
                return
            unit = _Unit(spec, module)
            threadpool_limits(limits=1)  # the source, the learner and their libraries are loaded now
            emit({"kind": "load", "ok": True, "reason": None, "error": None,
                  "seconds": time.perf_counter() - started})
            caught.clear()
            emit({**unit.warm_up(), "warnings": _warning_counts(caught)})
            caught.clear()
            mode = spec.get("mode", "cases")
            for stage, seeds in spec["stages"]:
                checks = () if stage == "screen" else tuple(spec.get("stage_checks", {}).get(stage, spec["checks"]))
                for seed in seeds:
                    record = unit.case(stage, seed, checks) if mode == "cases" else unit.classification_case(seed, mode)
                    record["warnings"] = _warning_counts(caught)
                    caught.clear()
                    emit(record)
            pools = require_one_thread()
            emit({"kind": "done", "seconds": time.perf_counter() - started,
                  "threads": {f"{pool['user_api']}:{pool['prefix']}:{index}": pool["num_threads"]
                              for index, pool in enumerate(pools)}})


def _warning_counts(caught) -> dict:
    counts: dict[str, int] = {}
    for item in caught:
        counts[item.category.__name__] = counts.get(item.category.__name__, 0) + 1
    return counts


def _child(spec: dict) -> None:
    try:
        unit_main(spec)
    except Exception:  # an error of the harness itself, not of the source: kept for the parent to raise
        with open(spec["output"], "a", encoding="utf-8") as handle:
            handle.write(json.dumps({"kind": "harness_error", "error": traceback.format_exc()}) + "\n")


def read_records(path: Path) -> list[dict]:
    """The JSON lines of a unit; a line cut off by a stopped process is ignored."""
    records = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            break
    return records


OUTCOME_KEYS = ("timed_out", "finished", "crashed", "exitcode", "seconds", "time_limit")


def outcome_path(spec: dict) -> Path:
    return Path(spec["output"] + ".outcome.json")


def _file_digest(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def spec_digest(spec: dict, time_limit: float | None) -> str:
    """Digest of everything a unit's result depends on: its specification and its time limit."""
    return hashlib.sha256(json.dumps([spec, time_limit], sort_keys=True).encode("utf-8")).hexdigest()


def stored_result(spec: dict, time_limit: float | None) -> dict | None:
    """The result of a unit that an earlier run ended under the same specification and limit, else None."""
    path = outcome_path(spec)
    if not path.is_file() or not Path(spec["output"]).is_file():
        return None
    try:
        outcome = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if outcome.get("spec_sha256") != spec_digest(spec, time_limit) or outcome.get("crashed", True):
        return None
    if outcome.get("records_sha256") != _file_digest(spec["output"]):
        return None
    return {"records": read_records(spec["output"]), **{key: outcome[key] for key in OUTCOME_KEYS}}


def run_unit(spec: dict, time_limit: float | None, resume: bool = False) -> dict:
    """Run one unit in a fresh spawned process; stop it at ``time_limit`` seconds.

    With ``resume``, a unit that an earlier run ended under the same specification and limit is not run again.
    """
    named = set(spec.get("checks", ())).union(*spec.get("stage_checks", {}).values())
    if spec.get("mode", "cases") not in MODES or named - set(CHECKS):
        raise ValueError("Unknown mode or check in a unit specification")
    stored = stored_result(spec, time_limit) if resume else None
    if stored is not None:
        return stored
    Path(spec["output"]).parent.mkdir(parents=True, exist_ok=True)
    outcome_path(spec).unlink(missing_ok=True)
    Path(spec["output"]).write_text("")
    process = multiprocessing.get_context("spawn").Process(target=_child, args=(spec,), daemon=True)
    started = time.perf_counter()
    process.start()
    process.join(time_limit)
    timed_out = process.is_alive()
    if timed_out:
        process.terminate()
        process.join(10)
        if process.is_alive():
            process.kill()
            process.join()
    seconds = time.perf_counter() - started
    records = read_records(spec["output"])
    errors = [record["error"] for record in records if record.get("kind") == "harness_error"]
    if errors:
        raise RuntimeError(f"The harness failed in unit {spec['output']}:\n{errors[0]}")
    finished = any(record.get("kind") == "done" for record in records)
    load_failed = bool(records) and records[0].get("kind") == "load" and not records[0]["ok"]
    # A process that ended without finishing and without a time-out (a source that exits the interpreter or
    # crashes it) is reported as crashed, with the cases it finished.
    crashed = not (timed_out or finished or load_failed)
    result = {"records": records, "timed_out": timed_out, "finished": finished, "crashed": crashed,
              "exitcode": process.exitcode, "seconds": seconds, "time_limit": time_limit}
    partial = Path(spec["output"] + ".outcome.partial")
    partial.write_text(json.dumps({"spec_sha256": spec_digest(spec, time_limit),
                                   "records_sha256": _file_digest(spec["output"]),
                                   **{key: result[key] for key in OUTCOME_KEYS}}, sort_keys=True) + "\n",
                       encoding="utf-8")
    os.replace(partial, outcome_path(spec))
    return result


def run_units(specs: list[dict], time_limits: list[float], workers: int = 1, resume: bool = False) -> list[dict]:
    """Run units in parallel worker processes and return their results in the order of ``specs``."""
    if len(specs) != len(time_limits):
        raise ValueError("One time limit per unit")
    if workers < 1:
        raise ValueError("workers must be positive")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda spec, limit: run_unit(spec, limit, resume), specs, time_limits))
