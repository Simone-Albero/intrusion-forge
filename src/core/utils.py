import functools
import json
import logging
import math
import time
from collections.abc import Callable, Iterable
from pathlib import Path

import joblib


def _nan_to_none(obj: object) -> object:
    """Recursively replace float NaN with None for JSON serialization."""
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, dict):
        return {k: _nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_nan_to_none(v) for v in obj]
    return obj


def save_to_json(data: object, file_path: str | Path) -> None:
    """Save data to a JSON file."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "w") as f:
        json.dump(_nan_to_none(data), f, indent=4)


def load_from_json(file_path: str | Path) -> object:
    """Load data from a JSON file."""
    file_path = Path(file_path)
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def save_to_joblib(data: object, file_path: str | Path) -> None:
    """Save data (typically a sklearn estimator) via joblib."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(data, file_path)


def load_from_joblib(file_path: str | Path) -> object:
    """Load data previously written with save_to_joblib."""
    return joblib.load(Path(file_path))


_TIMING_RECORDS: list[dict] = []


def timed(fn: Callable) -> Callable:
    """Measure the wall-clock time of `fn`, log it and record it for `flush_timing`."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        output = fn(*args, **kwargs)
        elapsed_s = time.perf_counter() - t0
        logging.getLogger(fn.__module__).info(
            "%s completed in %.2f s", fn.__qualname__, elapsed_s
        )
        _TIMING_RECORDS.append({"function": fn.__qualname__, "duration_s": elapsed_s})
        return output

    return wrapper


def flush_timing(path: str | Path) -> None:
    """Append accumulated timing records to JSON file and clear the in-memory list."""
    path = Path(path)
    existing = load_from_json(path) if path.exists() else []
    save_to_json(existing + _TIMING_RECORDS, path)
    _TIMING_RECORDS.clear()


def skip_if_unchanged(
    outputs: Iterable[Path],
    record: Path,
    fingerprint: dict,
    *,
    force: bool,
    stage_name: str,
) -> bool:
    """True (and log) when every output exists and `record` holds this `fingerprint`."""
    if force or not all(Path(p).exists() for p in outputs):
        return False
    log = logging.getLogger(__name__)
    if not record.exists():
        log.info("[RECOMPUTE] %s: no record of the inputs of its outputs.", stage_name)
        return False
    previous = load_from_json(record)
    # Another schema differs everywhere: its first key would name the wrong cause.
    if previous.get("schema") != fingerprint["schema"]:
        changed = "schema"
    else:
        changed = first_difference(previous, fingerprint)
    if changed is not None:
        log.info("[RECOMPUTE] %s: its inputs changed (%s).", stage_name, changed)
        return False
    log.info(
        "[STAGE-SKIP] Skipping %s — outputs present and unchanged "
        "(force=true to recompute).",
        stage_name,
    )
    return True


def first_difference(previous: dict, current: dict) -> str | None:
    """Name a key whose value differs (missing counts as None); None if all match."""
    for key in sorted(set(previous) | set(current)):
        if previous.get(key) != current.get(key):
            return key
    return None
