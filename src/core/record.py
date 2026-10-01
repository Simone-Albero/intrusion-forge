import hashlib
import json
import logging
import shutil
from pathlib import Path

from src.core.utils import first_difference, load_from_json, save_to_json

RECORD = "record.json"

logger = logging.getLogger(__name__)


def stage_id(schema: int, config: dict, inputs: dict) -> str:
    """Identifier of what a stage builds: its schema, its config and the ids it reads."""
    blob = json.dumps(
        {"schema": schema, "config": config, "inputs": inputs}, sort_keys=True
    )
    return hashlib.blake2b(blob.encode(), digest_size=8).hexdigest()


def write_record(stage_dir: Path, *, schema: int, config: dict, inputs: dict) -> str:
    """Write the stage's record, the last file it leaves: it vouches for the others."""
    record = {
        "schema": schema,
        "config": config,
        "inputs": inputs,
        "id": stage_id(schema, config, inputs),
    }
    save_to_json(record, Path(stage_dir) / RECORD)
    return record["id"]


def read_record(stage_dir: Path) -> dict:
    """The stage's record; a stage that never finished has none."""
    path = Path(stage_dir) / RECORD
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}: run `make {path.parent.name}` first.")
    return load_from_json(path)


def clear_dir(stage_dir: Path, *, keep: tuple[str, ...] = ()) -> None:
    """Empty a stage's folder, sparing the entries named in `keep`."""
    stage_dir = Path(stage_dir)
    if stage_dir.exists():
        for child in stage_dir.iterdir():
            if child.name in keep:
                continue
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    stage_dir.mkdir(parents=True, exist_ok=True)


def is_current(
    stage_dir: Path,
    outputs: tuple[str, ...],
    *,
    schema: int,
    config: dict,
    inputs: dict,
    force: bool,
) -> bool:
    """True (and log) when every output exists and the record is what a rerun would write."""
    stage_dir = Path(stage_dir)
    name = stage_dir.name
    missing = [o for o in outputs if not (stage_dir / o).exists()]
    if force or len(missing) == len(outputs):
        return False
    if missing:
        logger.info(
            "[RECOMPUTE] %s: %d of %d outputs missing.",
            name,
            len(missing),
            len(outputs),
        )
        return False
    if not (stage_dir / RECORD).exists():
        logger.info("[RECOMPUTE] %s: no record of the inputs of its outputs.", name)
        return False
    previous = load_from_json(stage_dir / RECORD)
    # Named separately: the key that differs inside config or inputs says what changed.
    changed = (
        "schema"
        if previous["schema"] != schema
        else first_difference(previous["config"], config)
        or first_difference(previous["inputs"], inputs)
    )
    if changed is not None:
        logger.info("[RECOMPUTE] %s: its inputs changed (%s).", name, changed)
        return False
    logger.info(
        "[CACHED] %s: outputs present and unchanged (force=true to recompute).", name
    )
    return True
