import logging
import shutil
import uuid
from pathlib import Path

from src.core.utils import first_difference, load_from_json, save_to_json

RECORD = "record.json"

logger = logging.getLogger(__name__)


def write_record(stage_dir: Path, *, config: dict, inputs: dict) -> str:
    """Write the stage's record, the last file it leaves: it vouches for the others."""
    # A fresh id on every write: a stage rewritten under the same config still
    # invalidates the stages that read it.
    record = {"config": config, "inputs": inputs, "id": uuid.uuid4().hex}
    save_to_json(record, Path(stage_dir) / RECORD)
    return record["id"]


def read_record(stage_dir: Path) -> dict:
    """The stage's record; a stage that never finished has none."""
    path = Path(stage_dir) / RECORD
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}: run `make {path.parent.name}` first.")
    return load_from_json(path)


def clear_dir(stage_dir: Path) -> None:
    """Empty a stage's folder."""
    stage_dir = Path(stage_dir)
    if stage_dir.exists():
        for child in stage_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    stage_dir.mkdir(parents=True, exist_ok=True)


def is_current(stage_dir: Path, *, config: dict, inputs: dict, force: bool) -> bool:
    """True (and log) when the record holds this config and these inputs."""
    stage_dir = Path(stage_dir)
    name = stage_dir.name
    if force or not (stage_dir / RECORD).exists():
        return False
    previous = load_from_json(stage_dir / RECORD)
    # Named separately: the key that differs inside config or inputs says what changed.
    changed = first_difference(previous["config"], config) or first_difference(
        previous["inputs"], inputs
    )
    if changed is not None:
        logger.info("[RECOMPUTE] %s: its inputs changed (%s).", name, changed)
        return False
    logger.info("[CACHED] %s: unchanged (force=true to recompute).", name)
    return True
