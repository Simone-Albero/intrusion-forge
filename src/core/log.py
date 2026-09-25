import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.logging import RichHandler

from src.core.utils import save_to_json
from src.domain.plot.base import Plot


def setup_logger(log_file: str = "resources/logs.txt") -> logging.Logger:
    """Configure the root logger with console and append-mode file handlers."""
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    formatter = logging.Formatter(fmt="%(asctime)s: %(message)s", datefmt="%H:%M:%S")

    if not any(isinstance(h, RichHandler) for h in root.handlers):
        rich_handler = RichHandler(
            rich_tracebacks=True, show_time=False, show_path=False, markup=True
        )
        rich_handler.setFormatter(formatter)
        rich_handler.setLevel(logging.INFO)
        root.addHandler(rich_handler)

    resolved_path = os.path.abspath(log_file)
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    existing = [
        h
        for h in root.handlers
        if isinstance(h, logging.FileHandler)
        and getattr(h, "baseFilename", "") == resolved_path
    ]
    if not existing:
        fh = logging.FileHandler(filename=log_file, mode="a", encoding="utf-8")
        fh.setFormatter(formatter)
        fh.setLevel(logging.INFO)
        root.addHandler(fh)

    return root


@dataclass
class LogBundle:
    """Structured payload routed to subscribers, keyed by extension-less relative path."""

    figures: dict[str, Plot] = field(default_factory=dict)
    json: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "LogBundle":
        """Build a LogBundle from a flat dict keyed by 'figure/' or 'json/' paths."""
        figures: dict[str, Plot] = {}
        json_: dict[str, Any] = {}
        for key, value in d.items():
            if key.startswith("figure/"):
                figures[key[len("figure/") :]] = value
            elif key.startswith("json/"):
                json_[key[len("json/") :]] = value
            else:
                raise ValueError(
                    f"Artifact key {key!r} has no known prefix: "
                    "expected 'figure/' or 'json/'."
                )
        return cls(figures=figures, json=json_)


class LogDispatcher:
    """Routes LogBundle events to all registered subscribers."""

    def __init__(self) -> None:
        self._subscribers: list = []

    def subscribe(self, sub) -> None:
        """Register a subscriber: it names the bundle part it `consumes` and has `on_log`."""
        self._subscribers.append(sub)

    def publish(self, bundle: LogBundle) -> None:
        """Send bundle to all registered subscribers, refusing a part nobody writes."""
        for part in ("figures", "json"):
            names = getattr(bundle, part)
            if names and not any(sub.consumes == part for sub in self._subscribers):
                raise ValueError(
                    f"No subscriber writes {part} on this bus: {sorted(names)} "
                    "would be lost."
                )
        for sub in self._subscribers:
            sub.on_log(bundle)


class FilesystemFigureSubscriber:
    """Writes figures from LogBundle as image files under base_path / {name}.{format}."""

    consumes = "figures"

    def __init__(self, base_path: Path) -> None:
        self._base_path = Path(base_path)

    def on_log(self, bundle: LogBundle) -> None:
        """Write every figure in the bundle."""
        for name, plot in bundle.figures.items():
            out = self._base_path / f"{name}.{plot.format}"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(plot.data)


class JSONSubscriber:
    """Saves json artifacts from LogBundle under base_path / f"{name}.json"."""

    consumes = "json"

    def __init__(self, base_path: Path) -> None:
        self._base_path = Path(base_path)

    def on_log(self, bundle: LogBundle) -> None:
        """Write every JSON artifact in the bundle."""
        for name, value in bundle.json.items():
            save_to_json(value, self._base_path / f"{name}.json")
