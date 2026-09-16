import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.logging import RichHandler

from src.domain.plot.base import Plot

from .utils import save_to_json, save_to_pickle


def setup_logger(
    *,
    level: int = logging.INFO,
    fmt: str = "%(asctime)s: %(message)s",
    date_fmt: str = "%H:%M:%S",
    console: bool = True,
    log_file: str | None = None,
    file_level: int | None = None,
) -> logging.Logger:
    """Configure the root logger with optional console and append-mode file handlers."""
    root = logging.getLogger()
    root.setLevel(level)

    formatter = logging.Formatter(fmt=fmt, datefmt=date_fmt)

    if console and not any(isinstance(h, RichHandler) for h in root.handlers):
        rich_handler = RichHandler(
            rich_tracebacks=True, show_time=False, show_path=False, markup=True
        )
        rich_handler.setFormatter(formatter)
        rich_handler.setLevel(level)
        root.addHandler(rich_handler)

    if log_file:
        file_level = file_level or level
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
            fh.setLevel(file_level)
            root.addHandler(fh)

    return root


@dataclass
class LogBundle:
    """Structured payload routed to subscribers, keyed by extension-less relative path."""

    figures: dict[str, Plot] = field(default_factory=dict)
    json: dict[str, Any] = field(default_factory=dict)
    pickle: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "LogBundle":
        """Build a LogBundle from a flat dict keyed by 'figure/', 'json/' or 'pickle/' paths."""
        figures: dict[str, Plot] = {}
        json_: dict[str, Any] = {}
        pickle_: dict[str, Any] = {}
        for key, value in d.items():
            if key.startswith("figure/"):
                figures[key[len("figure/") :]] = value
            elif key.startswith("json/"):
                json_[key[len("json/") :]] = value
            elif key.startswith("pickle/"):
                pickle_[key[len("pickle/") :]] = value
        return cls(figures=figures, json=json_, pickle=pickle_)


class LogDispatcher:
    """Routes LogBundle events to all registered subscribers."""

    def __init__(self) -> None:
        self._subscribers: list = []

    def subscribe(self, sub) -> None:
        """Register a subscriber. sub must implement on_log(bundle: LogBundle)."""
        self._subscribers.append(sub)

    def clear(self) -> None:
        """Remove all registered subscribers."""
        self._subscribers.clear()

    def publish(self, bundle: LogBundle) -> None:
        """Send bundle to all registered subscribers."""
        for sub in self._subscribers:
            sub.on_log(bundle)


class FilesystemFigureSubscriber:
    """Writes figures from LogBundle as image files under base_path / {name}.{format}."""

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

    def __init__(self, base_path: Path) -> None:
        self._base_path = Path(base_path)

    def on_log(self, bundle: LogBundle) -> None:
        """Write every JSON artifact in the bundle."""
        for name, value in bundle.json.items():
            save_to_json(value, self._base_path / f"{name}.json")


class PickleSubscriber:
    """Saves pickle artifacts from LogBundle under base_path / f"{name}.pkl"."""

    def __init__(self, base_path: Path) -> None:
        self._base_path = Path(base_path)

    def on_log(self, bundle: LogBundle) -> None:
        """Write every pickle artifact in the bundle."""
        for name, value in bundle.pickle.items():
            save_to_pickle(value, self._base_path / f"{name}.pkl")
