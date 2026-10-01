import logging
import os
from pathlib import Path

from rich.logging import RichHandler


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
