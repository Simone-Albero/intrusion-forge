from pathlib import Path
import random
from typing import Optional, Tuple, Union

import numpy as np
import torch

PathLike = Union[str, Path]


class PathUtils:
    """General utilities for path and format management."""

    @staticmethod
    def resolve_path_and_format(
        base_path: PathLike,
        name: Optional[str] = None,
        fmt: Optional[str] = None,
    ) -> Tuple[Path, str]:
        """
        Resolve final file path and format from inputs.
        """
        base = Path(base_path)

        # File path with extension
        if base.suffix:
            resolved_fmt = base.suffix.lstrip(".").lower()
            if name:
                final_path = base.with_name(Path(name).stem + base.suffix)
            else:
                final_path = base
            return final_path, resolved_fmt

        # Directory path
        if not name:
            raise ValueError("Name required when base_path is a directory")

        name_path = Path(name)

        if name_path.suffix:
            resolved_fmt = name_path.suffix.lstrip(".").lower()
            final_path = base / name_path.name
        elif fmt:
            resolved_fmt = fmt
            final_path = base / f"{name_path.name}.{fmt}"
        else:
            raise ValueError("Cannot infer format: provide name with suffix or fmt")

        return final_path, resolved_fmt

    @staticmethod
    def get_all_files_with_suffix(directory: PathLike, suffix: str) -> list:
        """
        List all files in a directory with a given suffix.
        """
        return list(Path(directory).glob(f"*{suffix}"))

    @staticmethod
    def set_format(path: PathLike, fmt: str) -> Path:
        """
        Return a new Path with the extension replaced by the given format.
        """
        p = Path(path)
        return p.with_suffix(f".{fmt}")

    @staticmethod
    def split_path(path: PathLike) -> Tuple[str, str, str]:
        """
        Split a path into directory, filename (without extension), and extension.
        """
        p = Path(path)
        return str(p.parent), p.stem, p.suffix.lstrip(".")

    @staticmethod
    def ensure_dir_exists(path: PathLike) -> None:
        """
        Ensure the directory for the given path exists.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
