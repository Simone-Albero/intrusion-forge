from pathlib import Path

import numpy as np
import pandas as pd

_LOADERS = {
    ".parquet": pd.read_parquet,
    ".csv": pd.read_csv,
}

_SAVERS = {
    ".parquet": lambda df, p, **kw: df.to_parquet(p, **kw),
    ".csv": lambda df, p, **kw: df.to_csv(p, **kw),
}

_ALL_EXTS = sorted(_LOADERS)


def load_df(file_path: str | Path, **kwargs) -> pd.DataFrame:
    """Load a DataFrame from a file based on its extension."""
    file_path = Path(file_path)
    ext = file_path.suffix.lower()
    loader = _LOADERS.get(ext)
    if loader is None:
        raise ValueError(f"Unsupported file extension: {ext!r}. Supported: {_ALL_EXTS}")
    return loader(file_path, **kwargs)


def save_df(
    df: pd.DataFrame,
    file_path: str | Path,
    *,
    index: bool = False,
    **kwargs,
) -> None:
    """Save a DataFrame to a file based on its extension."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    ext = file_path.suffix.lower()
    saver = _SAVERS.get(ext)
    if saver is None:
        raise ValueError(f"Unsupported file extension: {ext!r}. Supported: {_ALL_EXTS}")
    saver(df, file_path, index=index, **kwargs)


def save_figures(figures: dict, folder: str | Path) -> None:
    """Write each figure to `folder/<name>.<format>`; a name may carry subfolders."""
    for name, plot in figures.items():
        out = Path(folder) / f"{name}.{plot.format}"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(plot.data)


def save_arrays(arrays: dict[str, np.ndarray], path: str | Path) -> None:
    """Write named arrays to one compressed .npz file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_arrays(path: str | Path) -> dict[str, np.ndarray]:
    """Read every array of an .npz file."""
    with np.load(path) as stored:
        return {name: stored[name] for name in stored.files}
