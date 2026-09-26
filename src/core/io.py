from pathlib import Path

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
