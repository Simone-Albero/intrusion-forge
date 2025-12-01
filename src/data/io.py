from typing import Literal, Optional
from pathlib import Path
import pickle

import pandas as pd


def load_df(file_path: str, **kwargs) -> pd.DataFrame:
    ext = Path(file_path).suffix.lower()
    if ext == ".parquet":
        return pd.read_parquet(file_path, **kwargs)
    elif ext == ".csv":
        return pd.read_csv(file_path, **kwargs)
    elif ext == ".pkl" or ext == ".pickle":
        with open(file_path, "rb") as f:
            return pickle.load(f)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")


def save_df(
    df: pd.DataFrame,
    file_path: str,
    index: bool = False,
    **kwargs,
) -> None:
    ext = Path(file_path).suffix.lower()
    if ext == ".parquet":
        df.to_parquet(file_path, index=index, **kwargs)
    elif ext == ".csv":
        df.to_csv(file_path, index=index, **kwargs)
    elif ext == ".pkl" or ext == ".pickle":
        with open(file_path, "wb") as f:
            pickle.dump(df, f)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")
