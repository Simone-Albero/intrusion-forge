from pathlib import Path
import pickle

import pandas as pd


def load_df(file_path: str, **kwargs) -> pd.DataFrame:
    """Load a DataFrame from a file based on its extension."""
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
    """Save a DataFrame to a file based on its extension."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    ext = file_path.suffix.lower()
    if ext == ".parquet":
        df.to_parquet(file_path, index=index, **kwargs)
    elif ext == ".csv":
        df.to_csv(file_path, index=index, **kwargs)
    elif ext == ".pkl" or ext == ".pickle":
        with open(file_path, "wb") as f:
            pickle.dump(df, f)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")


def load_listed_dfs(base_dir: Path, file_names: list[str]) -> list[pd.DataFrame]:
    """Load a list of DataFrames from a base directory.

    Args:
        base_dir: Path to the directory containing data files
        file_names: List of file names (without extension)

    Returns:
        List of DataFrames
    """
    dfs = []
    for file_name in file_names:
        try:
            df = load_df(base_dir / f"{file_name}.parquet")
            dfs.append(df)
        except FileNotFoundError as e:
            raise ValueError(
                f"Data file not found: {base_dir / f'{file_name}.parquet'}"
            )
        except Exception as e:
            raise ValueError(f"Error loading data file {file_name}: {e}")
    return dfs
