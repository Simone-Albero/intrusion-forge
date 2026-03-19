import json
import pickle
from pathlib import Path


def save_to_json(data: object, file_path: str | Path) -> None:
    """Save data to a JSON file."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)


def load_from_json(file_path: str | Path) -> object:
    """Load data from a JSON file."""
    file_path = Path(file_path)
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def save_to_pickle(data: object, file_path: str | Path) -> None:
    """Save data to a pickle file."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "wb") as f:
        pickle.dump(data, f)


def load_from_pickle(file_path: str | Path) -> object:
    """Load data from a pickle file."""
    file_path = Path(file_path)
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data
