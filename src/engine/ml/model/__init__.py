from pathlib import Path

from src.core.factory import discover_and_import_modules
from src.engine.ml.model.factory import MLClassifierFactory

discover_and_import_modules(package_path=Path(__file__).parent, package_name=__name__)

__all__ = ["MLClassifierFactory"]
