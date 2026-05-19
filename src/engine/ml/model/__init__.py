from pathlib import Path

from sklearn.base import BaseEstimator

from src.core.factory import Factory, discover_and_import_modules

MLClassifierFactory = Factory[BaseEstimator](component_type_name="ml_classifier")

_package_path = Path(__file__).parent
discover_and_import_modules(package_path=_package_path, package_name=__name__)

__all__ = ["MLClassifierFactory"]
