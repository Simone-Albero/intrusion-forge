from pathlib import Path

from src.core.factory import Factory, discover_and_import_modules
from .base import BaseModel

DLClassifierFactory = Factory[BaseModel](component_type_name="dl_classifier")

_package_path = Path(__file__).parent
discover_and_import_modules(package_path=_package_path, package_name=__name__)

__all__ = ["DLClassifierFactory"]
