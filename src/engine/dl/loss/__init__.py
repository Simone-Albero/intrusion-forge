from pathlib import Path

from src.core.factory import Factory, discover_and_import_modules

from .base import BaseLoss

LossFactory = Factory[BaseLoss](component_type_name="loss")

_package_path = Path(__file__).parent
discover_and_import_modules(package_path=_package_path, package_name=__name__)

__all__ = ["LossFactory"]
