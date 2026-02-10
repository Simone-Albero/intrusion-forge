from pathlib import Path

from ....common.factory import Factory, discover_and_import_modules
from torch.utils.data import Sampler

SamplerFactory = Factory[Sampler](component_type_name="sampler")

# Get the current package path
_package_path = Path(__file__).parent

# Auto-discover and import all step modules
discover_and_import_modules(package_path=_package_path, package_name=__name__)

# Public API
__all__ = [
    "SamplerFactory",
]
