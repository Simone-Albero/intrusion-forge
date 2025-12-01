import logging
import importlib
import pkgutil
from pathlib import Path
from typing import Type, TypeVar, Optional, Callable, Dict, Generic, Any, List

logger = logging.getLogger(__name__)
T = TypeVar("T")


class Factory(Generic[T]):
    """Generic factory for creating instances from configuration."""

    def __init__(self, component_type_name: str = "component"):
        """
        Initialize factory.

        Args:
            component_type_name: Name of the component type for logging/errors
        """
        self._registry: Dict[str, Type[T]] = {}
        self._component_type_name = component_type_name

    def register(self, name: Optional[str] = None) -> Callable:
        """
        Decorator to register a class.

        Args:
            name: Optional name to register the class under.
                  If not provided, uses snake_case of class name.

        Usage:
            @factory.register()
            class MyComponent:
                ...
        """

        def decorator(cls: Type[T]) -> Type[T]:
            component_name = name or self._class_to_snake_case(cls.__name__)
            self._registry[component_name] = cls
            return cls

        return decorator

    @staticmethod
    def _class_to_snake_case(class_name: str) -> str:
        """Convert class name to snake_case."""
        import re

        s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", class_name)
        return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()

    def create(self, class_name: str, params: Optional[Dict[str, Any]] = None) -> T:
        """
        Create an instance of a registered class.

        Returns:
            Instantiated component
        """

        component_class = self._registry.get(class_name)

        if not component_class:
            raise ValueError(
                f"Unknown {self._component_type_name} type: {class_name}. "
                f"Available types: {list(self._registry.keys())}"
            )

        return component_class(**params) if params else component_class()

    def create_from_list(self, names: list, params_list: list) -> list[T]:
        """
        Create multiple instances from a list of configurations.

        Returns:
            List of instantiated components
        """

        return [self.create(name, params) for name, params in zip(names, params_list)]

    def get_available(self) -> list[str]:
        """Get list of all registered types."""
        return sorted(self._registry.keys())

    def get_registry(self) -> Dict[str, Type[T]]:
        """Get the full registry."""
        return self._registry.copy()


def discover_and_import_modules(package_path: Path, package_name: str) -> List[str]:
    """
    Recursively discover and import all Python modules in a package.

    This function walks through all subdirectories and imports every .py file
    it finds, triggering any @register_step decorators in those files.

    Args:
        package_path: Path to the package directory
        package_name: Fully qualified package name (e.g., 'src.idspy.builtins.step.data')

    Returns:
        List of imported module names
    """
    imported_modules = []

    try:
        # Walk through all modules in the package
        for _, modname, _ in pkgutil.walk_packages(
            path=[str(package_path)],
            prefix=f"{package_name}.",
        ):
            # Skip __pycache__ and other special directories
            if modname.endswith(".__pycache__"):
                continue

            try:
                # Import the module to trigger decorator registration
                importlib.import_module(modname)
                imported_modules.append(modname)
            except Exception as e:
                logger.warning(f"Failed to import {modname}: {e}")

    except Exception as e:
        logger.error(f"Error during module discovery in {package_name}: {e}")

    return imported_modules
