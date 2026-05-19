"""Central re-export of all component registries used across the project.

Import the relevant factory from here instead of digging into the package
that owns it; the package owners remain free to move without breaking
callers.
"""

from src.engine.dl.loss import LossFactory
from src.engine.dl.model import DLClassifierFactory
from src.engine.ml.model import MLClassifierFactory

__all__ = ["DLClassifierFactory", "LossFactory", "MLClassifierFactory"]
