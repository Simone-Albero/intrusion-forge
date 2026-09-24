from src.core.factory import Factory
from src.engine.dl.model.base import BaseModel

DLClassifierFactory = Factory[BaseModel](component_type_name="dl_classifier")
