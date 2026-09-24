from src.core.factory import Factory
from src.engine.dl.loss.base import BaseLoss

LossFactory = Factory[BaseLoss](component_type_name="loss")
