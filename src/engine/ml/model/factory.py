from sklearn.base import BaseEstimator

from src.core.factory import Factory

MLClassifierFactory = Factory[BaseEstimator](component_type_name="ml_classifier")
