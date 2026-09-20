from src.core.factory import Factory
from src.domain.clustering.base import FitFn

ClusteringFactory = Factory[FitFn](component_type_name="clustering_algorithm")
