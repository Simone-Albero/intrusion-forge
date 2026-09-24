from sklearn.neighbors import KNeighborsClassifier

from src.engine.ml.model.factory import MLClassifierFactory

MLClassifierFactory.register("knn")(KNeighborsClassifier)
