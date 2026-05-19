from sklearn.neighbors import KNeighborsClassifier

from . import MLClassifierFactory

MLClassifierFactory.register("knn")(KNeighborsClassifier)
