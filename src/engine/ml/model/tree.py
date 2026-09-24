from sklearn.tree import DecisionTreeClassifier

from src.engine.ml.model.factory import MLClassifierFactory

MLClassifierFactory.register("decision_tree")(DecisionTreeClassifier)
