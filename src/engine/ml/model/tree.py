from sklearn.tree import DecisionTreeClassifier

from . import MLClassifierFactory

MLClassifierFactory.register("decision_tree")(DecisionTreeClassifier)
