from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

from src.engine.ml.model.factory import MLClassifierFactory

MLClassifierFactory.register("random_forest")(RandomForestClassifier)
MLClassifierFactory.register("hist_gradient_boosting")(HistGradientBoostingClassifier)
