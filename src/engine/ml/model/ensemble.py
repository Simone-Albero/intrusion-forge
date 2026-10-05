from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
)

from src.engine.ml.model.factory import MLClassifierFactory, MLRegressorFactory

MLClassifierFactory.register("random_forest")(RandomForestClassifier)
MLClassifierFactory.register("hist_gradient_boosting")(HistGradientBoostingClassifier)
MLRegressorFactory.register("random_forest")(RandomForestRegressor)
