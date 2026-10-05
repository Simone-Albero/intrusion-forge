from xgboost import XGBClassifier, XGBRegressor

from src.engine.ml.model.factory import MLClassifierFactory, MLRegressorFactory

MLClassifierFactory.register("xgboost")(XGBClassifier)
MLRegressorFactory.register("xgboost")(XGBRegressor)
