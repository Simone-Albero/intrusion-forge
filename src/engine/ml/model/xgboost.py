from xgboost import XGBClassifier

from src.engine.ml.model.factory import MLClassifierFactory

MLClassifierFactory.register("xgboost")(XGBClassifier)
