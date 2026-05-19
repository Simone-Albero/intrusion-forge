from xgboost import XGBClassifier

from . import MLClassifierFactory

MLClassifierFactory.register("xgboost")(XGBClassifier)
