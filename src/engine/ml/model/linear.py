from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

from src.engine.ml.model.factory import MLClassifierFactory

MLClassifierFactory.register("logistic_regression")(LogisticRegression)
MLClassifierFactory.register("lda")(LinearDiscriminantAnalysis)
