from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, Ridge

from src.engine.ml.model.factory import MLClassifierFactory, MLRegressorFactory

MLClassifierFactory.register("logistic_regression")(LogisticRegression)
MLClassifierFactory.register("lda")(LinearDiscriminantAnalysis)
MLRegressorFactory.register("ridge")(Ridge)
