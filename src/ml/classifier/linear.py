from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

from . import MLClassifierFactory

MLClassifierFactory.register("logistic_regression")(LogisticRegression)
MLClassifierFactory.register("lda")(LinearDiscriminantAnalysis)
