from sklearn.naive_bayes import GaussianNB

from src.engine.ml.model.factory import MLClassifierFactory

MLClassifierFactory.register("naive_bayes")(GaussianNB)
