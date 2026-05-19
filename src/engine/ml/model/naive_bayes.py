from sklearn.naive_bayes import GaussianNB

from . import MLClassifierFactory

MLClassifierFactory.register("naive_bayes")(GaussianNB)
