from sklearn.svm import SVC

from . import MLClassifierFactory

MLClassifierFactory.register("svm_rbf")(SVC)
