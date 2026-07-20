import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC, LinearSVC

from . import MLClassifierFactory


class CalibratedLinearSVC(BaseEstimator, ClassifierMixin):
    """LinearSVC wrapped in CalibratedClassifierCV for predict_proba; sigmoid cv=3 stays O(n), avoiding SVC(probability=True)'s Platt-scaling overhead."""

    def __init__(self, C: float = 1.0, max_iter: int = 2000, random_state: int | None = None):
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X, y):
        base = LinearSVC(C=self.C, max_iter=self.max_iter, random_state=self.random_state)
        self._calibrated = CalibratedClassifierCV(base, method="sigmoid", cv=3)
        self._calibrated.fit(X, y)
        self.classes_ = self._calibrated.classes_
        return self

    def predict(self, X) -> np.ndarray:
        return self._calibrated.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        return self._calibrated.predict_proba(X)


MLClassifierFactory.register("svm_rbf")(SVC)
MLClassifierFactory.register("linear_svc")(CalibratedLinearSVC)
