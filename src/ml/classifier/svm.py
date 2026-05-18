from sklearn.svm import SVC

from . import MLClassifierFactory


class SVCProbabilityTrue(SVC):
    """SVC defaulting `probability=True` so `predict_proba` is available.

    Accepts an explicit `probability=False` override without clashing.
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("probability", True)
        super().__init__(*args, **kwargs)


MLClassifierFactory.register("svm_rbf")(SVCProbabilityTrue)
