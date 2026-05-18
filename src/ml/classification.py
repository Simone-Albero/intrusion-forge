import numpy as np
from src.common.factory import Factory
from sklearn.metrics import *
from sklearn.base import BaseEstimator
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from xgboost import XGBClassifier

class SVCProbabilityTrue(SVC):
    """SVC with probability=True by default."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, probability=True, **kwargs)

models = {
    "naive_bayes": GaussianNB,
    "logistic_regression": LogisticRegression,
    "svm_rbf": SVCProbabilityTrue,
    "knn": KNeighborsClassifier,
    "decision_tree": DecisionTreeClassifier,
    "random_forest": RandomForestClassifier,
    "hist_gradient_boosting": HistGradientBoostingClassifier,
    "lda": LinearDiscriminantAnalysis,
    "xgboost": XGBClassifier
}

SklearnClassifierFactory = Factory[BaseEstimator](component_type_name="ml_classifier") # module-level singleton

for model in models:
    SklearnClassifierFactory.register(model)(models[model])


def train_sklearn_classifier(
    X: np.ndarray,
    y: np.ndarray,
    classifier_name: str,
    classifier_params: dict,
) -> BaseEstimator:
    """Fit and return a sklearn/XGBoost classifier."""

    model = SklearnClassifierFactory.create(classifier_name, classifier_params)
    model.fit(X, y)
    return model


def evaluate_sklearn_classifier(
    clf: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
) -> dict:
    """Compute classification metrics for a fitted classifier.

    Output keys: 'accuracy', 'precision_macro', 'recall_macro', 'f1_macro',
    'precision_weighted', 'recall_weighted', 'f1_weighted',
    'precision_per_class', 'recall_per_class', 'f1_per_class',
    'confusion_matrix' (np.ndarray, shape [n_classes, n_classes]),
    'confidences' (np.ndarray, shape [n_samples])
    """

    y_pred = clf.predict(X)
    metrics = {"accuracy":accuracy_score(y, y_pred), 
               "precision_macro":precision_score(y, y_pred, average="macro"),
               "recall_macro":recall_score(y, y_pred, average="macro"),
               "f1_macro":f1_score(y, y_pred, average="macro"),
               "precision_weighted":precision_score(y, y_pred, average="weighted"),
               "recall_weighted":recall_score(y, y_pred, average="weighted"),
               "f1_weighted":f1_score(y, y_pred, average="weighted"),
               "precision_per_class":precision_score(y, y_pred, average=None),
               "recall_per_class":recall_score(y, y_pred, average=None),
               "f1_per_class":f1_score(y, y_pred, average=None),
               "confusion_matrix":confusion_matrix(y, y_pred),
               "confidences": clf.predict_proba(X)}
    return metrics
    

