from sklearn.neural_network import MLPRegressor

from src.engine.ml.model.factory import MLRegressorFactory

MLRegressorFactory.register("mlp")(MLPRegressor)
