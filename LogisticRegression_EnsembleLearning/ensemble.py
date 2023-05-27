from data_handler import bagging_sampler
import numpy as np
import pandas as pd


class BaggingClassifier:
    def __init__(self, base_estimator, n_estimator):
        """
        :param base_estimator:
        :param n_estimator:
        :return:
        """
        # todo: implement
        self.base_estimator = base_estimator
        self.n_estimator = n_estimator
        self.estimators = []

    def fit(self, X, y):
        """
        :param X:
        :param y:
        :return: self
        """
        # todo: implement
        assert X.shape[0] == y.shape[0]
        assert len(X.shape) == 2

        # self.m, self.n = X.shape
        # self.W = np.zeros(self.n)
        # self.b = 0
        self.X = X
        self.y = y

        for i in range(self.n_estimator):
            X_sample, y_sample = bagging_sampler(X, y)
            self.estimators.append(self.base_estimator.fit(X_sample, y_sample))

    def predict(self, X):
        """
        function for predicting labels of for all datapoint in X
        apply majority voting
        :param X:
        :return:
        """
        # todo: implement
        y_pred = np.zeros(X.shape[0])
        for estimator in self.estimators:
            y_pred += estimator.predict(X)
        y_pred = np.round(y_pred / self.n_estimator)
        y_pred_cls = np.where(y_pred > 0.5, 1, 0)

        return np.array(y_pred_cls)
