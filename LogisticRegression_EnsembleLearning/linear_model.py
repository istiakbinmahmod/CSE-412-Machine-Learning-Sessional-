import numpy as np


class LogisticRegression:
    def __init__(self, params):
        """
        figure out necessary params to take as input
        :param params:
        """
        # todo: implement
        self.learning_rate = params['learning_rate']
        self.n_iters = params['n_iters']

    def fit(self, X, y):
        """
        :param X:
        :param y:
        :return: self
        """
        assert X.shape[0] == y.shape[0]
        assert len(X.shape) == 2
        # todo: implement
        self.m, self.n = X.shape
        self.W = np.zeros(self.n)
        self.b = 0
        self.X = X
        self.y = y

        # gradient descent
        for i in range(self.n_iters):
            self.update_weights()

        return self

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def update_weights(self):
        """
        update weights using gradient descent
        :return:
        """
        Y_hat = self.sigmoid(np.dot(self.X, self.W) + self.b)
        dW = (1 / self.m) * np.dot(self.X.T, (Y_hat - self.y))
        db = (1 / self.m) * np.sum(Y_hat - self.y)
        self.W -= self.learning_rate * dW
        self.b -= self.learning_rate * db

    def predict(self, X):
        """
        function for predicting labels of for all datapoint in X
        :param X:
        :return:
        """
        # todo: implement
        y_pred = self.sigmoid(z=np.dot(X, self.W) + self.b)
        y_pred_cls = [1 if i > 0.5 else 0 for i in y_pred]
        return np.array(y_pred_cls)
