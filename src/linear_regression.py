import numpy as np


class LinearRegression:
    def __init__(self, theta0=0, theta1=0, mean=0, std=1,
                 learning_rate=0.01, iteration=1000):
        self.__theta0 = theta0
        self.__theta1 = theta1
        self.__learning_rate = learning_rate
        self.__iteration = iteration
        self.__mean = mean
        self.__std = std

    def normalize_data(self, x):
        return (x - self.__mean) / self.__std

    def fit_gradient_descent(self, x, y):
        self.__theta0 = 0
        self.__theta1 = 0
        self.__mean = np.mean(x)
        self.__std = np.std(x)
        if self.__std == 0:
            self.__std = self.__mean
        x = self.normalize_data(x)

        for _ in range(self.__iteration):
            prediction = self.predict(x, normalize=False)
            error = prediction - y

            tmp_theta0 = self.__learning_rate * np.mean(error)
            tmp_theta1 = self.__learning_rate * np.mean(error * x)
            self.__theta0 -= tmp_theta0
            self.__theta1 -= tmp_theta1

    def predict(self, x, normalize=True):
        if normalize:
            x = self.normalize_data(x)
        return self.__theta0 + self.__theta1 * x

    def get_result(self):
        return {
            "theta0": self.__theta0,
            "theta1": self.__theta1,
            "mean": self.__mean,
            "std": self.__std
        }

    def compute_cost_mse(self, x, y):
        prediction = self.predict(x)
        error = prediction - y

        return np.mean(error ** 2)

    def compute_cost_mae(self, x, y):
        prediction = self.predict(x)
        error = prediction - y

        return np.mean(np.abs(error))

    def set_learning_rate(self, learning_rate):
        self.__learning_rate = learning_rate

    def set_iteration(self, iteration):
        self.__iteration = iteration
