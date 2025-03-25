import numpy as np


class LinearRegression:
    def __init__(self, theta0=0, theta1=0, learning_rate=0.001, iteration=1000):
        self.__theta0 = theta0
        self.__theta1 = theta1
        self.learning_rate = learning_rate
        self.iteration = iteration
        self.mean = 0
        self.std = 0
        self.normalized = False

    def normalize_data(self, x):
        if not self.normalized:
            self.mean = np.mean(x)
            self.std = np.std(x)
        return (x - self.mean) / self.std

    def fit_gradient_descent(self, x, y):
        x = self.normalize_data(x)
        y = self.normalize_data(y)
        m = len(x)

        for _ in range(self.iteration):
            prediction = self.predict(x, normalize=False)
            error = prediction - y

            tmp_theta0 = self.learning_rate * (np.sum(error) / m)
            tmp_theta1 = self.learning_rate * (np.sum(error * x) / m)
            self.__theta0 -= tmp_theta0
            self.__theta1 -= tmp_theta1

    def predict(self, x, normalize=True):
        return self.__theta0 + self.__theta1 * x

    def get_theta0(self):
        return self.__theta0

    def get_theta1(self):
        return self.__theta1
