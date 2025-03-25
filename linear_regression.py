import numpy as np


class LinearRegression:
    def __init__(self, theta0=0, theta1=0, mean=0, std=1,
                 learning_rate=0.001, iteration=1000):
        self.__theta0 = theta0
        self.__theta1 = theta1
        self.learning_rate = learning_rate
        self.iteration = iteration
        self.mean = mean
        self.std = std

    def normalize_data(self, x):
        return (x - self.mean) / self.std

    def fit_gradient_descent(self, x, y):
        self.mean = np.mean(x)
        self.std = np.std(x) if np.std(x) != 0 else self.mean
        x = self.normalize_data(x)
        m = len(x)

        for _ in range(self.iteration):
            prediction = self.predict(x, normalize=False)
            error = prediction - y

            tmp_theta0 = self.learning_rate * (np.sum(error) / m)
            tmp_theta1 = self.learning_rate * (np.sum(error * x) / m)
            self.__theta0 -= tmp_theta0
            self.__theta1 -= tmp_theta1

    def predict(self, x, normalize=True):
        if normalize:
            x = self.normalize_data(x)
            print(x)
        return self.__theta0 + self.__theta1 * x

    def get_result(self):
        return {
            "theta0": self.__theta0,
            "theta1": self.__theta1,
            "mean": self.mean,
            "std": self.std
        }

    def get_theta0(self):
        return self.__theta0

    def get_theta1(self):
        return self.__theta1
