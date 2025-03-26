import numpy as np
import matplotlib.pyplot as plt


def plot_two_graphs(milage, price, model1, model2):
    plt.scatter(milage, price)

    x = np.arange(0, np.max(milage) + 1)
    y1 = model1.predict(x)
    y1 = np.round(y1)

    y2 = model2.predict(x.reshape(-1, 1))
    y2 = np.round(y2)

    plt.plot(x, y1, label="Model 1")
    plt.plot(x, y2, label="Model 2")
    plt.xlabel("Mileage")
    plt.ylabel("Price")
    plt.xlim(xmin=0)
    plt.legend()

    plt.show()


def plot_graph(mileage, price, model):
    plt.scatter(mileage, price)

    x = np.arange(0, np.max(mileage) + 1)
    y = model.predict(x)
    y = np.round(y)

    plt.plot(x, y)
    plt.xlabel("Mileage")
    plt.ylabel("Price")
    plt.xlim(xmin=0)

    plt.show()


def plot_cost(x, y, x_label):
    plt.plot(x, y)

    plt.xlabel(x_label)
    plt.ylabel("Mean squared error")
    plt.show()
