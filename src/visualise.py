import numpy as np
import matplotlib.pyplot as plt


def plot_graph(mileage, price, model):
    plt.scatter(mileage, price)

    x = np.arange(0, np.max(mileage) + 1)
    y = model.predict(x)
    y = np.round(y)

    plt.plot(x, y)
    plt.xlabel('Mileage')
    plt.ylabel('Price')
    plt.xlim(xmin=0)

    plt.show()
