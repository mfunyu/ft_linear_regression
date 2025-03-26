from sklearn.linear_model import LinearRegression
import sys
import numpy as np

import utils
from linear_regression import LinearRegression as myLinearRegression
import visualise


def get_filename_from_arg() -> str:
    if len(sys.argv) != 2:
        return None

    return sys.argv[1]


def main():
    filename, learning_rate, iteration = utils.get_arg_options()
    mileage, price = utils.load_dataset_from_csv(filename)

    try:
        print("Learning rate:", learning_rate)
        print("Iteration    :", iteration)
        my_model = myLinearRegression(learning_rate=learning_rate, iteration=iteration)
        my_model.fit_gradient_descent(mileage, price)
        print("Cost: ", my_model.compute_cost_mse(mileage, price))

        model = LinearRegression()
        model.fit(mileage.reshape(-1, 1), price)
        print("Cost: ", np.mean((model.predict(mileage.reshape(-1, 1)) - price) ** 2))

        visualise.plot_two_graphs(mileage, price, my_model, model)
    except Exception as e:
        print("Error:", e)


if __name__ == "__main__":
    main()
