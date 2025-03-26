import numpy as np

import visualise
import utils
from linear_regression import LinearRegression


def cost_change_test(model, mileage, price, x_data, setter, x_label):
    mean_squared_errors = []

    for i in x_data:
        setter(i)
        model.fit_gradient_descent(mileage, price)
        mse = model.compute_cost_mse(mileage, price)
        mse = round(mse)
        mean_squared_errors.append(mse)

    visualise.plot_cost(x_data, mean_squared_errors, x_label)


def learning_rate_cost_change_test(model, mileage, price, iteration):
    print("Running learning rate change test...")
    data = np.arange(0.0005, 0.02, 0.0001)
    label = "Learning rate"
    model.set_iteration(iteration)
    cost_change_test(model, mileage, price, data, model.set_learning_rate, label)


def iteration_cost_change_test(model, mileage, price, learning_rate):
    print("Running iteration change test...")
    data = np.arange(100, 10000, 100)
    label = "Iteration"
    model.set_learning_rate(learning_rate)
    cost_change_test(model, mileage, price, data, model.set_iteration, label)


def run_analysis(model, mileage, price):
    model.fit_gradient_descent(mileage, price)

    mse = model.compute_cost_mse(mileage, price)
    print("Mean squared error:", mse)
    mae = model.compute_cost_mae(mileage, price)
    print("Mean absolute error:", mae)

    visualise.plot_graph(mileage, price, model)


def main():
    filename, learning_rate, iteration = utils.get_arg_options()
    mileage, price = utils.load_dataset_from_csv(filename)

    try:
        print("Learning rate:", learning_rate)
        print("Iteration    :", iteration)
        model = LinearRegression(learning_rate=learning_rate, iteration=iteration)

        run_analysis(model, mileage, price)
        learning_rate_cost_change_test(model, mileage, price, iteration)
        iteration_cost_change_test(model, mileage, price, learning_rate)
    except Exception as e:
        print("Error:", e)


if __name__ == "__main__":
    main()
