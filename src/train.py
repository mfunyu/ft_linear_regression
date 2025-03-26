import json
import numpy as np

import utils
import visualise
from linear_regression import LinearRegression

TARGET_FILE = "trained_data.json"


def print_result(model):
    data = model.get_result()

    print("\n[Calculated data]")
    for key, value in data.items():
        print(f"{key}: {value}")


def save_data(model, filename):
    data = model.get_result()
    data["data_file"] = filename

    try:
        with open(TARGET_FILE, "w") as f:
            json.dump(data, f)
    except Exception as e:
        print("Error:", e)


def cost_test(model, mileage, price, x_data, setter, x_label):
    mean_squared_errors = []

    for i in x_data:
        setter(i)
        model.fit_gradient_descent(mileage, price)
        mse = model.compute_cost(mileage, price)
        mse = round(mse)
        mean_squared_errors.append(mse)

    visualise.plot_cost(x_data, mean_squared_errors, x_label)


def run_analysis(model, mileage, price):

    visualise.plot_graph(mileage, price, model)
    mse = model.compute_cost(mileage, price)

    test = input("run cost test? (y/n): ")
    if test == "y":
        print("Mean squared error:", mse)
        learning_rates = np.arange(0.0005, 0.01, 0.0001)
        cost_test(model, mileage, price, learning_rates, model.set_learning_rate, "Learning rate")
        iterations = np.arange(100, 2000, 10)
        cost_test(model, mileage, price, iterations, model.set_iteration, "Iteration")
    else:
        print("Goodbye")


def main():
    filename = utils.get_filename_from_arg()
    if not filename:
        print("[Usage] python train.py <filename>")
        exit()
    mileage, price = utils.load_dataset_from_csv(filename)

    model = LinearRegression(learning_rate=1, iteration=1000)
    model.fit_gradient_descent(mileage, price)

    print_result(model)
    save_data(model, filename)

    run_analysis(model, mileage, price)


if __name__ == "__main__":
    main()
