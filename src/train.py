import sys
import json
import pandas as pd
import numpy as np

import visualise
from linear_regression import LinearRegression

TARGET_FILE = "trained_data.json"


def get_filename() -> str:
    if len(sys.argv) != 2:
        print("[Usage] python3 predict.py [filename]")
        exit()

    return sys.argv[1]


def load_dataset(filename):
    try:
        with open(filename, "r") as csvfile:
            print("Loading data from", filename)
            data_frame = pd.read_csv(csvfile)
            mileage = data_frame["km"].values
            price = data_frame["price"].values

    except Exception as e:
        print("Error:", e)
        exit()

    if len(mileage) == 0 or len(price) == 0:
        print("Error: No data found in", filename)
        exit()
    if len(mileage) != len(price):
        print("Error: Mismatched data between mileage and price")
        exit()

    return mileage, price


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
    filename = get_filename()
    mileage, price = load_dataset(filename)

    model = LinearRegression(learning_rate=0.01, iteration=1000)
    model.fit_gradient_descent(mileage, price)

    print_result(model)
    save_data(model, filename)

    run_analysis(model, mileage, price)


if __name__ == "__main__":
    main()
