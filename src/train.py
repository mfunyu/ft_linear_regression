import json
import sys

import utils
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


def main():
    filename, learning_rate, iteration = utils.get_arg_options()
    mileage, price = utils.load_dataset_from_csv(filename)

    try:
        print("Learning rate:", learning_rate)
        print("Iteration    :", iteration)
        model = LinearRegression(learning_rate=learning_rate, iteration=iteration)
        model.fit_gradient_descent(mileage, price)

        print_result(model)
        save_data(model, filename)
    except Exception as e:
        print("Error:", e)


if __name__ == "__main__":
    main()
