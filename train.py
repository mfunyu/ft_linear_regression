import sys
import json
import pandas as pd

import visualise
from linear_regression import LinearRegression

FILENAME = "trained_data.json"


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

    return mileage, price


def print_result(model):
    data = model.get_result()

    print("\n[Calculated data]")
    for key, value in data.items():
        print(f"{key}: {value}")


def save_data(model):
    data = model.get_result()
    data["data_file"] = FILENAME

    try:
        with open(FILENAME, "w") as f:
            json.dump(data, f)
    except Exception as e:
        print("Error:", e)


def main():
    filename = get_filename()
    mileage, price = load_dataset(filename)

    model = LinearRegression(learning_rate=0.01, iteration=1000)
    model.fit_gradient_descent(mileage, price)

    print_result(model)
    save_data(model)

    visualise.plot_graph(mileage, price, model)


if __name__ == "__main__":
    main()
