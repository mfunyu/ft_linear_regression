import sys
import json
import pandas as pd
import matplotlib.pyplot as plt

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
    theta0 = model.get_theta0()
    theta1 = model.get_theta1()

    print("\n[Calculated data]")
    print("Theta0:", theta0)
    print("Theta1:", theta1)
    print(f"y = {theta0} + {theta1} * x")


def save_data(model):
    data = {
        "data_file": FILENAME,
        "theta0": model.get_theta0(),
        "theta1": model.get_theta1()
    }
    try:
        with open(FILENAME, "w") as f:
            json.dump(data, f)
    except Exception as e:
        print("Error:", e)


def plot_graph(mileage, price, model):
    y = model.predict(mileage)
    print(mileage, y)

    plt.scatter(mileage, price)

    plt.plot(mileage, y)
    plt.show()


def main():
    filename = get_filename()
    mileage, price = load_dataset(filename)

    model = LinearRegression(learning_rate=0.001, iteration=10)
    model.fit_gradient_descent(mileage, price)

    print_result(model)
    save_data(model)
    plot_graph(mileage, price, model)


if __name__ == "__main__":
    main()
