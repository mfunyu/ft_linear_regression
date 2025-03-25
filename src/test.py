import json
import numpy as np

from linear_regression import LinearRegression

TARGET_FILE = "trained_data.json"


def get_data():
    try:
        with open(TARGET_FILE, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Warning: {TARGET_FILE} not found")
        data = {}
    except Exception as e:
        print("Error:", e)
        exit()

    return data


def main():
    data = get_data()
    model = LinearRegression(data.get("theta0", 0), data.get("theta1", 0),
                             mean=data.get("mean", 0), std=data.get("std", 1))





if __name__ == "__main__":
    main()
