import json
import sys
import pandas as pd


def get_json_data_from_file(filename) -> dict:
    try:
        with open(filename, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Warning: {filename} not found")
        data = {}
    except Exception as e:
        print("Error:", e)
        exit()

    return data


def get_filename_from_arg() -> str:
    if len(sys.argv) != 2:
        return None

    return sys.argv[1]


def load_dataset_from_csv(filename):
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
