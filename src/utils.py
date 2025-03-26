import json
import pandas as pd
import argparse


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


def _validate_learning_rate(value):
    float_value = float(value)
    if float_value < 0 or float_value > 1:
        raise argparse.ArgumentTypeError("Learning rate must be between 0 and 1.")
    return float_value


def _validate_iteration(value):
    int_value = int(value)
    if int_value < 0:
        raise argparse.ArgumentTypeError("Iteration must be a positive integer.")
    return int_value


def get_arg_options():
    parser = argparse.ArgumentParser()

    parser.add_argument('filename', type=str, help='Filename to process')
    parser.add_argument('-l', type=_validate_learning_rate, default=0.01, help="Learning rate")
    parser.add_argument('-i', type=_validate_iteration, default=1000, help="Number of iterations")

    args = parser.parse_args()

    return args.filename, args.l, args.i
