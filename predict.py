import json

from linear_regression import LinearRegression

FILENAME = "trained_data.json"


def get_data():
    try:
        with open(FILENAME, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Warning: {FILENAME} not found")
        data = {}
    except Exception as e:
        print("Error:", e)
        exit()

    return data


def main():
    data = get_data()

    model = LinearRegression(data.get("theta0", 0), data.get("theta1", 0),
                             mean=data.get("mean", 0), std=data.get("std", 1))

    while (1):
        try:
            mileage = input("Enter a mileage: ")
            mileage = int(mileage)
        except Exception as e:
            print("Error:", e)
            continue

        estimated_price = model.predict(mileage)
        print("Estimated price:", round(estimated_price))


if __name__ == "__main__":
    main()
