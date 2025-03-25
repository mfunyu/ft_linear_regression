import json

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

    print("Enter 'exit' to quit")
    while (1):
        try:
            mileage = input("Enter a mileage: ")
            if mileage == "exit":
                break
            mileage = int(mileage)
            if mileage < 0:
                raise ValueError("Mileage must cannot be negative")
        except Exception as e:
            print("Error:", e)
            continue

        try:
            estimated_price = model.predict(mileage)
            print("Estimated price:", round(estimated_price))
        except Exception as e:
            print("Error:", e)


if __name__ == "__main__":
    main()
