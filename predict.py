import json

from linear_regression import LinearRegression

FILENAME = "trained_data.json"


def get_data():
    try:
        with open(FILENAME, "r") as f:
            data = json.load(f)
            theta0 = data.get("theta0", 0)
            theta1 = data.get("theta1", 0)
    except FileNotFoundError:
        print(f"Warning: {FILENAME} not found")
        theta0 = 0
        theta1 = 0
    except Exception as e:
        print("Error:", e)
        exit()

    return theta0, theta1


def main():
    theta0, theta1 = get_data()

    model = LinearRegression(theta0, theta1)

    while (1):
        try:
            mileage = input("Enter a mileage: ")
            mileage = int(mileage)
        except Exception as e:
            print("Error:", e)
            continue

        estimated_price = model.predict(mileage)
        print("Estimated price:", estimated_price)


if __name__ == "__main__":
    main()
