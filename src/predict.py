import utils
from linear_regression import LinearRegression

TARGET_FILE = "trained_data.json"


def main():
    data = utils.get_json_data_from_file(TARGET_FILE)
    model = LinearRegression(data.get("theta0", 0), data.get("theta1", 0),
                             mean=data.get("mean", 0), std=data.get("std", 1))

    print("Enter 'exit' to quit")
    while (1):
        try:
            input_str = input("Enter a mileage: ")
            if input_str == "exit":
                break
            mileage = int(input_str)
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
