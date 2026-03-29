import sys

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

"""
Housing Prices Dataset
----------------------
Source: Kaggle
Dataset: Housing Prices Dataset by Yasser H
URL: https://www.kaggle.com/datasets/yasserh/housing-prices-dataset
Accessed: March 29, 2026

Description:
This dataset contains housing information such as area (sqft), price ($),
number of bedrooms, bathrooms, stories, and other features. It is used
here for a linear regression project to predict house prices based on area.
"""


def load_data(csv_file):
    try:
        df = pd.read_csv(csv_file)
        if "area" not in df.columns or "price" not in df.columns:
            raise KeyError("Dataset must have 'area' and 'price' columns.")

        # Sample data
        print("-" * 40)
        print("Sample data (first 5 rows):")
        print(df[["area", "price"]].head(), "\n")

        # Basic dadtaset info
        print(f"Dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")
        print(f"Columns: {list(df.columns)}\n")

        # Summary stats for numeric columns
        print("Summary statistics for numeric columns:")
        print(df[["area", "price"]].describe().round(2))
        print("-" * 40)

        X = df[["area"]]
        y = df[["price"]]

        return X, y
    except FileNotFoundError:
        print("Error: Housing.csv file not found.")
        return None
    except KeyError as e:
        print(f"Error: {e}")
        return None


def explore_data(data):
    X, y = data

    area = X["area"]
    price = y["price"]

    stats = {
        "Mean": lambda s: s.mean(),
        "Median": lambda s: s.median(),
        "Mode": lambda s: s.mode()[0],
        "Min": lambda s: s.min(),
        "Max": lambda s: s.max(),
        "Sum": lambda s: s.sum(),
        "Count": lambda s: s.count(),
        "Std Dev": lambda s: s.std(),
        "Variance": lambda s: s.var(),
    }

    def print_stats(name, series):
        print(f"--- {name} ---")
        for stat_name, func in stats.items():
            value = func(series)
            if isinstance(value, float):
                print(f"{stat_name}: {value:.2f}")
            else:
                print(f"{stat_name}: {value}")
        print()

    print_stats("Area", area)
    print_stats("Price", price)

    plt.figure(figsize=(8, 6))
    plt.scatter(X, y, color="blue", alpha=0.6, label="Actual data")

    plt.xlabel("Area (sqft)")
    plt.ylabel("Price ($ millions)")
    plt.title("House Price vs Area")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()
    plt.close()


def train_model(data):
    X, y = data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)

    print("Coefficient (slope): ", model.coef_[0])
    print("Intercept: ", model.intercept_)
    print(f"R²: {r2:.4f}")

    return model


def plot_data(data, model):
    X, y = data

    plt.figure(figsize=(8, 6))
    plt.scatter(X, y, color="blue", alpha=0.6, label="Actual data")

    y_line = model.coef_[0] * X + model.intercept_
    plt.plot(X, y_line, color="red", linewidth=2, label="Regression line")

    plt.xlabel("Area (sqft)")
    plt.ylabel("Price ($ millions)")
    plt.title("House Price vs Area with Regression Line")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()
    plt.close()


def predict_price(model):
    while True:
        try:
            X_test = int(input("Please enter a house size (in square feet): "))
            if X_test < 0:
                raise ValueError("Input wasn't a positive number")
            X_test_df = pd.DataFrame([[X_test]], columns=["area"])
            y_pred = model.predict(X_test_df)
            estimated_price = y_pred.item()
            print(f"Estimated house price: ${estimated_price:,.2f}")
            again = (
                input("Do you want to predict another house price? (y/n): ")
                .strip()
                .lower()
            )
            if again != "y":
                break
        except ValueError:
            print("Sorry, that wasn't a positive number value.")
            continue


def main():
    data = load_data("./housing.csv")
    if data is not None:
        explore_data(data)
        model = train_model(data)
        plot_data(data, model)
        predict_price(model)


if __name__ == "__main__":
    main()
    print("Goodbye!")
    sys.exit()
