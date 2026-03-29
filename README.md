# Programming In Python - Assigment 3: Using Machine Learning

## Introduction

This repository holds my code for [Study.com](https://study.com)'s Programming
In Python course's Assignment: Using Machine Learning. The program predicts
housing prices based solely on house size (area in square feet) using a simple
linear regression model built with Python.

**Cloning Repository**

```sh
git clone <this url> && cd housing_prices
```

**Virtual Environment**

```sh
python -m venv .venv
```

```sh
source .venv/bin/activate
```

```sh
python -m pip install --upgrade pip
```

**Install Dependencies**

```sh
python -m pip install -r requirements.txt
```

## How the Program Works

**1. Load Dataset**

- Reads a CSV file (`housing.csv`) containing `area` and `price` columns using
  pandas.
- The program prints sample data, column info, and summary statistics.

**2. Explore and Visualize Data**

- Displays basic statistics (mean, median, min, max, etc.) for area and price.
- Plots a scatter diagram of house prices versus area.

**3. Train Linear Regression Model**

- Scales the `area` feature using scikit-learn's `StandardScaler`.
- Fits a linear regression model to predict price from area.
- Outputs the model's coefficient (slope), interpcept and R<sup>2</sup> score.

**4. Visualize Model Fit**

- Overlays the regression line on the scatter plot of the data.

**5. Predict Price from User Input**

- Prompts the user to enter a house size (sqft), validates the input, and
  predicts the estimated price using the trained model.

**6. Modular Design**

Functions include:

- `load_data()` - loads CSV data
- `explore_data()` - shows stats and initial plot
- `train_model()` - scale data and train regression model
- `plot_data()` - plot regression line on scatter plot
- `predict_price()` - interactive price prediction

The program uses try-except blocks to handle file errors and invalid user inputs
gracefully and is commented minimally but clearly following Pythonic
conventions.

### References

The following article(s) were used as references for this project:

- [Sklearn Linear Regression: A Complete Guide with Examples](https://www.datacamp.com/tutorial/sklearn-linear-regression)

### Resources

The dataset for this project can be found
[here on kaggle](https://www.kaggle.com/code/yasserh/housing-price-prediction-best-ml-algorithms).
