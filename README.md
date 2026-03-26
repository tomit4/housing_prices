# Programming In Python - Assigment 3: Using Machine Learning

## Introduction

This repository hold's my code for [Study.com](https://study.com)'s Programming
In Python course's Assignment: Using Machine Learning.

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

## Prompt

**Housing Price Prediction with Linear Regression**. Predicting real-world
outcomes using data is a cornerstone of data science and machine learning.

In this assignment, you'll build a Python program that predicts housing prices
based on house sizes using linear regression--one of the most fundamental
predictive modeling techniques. You'll practice working with CSV data,
visualizing trends, training a regression model, and creating a user-friendly
interface to make predictions.

This assignment emphasizes data handling, model interpretation, and interactive
program design using popular libraries like pandas, matplotlib, and
scikit-learn.

Your program should:

1)Load Dataset from CSV File

- Use a provided CSV file (e.g., containing columns Size (sqft) and Price ($)).
- Use pandas to read and manage the dataset.

2)Explore and Visualize the Data

- Display basic statistics (mean, min, max, etc.).
- Plot the data using matplotlib to show the relationship between house size and
  price.

3)Train a Linear Regression Model

- Use scikit-learn to fit a linear regression model to the data.
- Clearly display the model's coefficient, intercept, and R? score.

4)Visualize Model Fit

- Display a scatter plot of the data points along with the regression line.

5)Predict Price from User Input

- Prompt the user to enter a house size (in square feet).
- Predict and display the estimated house price based on the trained model.
- Handle invalid inputs gracefully (e.g., non-numeric or negative values).

6)Modular Design

- Organize the program using functions (e.g., load_data(), train_model(),
  predict_price(), etc.) to promote clarity and reusability.

**Additional Notes**

- Include sample data in your submission (housing_data.csv) with at least 20
  rows.
- Be sure to use try-except blocks to handle potential errors (e.g., file not
  found, input errors).
- Comment your code and include a short README or documentation block explaining
  how the program works.

## Grading Rubric

Your output will be graded based on the following rubric:

| **Criteria**                                   | **Excellent(5)**                                                                      |
| ---------------------------------------------- | ------------------------------------------------------------------------------------- |
| Data Loading & Exploration (x2)                | Reads CSV with pandas, displays clear summary statistics (mean, min, max, etc.)       |
| Data Visualization (x2)                        | Uses matplotlib to create clear, labeled scatter plots of data and regression line    |
| Model Training & Output (x3)                   | Accurately fits regression model with scikit-learn, displays slope, intercept, and R? |
| Prediction Functionality & Input Handling (x3) | Accepts user input, validates it, and displays accurate predictions                   |
| Program Structure & Modularity (x2)            | Code is well-organized into meaningful, reusable functions with descriptive names     |
| User Interaction & Error Handling (x3)         | Friendly interface; handles invalid or missing inputs without crashing                |

### Potential Additional Learning Resources

**Youtube**

- [Python Pandas Tutorial (Part 1): Getting Started with Data Analysis - Installation and Loading Data](https://www.youtube.com/watch?v=ZyhVh-qRZPA&list=PL-osiE80TeTt2d9bfVyTiXJA-UTHn6WwU&index=129)
- [Matplotlib Tutorial (Part 1): Creating and Customizing Our First Plots](https://www.youtube.com/watch?v=UO98lJQ3QGI&list=PL-osiE80TeTt2d9bfVyTiXJA-UTHn6WwU&index=113)
- [Scikit-learn Crash Course - Machine Learning Library for Python](https://www.youtube.com/watch?v=0B5eIE_1vpU)
- [Scikit-Learn Tutorials - Master Machine Learning](https://www.youtube.com/watch?v=SjOfbbfI2qY&list=PLcQVY5V2UY4LNmObS0gqNVyNdVfXnHwu8)
- [Scikit-Learn Python Tutorial | Machine Learning with Scikit-learn](https://www.youtube.com/watch?v=2WztaC6kyLs&list=PLS1QulWo1RIa7ha9SewcZlsTQVwL7n7oq)
- [Linear Regression, Clearly Explained!!!](https://www.youtube.com/watch?v=7ArmBVF2dCs)

**Official Docs/Tutorials**

- [Pandas Getting started tutorials](https://pandas.pydata.org/pandas-docs/stable/getting_started/intro_tutorials/index.html#)
- [Matplotlib Quick start guide](https://matplotlib.org/stable/users/explain/quick_start.html)
- [Scikit Learn User Guide](https://scikit-learn.org/stable/user_guide.html)
