"""
CS6140 Project 3
Yihan Xu
Jake Van Meter
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn import linear_model as lm
from typing import *


def linear_regression(X_train: pd.DataFrame | pd.Series, y_train: pd.Series) -> lm.LinearRegression:
    if len(X_train.shape) == 1:
        # If the X_train is a series, transform it into a dataframe of shape (N rows, 1 column)
        X_train = pd.DataFrame(X_train)
    # y_train = y_train.to_numpy().reshape((-1, 1))
    # Create and train the linear model on training data
    linear_model = lm.LinearRegression().fit(X=X_train, y=y_train)
    return linear_model


def logistic_regression_with_lasso(X_train: pd.DataFrame, y_train: pd.Series) -> lm.LinearRegression:
    sel_ = LogisticRegression(C=0.5, penalty='l1', solver='liblinear', random_state=10)
    sel_.fit(X_train, y_train)
    return sel_


def polynomial_regression(X_train: pd.DataFrame | pd.Series, y_train: pd.Series, degree: int) -> List[lm.LinearRegression]:
    if len(X_train.shape) == 1:
        # If the X_train is a series, transform it into a dataframe of shape (N rows, 1 column)
        X_train = pd.DataFrame(X_train)
    models = []
    for i in range(degree):
        # Build new training df
        transformed_data = np.power(X_train, i+1)
        # Create and train the linear model on training data
        model = lm.LinearRegression().fit(X=transformed_data, y=y_train)
        models.append(model)
    return models

def top_scoring_polynomial_regression_model(models: List[lm.LinearRegression],
                                            X_test: pd.DataFrame | pd.Series,
                                            y_test: pd.Series) -> tuple[lm.LinearRegression, int]:
    if len(X_test.shape) == 1:
        # If the X_train is a series, transform it into a dataframe of shape (N rows, 1 column)
        X_test = pd.DataFrame(X_test)
    # Get the model with the best fit
    max_score = float("-inf")  # initialize to 0 correlation
    max_i = 0
    for i in range(len(models)):
        transformed_data = np.power(X_test, i+1)
        cur_model = models[i]
        cur_score = cur_model.score(transformed_data, y_test)
        print(f"Degree {i + 1} R squared score: {cur_score}")
        # Update max score
        if cur_score > max_score:
            max_score = cur_score
            max_i = i
    return models[max_i], max_i + 1

def sqrt_regression(X_train: pd.DataFrame | pd.Series, y_train: pd.Series) -> List[lm.LinearRegression]:
    if len(X_train.shape) == 1:
        # If the X_train is a series, transform it into a dataframe of shape (N rows, 1 column)
        X_train = pd.DataFrame(X_train)
    models = []
    # Build model on non-transformed data
    model = lm.LinearRegression().fit(X=X_train, y=y_train)
    models.append(model)
    # Build model on square rooted data
    transformed_data = X_train.copy()
    transformed_data[transformed_data < 0] = 0 # Replace negative numbers with 0 so no NaN values are produced from sqrt operation
    transformed_data = np.sqrt(transformed_data)
    # Create and train the linear model on training data
    model = lm.LinearRegression().fit(X=transformed_data, y=y_train)
    models.append(model)
    return models

def top_scoring_sqrt_model(models: List[lm.LinearRegression],
                           X_test: pd.DataFrame | pd.Series,
                           y_test: pd.Series) -> str:
    if len(X_test.shape) == 1:
        # If the X_train is a series, transform it into a dataframe of shape (N rows, 1 column)
        X_test = pd.DataFrame(X_test)
    # Get the r2 score of non-transformed input model
    non_transformed_score = models[0].score(X_test, y_test)
    # Get the r2 score of the transformed input model
    transformed_data = X_test.copy()
    transformed_data[transformed_data < 0] = 0 # Replace negative numbers with 0 so no NaN values are produced from sqrt operation
    transformed_data = np.sqrt(transformed_data)
    transformed_score = models[1].score(transformed_data, y_test)
    print(f"R squared score for non-transformed input: {non_transformed_score}")
    print(f"R squared score for transformed input: {transformed_score}")
    top_transformation = "non-transformed" if non_transformed_score >= transformed_score else "square root"
    return top_transformation

def cosine_regression(X_train: pd.DataFrame | pd.Series, y_train: pd.Series) -> List[lm.LinearRegression]:
    if len(X_train.shape) == 1:
        # If the X_train is a series, transform it into a dataframe of shape (N rows, 1 column)
        X_train = pd.DataFrame(X_train)
    models = []
    # Build model on non-transformed data
    model = lm.LinearRegression().fit(X=X_train, y=y_train)
    models.append(model)
    # Build model on cosine of input data
    transformed_data = pd.DataFrame(np.cos(X_train))
    # Create and train the linear model on training data
    model = lm.LinearRegression().fit(X=transformed_data, y=y_train)
    models.append(model)
    return models

def top_scoring_cosine_model(models: List[lm.LinearRegression],
                             X_test: pd.DataFrame | pd.Series,
                             y_test: pd.Series) -> str:
    if len(X_test.shape) == 1:
        # If the X_train is a series, transform it into a dataframe of shape (N rows, 1 column)
        X_test = pd.DataFrame(X_test)
    # Get the r2 score of non-transformed input model
    non_transformed_score = models[0].score(X_test, y_test)
    # Get the r2 score of the transformed input model
    transformed_data = pd.DataFrame(np.cos(X_test))
    transformed_score = models[1].score(transformed_data, y_test)
    print(f"R squared score for non-transformed input: {non_transformed_score}")
    print(f"R squared score for transformed input: {transformed_score}")
    top_transformation = "non-transformed" if non_transformed_score >= transformed_score else "cosine"
    return top_transformation
    pass

def print_linear_reg_model_metrics(model: lm.LinearRegression, X_train: pd.DataFrame | pd.Series,
                                   y_train: pd.Series, X_test: pd.DataFrame | pd.Series,
                                   y_test: pd.Series) -> None:
    # If X is a series, transform it into a dataframe of shape (N rows, 1 column)
    if len(X_train.shape) == 1:
        X_train = pd.DataFrame(X_train)
        X_test = pd.DataFrame(X_test)
    ind_vars = X_train.columns.values
    coefficients = model.coef_
    # trn_pred = np.reshape(model.predict(X_train), (-1, 1))
    # tst_pred = np.reshape(model.predict(X_test), (-1, 1))
    for i, feat in enumerate(ind_vars):
        print(f"Coefficient for {feat}: {coefficients[i]}")
    print(f"Intercept: {model.intercept_}")
    print(f"R squared on training set: {model.score(X_train, y_train)}")
    print(f"R squared on test set: {model.score(X_test, y_test)}")
    print("\n")
