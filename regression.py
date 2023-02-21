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
