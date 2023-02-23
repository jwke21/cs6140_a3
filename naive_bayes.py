"""
CS6140 Project 3
Yihan Xu
Jake Van Meter
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from consts import *
from utils import *
from classifier import ClassifierModel
from preprocessing import normalize_data
from sklearn.naive_bayes import MultinomialNB, BernoulliNB


class NaiveBayesMultinomial(ClassifierModel):
    def __init__(self) -> None:
        super().__init__(MultinomialNB())

class NaiveBayesBernoulli(ClassifierModel):
    def __init__(self) -> None:
        super().__init__(BernoulliNB())

def main():
    df_train = open_csv_as_df(FRMT_TRN_CSV_PATH)
    df_test = open_csv_as_df(FRMT_TST_CSV_PATH)

    X_train = df_train.drop([DEP_FEATURE], axis=1)
    y_train = df_train[DEP_FEATURE]
    X_test = df_test.drop([DEP_FEATURE], axis=1)
    y_test = df_test[DEP_FEATURE]

    X_train_numeric_normalized = normalize_data(X_train[NUMERIC_IND_FEATURES])
    X_test_numeric_normalized = normalize_data(X_test[NUMERIC_IND_FEATURES])
    X_train = pd.concat([X_train[CATEGORICAL_IND_FEATURES], X_train_numeric_normalized], axis=1)
    X_test = pd.concat([X_test[CATEGORICAL_IND_FEATURES], X_test_numeric_normalized], axis=1)

    # Replace all negative values of "Oldpeak" with 0
    X_train["Oldpeak"] = X_train["Oldpeak"].apply(lambda x: 0 if x < 0 else x)
    X_test["Oldpeak"] = X_test["Oldpeak"].apply(lambda x: 0 if x < 0 else x)

    input_features = [
    "Age",
    "Sex",
    "ATA", "NAP", "TA",
    "Cholesterol",
    "FastingBS",
    "MaxHR",
    "ExerciseAngina",
    "Oldpeak",
    "Flat", "Up",
    ]

    X_train = df_train[input_features]
    X_test = df_test[input_features]

    multinomial = NaiveBayesMultinomial()
    multinomial.train(X_train, y_train)
    multinomial.classify(X_test, y_test)
    print("Multinomial Naive Bayes metrics:\n")
    print(f"Bias: {multinomial.get_bias()}")
    print(f"Variance: {multinomial.get_variance()}")
    multinomial.compute_confusion_matrix(plot=True)

    print("Bernoulli Naive Bayes metrics:\n")
    bernoulli = NaiveBayesBernoulli()
    bernoulli.train(X_train, y_train)
    bernoulli.classify(X_test, y_test)
    print(f"Bias: {bernoulli.get_bias()}")
    print(f"Variance: {bernoulli.get_variance()}")
    bernoulli.compute_confusion_matrix(plot=True)
    

if __name__ == "__main__":
    main()
