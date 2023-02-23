"""
CS6140 Project 3
Yihan Xu
Jake Van Meter
"""

import pandas as pd
from consts import *
from utils import *
from classifier import ClassifierModel
from sklearn.naive_bayes import MultinomialNB, BernoulliNB


class NaiveBayesMultinomial(ClassifierModel):
    def __init__(self) -> None:
        super().__init__(MultinomialNB())

    # Overrided method so that X is formatted correctly
    def train(self, X: pd.DataFrame | pd.Series, y: pd.DataFrame | pd.Series) -> None:
        # Replace all negative values with 0
        X_new = X.mask(X < 0, 0)
        super().train(X_new, y)

    # Overrided method so that X is formatted correctly
    def classify(self, X: pd.DataFrame | pd.Series, y: pd.DataFrame | pd.Series) -> None:
        # Replace all negative values with 0
        X_new = X.mask(X < 0, 0)
        return super().classify(X_new, y)

class NaiveBayesBernoulli(ClassifierModel):
    def __init__(self) -> None:
        super().__init__(BernoulliNB())

def main():
    X_train = open_csv_as_df(X_TRAIN_CSV_PATH)
    y_train = open_csv_as_df(Y_TRAIN_CSV_PATH)
    X_test = open_csv_as_df(X_TEST_CSV_PATH)
    y_test = open_csv_as_df(Y_TEST_CSV_PATH)

    # Build and evaluate a Multinomial Naive Bayes model
    multinomial = NaiveBayesMultinomial()
    multinomial.train(X_train, y_train)
    multinomial.classify(X_test, y_test)
    print("\nMultinomial Naive Bayes metrics:\n")
    print(f"Bias: {multinomial.get_bias()}")
    print(f"Variance: {multinomial.get_variance()}")
    multinomial.compute_confusion_matrix(plot=True)

    # Build and evaluate a Bernoulli Naive Bayes model
    bernoulli = NaiveBayesBernoulli()
    bernoulli.train(X_train, y_train)
    bernoulli.classify(X_test, y_test)
    print("\nBernoulli Naive Bayes metrics:\n")
    print(f"Bias: {bernoulli.get_bias()}")
    print(f"Variance: {bernoulli.get_variance()}")
    bernoulli.compute_confusion_matrix(plot=True)
    

if __name__ == "__main__":
    main()
