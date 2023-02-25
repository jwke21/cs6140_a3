"""
CS6140 Project 3
Jake Van Meter
Yihan Xu
"""

import pandas as pd
from consts import *
from utils import *
from classifier import ClassifierModel
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB


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


class NaiveBayesGaussian(ClassifierModel):
    def __init__(self, prior_probabilities: Tuple[float, float] = None) -> None:
        super().__init__(GaussianNB(priors=prior_probabilities))

    def iteration1(self) -> None:
        X_train, y_train = self.X_train, self.y_train
        X_test, y_test = self.X_test, self.y_test
        self.clear_model()
        # Adjust prior probabilities for HeartDisease classes
        prior_probabilities = [0.4, 0.6]
        self.model = GaussianNB(priors=prior_probabilities)
        self.train(X_train, y_train)
        self.classify(X_test, y_test)
        self.evaluate(True, True, True, True, False, False)

    def iteration2(self) -> None:
        ind_features = [
            "Age",
            "Cholesterol",
            "MaxHR",
            "Oldpeak",
        ]
        X_train, y_train = self.X_train[ind_features].copy(), self.y_train.copy()
        X_test, y_test = self.X_test[ind_features].copy(), self.y_test.copy()
        self.clear_model()
        prior_probabilities = [0.4, 0.6]
        self.model = GaussianNB(priors=prior_probabilities)
        self.train(X_train, y_train)
        self.classify(X_test, y_test)
        self.evaluate(True, True, True, True, False, False)

    def iteration3(self, X_train, y_train, X_test, y_test) -> None:
        prior_probabilities = [0.4, 0.6]
        smoothing = 0.1
        self.clear_model()
        self.model = GaussianNB(priors=prior_probabilities, var_smoothing=smoothing)
        self.train(X_train, y_train)
        self.classify(X_test, y_test)
        self.evaluate(True, True, True, True, False, False)


def main():
    X_train = open_csv_as_df(X_TRAIN_CSV_PATH)
    y_train = open_csv_as_df(Y_TRAIN_CSV_PATH)
    X_test = open_csv_as_df(X_TEST_CSV_PATH)
    y_test = open_csv_as_df(Y_TEST_CSV_PATH)

    # Build and evaluate a Multinomial Naive Bayes model
    print("\n--------------------Multinomial Naive Bayes metrics--------------------\n")
    multinomial = NaiveBayesMultinomial()
    multinomial.train(X_train, y_train)
    multinomial.classify(X_test, y_test)
    multinomial.evaluate()

    # Build and evaluate a Bernoulli Naive Bayes model
    print("\n--------------------Bernoulli Naive Bayes metrics--------------------\n")
    bernoulli = NaiveBayesBernoulli()
    bernoulli.train(X_train, y_train)
    bernoulli.classify(X_test, y_test)
    bernoulli.evaluate()

    # Build and evaluate a Gaussian Naive Bayes model
    print("\n--------------------Gaussian Naive Bayes metrics--------------------\n")
    gaussian = NaiveBayesGaussian()
    gaussian.train(X_train, y_train)
    gaussian.classify(X_test, y_test)
    gaussian.evaluate()

    # GNB Iteration 1: Alter the prior probabilities
    print("\n--------------------Gaussian Naive Bayes Iteration 1--------------------\n")
    gaussian.iteration1()

    # GNB Iteration 2: Modify the input variables
    print("\n--------------------Gaussian Naive Bayes Iteration 2--------------------\n")
    gaussian.iteration2()

    # GNB Iteration 3
    print("\n--------------------Gaussian Naive Bayes Iteration 3--------------------\n")
    gaussian.iteration3(X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    main()
