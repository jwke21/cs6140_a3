from __future__ import annotations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.naive_bayes as nb
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from abc import ABC
from typing import *

SkLearnClassifier = TypeVar(
    "SkLearnClassifier",
    nb.BernoulliNB,
    nb.CategoricalNB,
    nb.ComplementNB,
    nb.GaussianNB,
    nb.MultinomialNB,
    svm
    # TODO: Add any SkLearn classifiers you use here
)


class ClassifierModel(object):
    def __init__(self, model: SkLearnClassifier) -> None:
        self.clear_model()
        self.model = model

    def train(self, X: pd.DataFrame | pd.Series, y: pd.DataFrame | pd.Series) -> None:
        self.model.fit(X, y)
        # Store training data
        self.X_train = X
        self.y_train = y
        # Calculate bias for future reference
        self.erate_train = 1 - self.model.score(X, y)

    def get_bias(self) -> float:
        if not self.erate_train:
            print("Error rate on training set has not been computed")
            return 0.0
        return self.erate_train

    def get_variance(self) -> float:
        if not self.erate_train:
            print("Error rate on training set has not been computed")
            return 0.0
        if not self.erate_test:
            print("Error rate on test set has not been computed")
            return 0.0
        return self.erate_test - self.erate_train

    def classify(self, X: pd.DataFrame | pd.Series, y: pd.DataFrame | pd.Series) -> None:
        score = self.model.score(X, y)
        # Store test data
        self.X_test = X
        self.y_test = y
        # Get error rate
        self.erate_test = 1 - score

    def compute_confusion_matrix(self, plot: bool = False) -> np.ndarray:
        if self.conf_matrix is not None:
            return self.conf_matrix
        y_pred = self.model.predict(self.X_test)
        self.conf_matrix = confusion_matrix(self.y_test, y_pred)
        if plot:
            self.plot_conf_matrix()
        return self.conf_matrix

    def plot_conf_matrix(self) -> None:
        if self.conf_matrix is None:
            print("Confusion matrix has not been computed")
            return
        print("\n--------------------CONFUSION MATRIX--------------------\n")
        print(f"{self.conf_matrix}\n")
        disp = ConfusionMatrixDisplay(self.conf_matrix)
        disp.plot()
        plt.show()

    def print_accuracy(y_test: pd.Series, y_pred: pd.Series) -> None:
        print(metrics.accuracy_score(y_test, y_pred))

    def get_roc(self):
        pass

    def get_f1(self):
        pass

    def clear_model(self) -> None:
        self.model = None
        self.erate_train = None
        self.erate_test = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.conf_matrix = None
