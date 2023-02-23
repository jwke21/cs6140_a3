from __future__ import annotations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.naive_bayes as nb
import sklearn.tree as tree
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score, roc_curve
from abc import ABC
from typing import *

SkLearnClassifier = TypeVar(
    "SkLearnClassifier",
    nb.BernoulliNB,
    nb.CategoricalNB,
    nb.ComplementNB,
    nb.GaussianNB,
    nb.MultinomialNB,
    tree.DecisionTreeClassifier,
    svm.SVC
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
        # Store prediction values for later use
        self.y_pred = self.model.predict(X)
        # Store test data
        self.X_test = X
        self.y_test = y
        # Score the model
        score = self.model.score(X, y)
        # Get error rate
        self.erate_test = 1 - score

    def compute_confusion_matrix(self, plot: bool = False) -> np.ndarray:
        # Guard against if classication has not been carried out yet
        if self.y_pred is None:
            print("Model has not conducted classification yet")
            return np.ndarray()
        self.conf_matrix = confusion_matrix(self.y_test, self.y_pred)
        # Get the precision and recall
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

    def print_accuracy(self) -> None:
        return metrics.accuracy_score(self.y_test, self.y_pred)

    def get_roc(self):
        # Guard against if classication has not been carried out yet
        pass

    def get_f1(self):
        # Guard against if classication has not been carried out yet
        pass

    def clear_model(self) -> None:
        self.model = None
        self.erate_train = None
        self.erate_test = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.y_pred = None # Predicted y values obtained after classication of X_test
        self.conf_matrix = None
