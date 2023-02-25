"""
CS6140 Project 3
Jake Van Meter
Yihan Xu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.naive_bayes as nb
import sklearn.tree as tree
from sklearn import svm
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from typing import *


SkLearnClassifier = TypeVar(
    "SkLearnClassifier",
    nb.BernoulliNB,
    nb.CategoricalNB,
    nb.ComplementNB,
    nb.GaussianNB,
    nb.MultinomialNB,
    tree.DecisionTreeClassifier,
    svm.SVC,
    LogisticRegression,
    KNeighborsClassifier
    # TODO: Add any SkLearn classifiers you use here
)


class ClassifierModel(object):
    def __init__(self, model: SkLearnClassifier) -> None:
        self.clear_model()
        self.model = model

    def train(self, X: pd.DataFrame | pd.Series, y: pd.DataFrame | pd.Series) -> None:
        # Store training data
        self.X_train = X
        self.y_train = np.ravel(y)
        # Train the model
        self.model.fit(X, self.y_train)
        # Get mean accuracy on training set
        self.y_pred_train = self.model.predict(X)
        # score = metrics.accuracy_score(y, self.y_pred_train)
        self.accuracy_train = self.model.score(X, y)
        # Get error rate on training set
        self.erate_train = 1 - self.accuracy_train

    def classify(self, X: pd.DataFrame | pd.Series, y: pd.DataFrame | pd.Series) -> None:
        # Store test data
        self.X_test = X
        self.y_test = np.ravel(y)
        # Get mean accuracy on test set
        self.y_pred_test = self.model.predict(X)
        # score = metrics.accuracy_score(y, self.y_pred_test)
        self.accuracy_test = self.model.score(X, self.y_test)
        # Get error rate on test set
        self.erate_test = 1 - self.accuracy_test

    def compute_confusion_matrix(self, plot: bool = False) -> np.ndarray:
        # Guard against if classication has not been carried out yet
        if self.y_pred_test is None:
            print("Model has not conducted classification yet")
            return np.ndarray()
        self.conf_matrix = metrics.confusion_matrix(self.y_test, self.y_pred_test)
        # Get the precision and recall
        if plot:
            self.plot_conf_matrix()
        return self.conf_matrix

    def plot_conf_matrix(self) -> None:
        if self.conf_matrix is None:
            print("Confusion matrix has not been computed")
            return
        print("\nConfusion Matrix:\n")
        print(f"{self.conf_matrix}\n")
        disp = metrics.ConfusionMatrixDisplay(self.conf_matrix)
        disp.plot(cmap="Blues", colorbar=False)
        plt.show()

    def print_accuracy(self) -> None:
        return metrics.accuracy_score(self.y_test, self.y_pred_test)

    def get_accuracy(self) -> None:
        if self.accuracy_test is None:
            print("Accuracy rate on the test set has not yet been calculated")
        return self.accuracy_train

    def get_bias(self) -> float:
        if self.erate_train is None:
            print("Error rate on training set has not been computed")
            return 0.0
        return self.erate_train

    def get_variance(self) -> float:
        if self.erate_train is None:
            print("Error rate on training set has not been computed")
            return 0.0
        if self.erate_test is None:
            print("Error rate on test set has not been computed")
            return 0.0
        return self.erate_test - self.erate_train

    def compute_roc(self, plot: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Get the the positive probabilities (i.e. that y is True or 1)
        proba = self.model.predict_proba(self.X_test)[:, 1]
        self.roc_characteristics = metrics.roc_curve(self.y_test, proba)
        self.fpr, self.tpr, _ = self.roc_characteristics
        # Print the area under curve
        self.auc = metrics.roc_auc_score(self.y_test, proba)
        print(f"Area Under Curve (AUC): {self.auc}")
        if plot:
            self.plot_roc()
        return self.roc_characteristics

    def plot_roc(self, roc_characteristics: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None) -> None:
        if roc_characteristics is None:
            fpr, tpr, thresholds = self.roc_characteristics
        else:
            fpr, tpr, thresholds = roc_characteristics
        plt.plot(fpr, tpr, label="ROC Curve")
        plt.xlabel("False Positive Rate (FPR)")
        plt.ylabel("True Positive Rate (TPR)")
        plt.legend()
        plt.show()
        

    def compute_f1_score(self, print_report: bool = False, plot_prc: bool = False) -> float:
        # F = 2 * (precision * recall) / (precision + recall) 
        self.fscore = metrics.f1_score(self.y_test, self.y_pred_test)
        if print_report:
            print(metrics.classification_report(self.y_test, self.y_pred_test))
        if plot_prc:
            self.plot_prc()
        return self.fscore

    def plot_prc(self) -> None:
        # Get the the positive probabilities (i.e. that y is True or 1)
        proba = self.model.predict_proba(self.X_test)[:, 1]
        precision, recall, thresholds = metrics.precision_recall_curve(self.y_test, proba)
        plt.plot(precision, recall, label="PRC Curve")
        plt.xlabel("Precision")
        plt.ylabel("Recall")
        plt.legend()
        plt.show()

    def evaluate(self, show_bias: bool = True, show_variance: bool = True, show_accuracy: bool = True, show_cm: bool = True, show_prc: bool = True, show_roc: bool = True) -> None:
        if show_bias:
            print(f"Bias: {self.get_bias()}")
        if show_variance:
            print(f"Variance: {self.get_variance()}")
        if show_accuracy:
            print(f"Accuracy on training set: {self.accuracy_train}")
            print(f"Accuracy on test set: {self.accuracy_test}")
        if show_cm:
            self.compute_confusion_matrix(plot=True)
        if show_prc:
            self.compute_f1_score(print_report=True, plot_prc=True)
        if show_roc:
            self.compute_roc(plot=True)
        print("")

    def clear_model(self) -> None:
        self.model = None
        self.erate_train = None
        self.erate_test = None
        self.X_train = None
        self.y_train = None
        self.y_pred_train = None # Predicted y values obtained after classication of X_train
        self.X_test = None
        self.y_test = None
        self.y_pred_test = None # Predicted y values obtained after classication of X_test
        self.conf_matrix = None
        self.accuracy_train = None
        self.accuracy_test = None
        self.fpr = None
        self.tpr = None
        self.roc_characteristics = None
        self.auc = None
        self.precision = None
        self.recall = None
        self.fscore = None
