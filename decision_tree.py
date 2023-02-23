"""
CS6140 Project 3
Yihan Xu
Jake Van Meter
"""

import pandas as pd
from consts import *
from utils import *
from classifier import ClassifierModel
from sklearn.tree import DecisionTreeClassifier

class TreeClassifier(ClassifierModel):
    def __init__(self) -> None:
        super().__init__(DecisionTreeClassifier())

    def train(self, X: pd.DataFrame | pd.Series, y: pd.DataFrame | pd.Series) -> None:
        return super().train(X, y)


def main():
    X_train = open_csv_as_df(X_TRAIN_CSV_PATH)
    y_train = open_csv_as_df(Y_TRAIN_CSV_PATH)
    X_test = open_csv_as_df(X_TEST_CSV_PATH)
    y_test = open_csv_as_df(Y_TEST_CSV_PATH)

    # Build and evaluate a Decision Tree Classifier model
    dt = TreeClassifier()
    dt.train(X_train, y_train)
    dt.classify(X_test, y_test)
    print("\n--------------------Decision Tree Classifier metrics--------------------\n")
    print(f"Bias: {dt.get_bias()}")
    print(f"Variance: {dt.get_variance()}")
    dt.compute_confusion_matrix(plot=True)

if __name__ == "__main__":
    main()
