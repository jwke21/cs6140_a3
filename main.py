"""
CS6140 Project 3
Yihan Xu
Jake Van Meter
"""

import sys
import preprocessing as pp
import classification as cls


def main():
    argv = sys.argv
    # If desired task is not specified, run all tasks
    if len(argv) == 1:
        argv.append("all")
    #################### Preprocessing ####################
    if "all" in argv or "1" in argv:
        pp.main()
    #################### Classification with iteration ####################
    if "all" in argv or "2" in argv:
        cls.main()
    #################### KNN ####################
    if "knn" in argv:
        cls.knn_classification()
    #################### Logistic Regression ####################
    if "lr" in argv:
        cls.logistic_regression_classifion()
    #################### SVM ####################
    if "svm" in argv:
        cls.svm_classification_with_iterations()
    #################### Multinomial Naive Bayes ####################
    if "mnb" in argv:
        cls.multinomial_nb_classification()
    #################### Bernoulli Naive Bayes ####################
    if "bnb" in argv:
        cls.bernoulli_nb_classification()
    #################### Gaussian Naive Bayes ####################
    if "gnb" in argv:
        cls.gaussian_nb_classification_with_iterations()
    #################### Decision Tree ####################
    if "dt" in argv:
        cls.decision_tree_classification()

if __name__ == "__main__":
    main()
