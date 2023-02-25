from classifier import *
from utils import *
from consts import *
from typing import *
from naive_bayes import *
from decision_tree import *
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


def train_and_predict(classifier: ClassifierModel, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame,
                      y_test: pd.Series, classifier_type: str) -> None:
    classifier.train(X_train, y_train)
    classifier.classify(X_test, y_test)
    print(f"Result for {classifier_type}")
    classifier.evaluate()

def get_input_data() -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    X_train = open_csv_as_df(X_TRAIN_CSV_PATH)
    y_train = open_csv_as_df(Y_TRAIN_CSV_PATH)
    X_test = open_csv_as_df(X_TEST_CSV_PATH)
    y_test = open_csv_as_df(Y_TEST_CSV_PATH)
    return X_train, y_train, X_test, y_test

def svm_classification() -> None:
    X_train, y_train, X_test, y_test = get_input_data()
    # SVM classifier
    svmcl = svm.SVC(probability=True)
    svm_classifier = ClassifierModel(svmcl)
    train_and_predict(svm_classifier, X_train, y_train, X_test, y_test, "Support Vector Machine")

def svm_classification_with_iterations() -> None:
    X_train, y_train, X_test, y_test = get_input_data()
    X_train_no_drop = open_csv_as_df(X_TRAIN_NO_DROP_CSV_PATH)
    X_test_no_drop = open_csv_as_df(X_TEST_NO_DROP_CSV_PATH)
    # SVM classifier
    svmcl = svm.SVC(probability=True)
    svm_classifier = ClassifierModel(svmcl)
    train_and_predict(svm_classifier, X_train, y_train, X_test, y_test, "Support Vector Machine")

    # SVM classifier iteration 1
    svmcl = svm.SVC(probability=True, C=2.0)
    svm_classifier = ClassifierModel(svmcl)
    train_and_predict(svm_classifier, X_train, y_train, X_test, y_test, "Support Vector Machine")

    # SVM classifier iteration 2
    svmcl = svm.SVC(probability=True, kernel='poly')
    svm_classifier = ClassifierModel(svmcl)
    train_and_predict(svm_classifier, X_train, y_train, X_test, y_test, "Support Vector Machine")

    # SVM classifier iteration 3
    svmcl = svm.SVC(probability=True)
    svm_classifier = ClassifierModel(svmcl)
    train_and_predict(svm_classifier, X_train_no_drop, y_train, X_test_no_drop, y_test, "Support Vector Machine")

def logistic_regression_classifion() -> None:
    X_train, y_train, X_test, y_test = get_input_data()
    # Logistic regression classifier
    lr = LogisticRegression(C=0.5, solver='liblinear')
    lr_classifier = ClassifierModel(lr)
    train_and_predict(lr_classifier, X_train, y_train, X_test, y_test, "Logistic Regression")

def knn_classification() -> None:
    X_train, y_train, X_test, y_test = get_input_data()
    # KNN classifier
    knn = KNeighborsClassifier(n_neighbors=10)
    knn_classifier = ClassifierModel(knn)
    train_and_predict(knn_classifier, X_train, y_train, X_test, y_test, "K Nearest neighbors")

def multinomial_nb_classification() -> None:
    X_train, y_train, X_test, y_test = get_input_data()
    # Build and evaluate a Multinomial Naive Bayes model
    print("\n--------------------Multinomial Naive Bayes metrics--------------------\n")
    multinomial = NaiveBayesMultinomial()
    multinomial.train(X_train, y_train)
    multinomial.classify(X_test, y_test)
    multinomial.evaluate()

def bernoulli_nb_classification() -> None:
    X_train, y_train, X_test, y_test = get_input_data()
    # Build and evaluate a Bernoulli Naive Bayes model
    print("\n--------------------Bernoulli Naive Bayes metrics--------------------\n")
    bernoulli = NaiveBayesBernoulli()
    bernoulli.train(X_train, y_train)
    bernoulli.classify(X_test, y_test)
    bernoulli.evaluate()

def gaussian_nb_classification() -> None:
    X_train, y_train, X_test, y_test = get_input_data()
    # Build and evaluate a Gaussian Naive Bayes model
    print("\n--------------------Gaussian Naive Bayes metrics--------------------\n")
    gaussian = NaiveBayesGaussian()
    gaussian.train(X_train, y_train)
    gaussian.classify(X_test, y_test)
    gaussian.evaluate()

def gaussian_nb_classification_with_iterations() -> None:
    X_train, y_train, X_test, y_test = get_input_data()
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

def decision_tree_classification() -> None:
    X_train, y_train, X_test, y_test = get_input_data()
    # Build and evaluate a Decision Tree Classifier model
    print("\n--------------------Decision Tree Classifier metrics--------------------\n")
    dt = TreeClassifier()
    dt.train(X_train, y_train)
    dt.classify(X_test, y_test)
    dt.evaluate()


def main():
    # Run all classifications
    svm_classification()
    logistic_regression_classifion()
    knn_classification()
    multinomial_nb_classification()
    bernoulli_nb_classification()
    gaussian_nb_classification()
    decision_tree_classification()
    
    # Run iterations
    svm_classification_with_iterations()
    gaussian_nb_classification_with_iterations()


if __name__ == "__main__":
    main()
