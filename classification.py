from classifier import *
from utils import *
from consts import *
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


def train_and_predict(classifier: ClassifierModel, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame,
                      y_test: pd.Series, classifier_type: str) -> None:
    classifier.train(X_train, y_train)
    classifier.classify(X_test, y_test)
    print(f"Result for {classifier_type}")
    classifier.evaluate()


def main():
    X_train = open_csv_as_df(X_TRAIN_CSV_PATH)
    y_train = open_csv_as_df(Y_TRAIN_CSV_PATH)
    X_test = open_csv_as_df(X_TEST_CSV_PATH)
    y_test = open_csv_as_df(Y_TEST_CSV_PATH)

    # SVM classifier
    svmcl = svm.SVC(probability=True)
    svm_classifier = ClassifierModel(svmcl)
    train_and_predict(svm_classifier, X_train, y_train, X_test, y_test, "Support Vector Machine")

    # Logistic regression classifier
    lr = LogisticRegression(C=0.5, solver='liblinear')
    lr_classifier = ClassifierModel(lr)
    train_and_predict(lr_classifier, X_train, y_train, X_test, y_test, "Logistic Regression")

    # KNN classifier
    knn = KNeighborsClassifier(n_neighbors=10)
    knn_classifier = ClassifierModel(knn)
    train_and_predict(knn_classifier, X_train, y_train, X_test, y_test, "K Nearest neighbors")


if __name__ == "__main__":
    main()
