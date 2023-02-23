from classifier import *
from utils import *
from consts import *


def main():
    X_train = open_csv_as_df(X_TRAIN_CSV_PATH)
    y_train = open_csv_as_df(Y_TRAIN_CSV_PATH)
    X_test = open_csv_as_df(X_TEST_CSV_PATH)
    y_test = open_csv_as_df(Y_TEST_CSV_PATH)

    # SVM classifier
    svmcl = svm.SVC()
    svm_classifier = ClassifierModel(svmcl)
    svm_classifier.train(X_train, y_train)
    svm_classifier.classify(X_test, y_test)
    print("Support Virtual Machine metrics:\n")
    print(f"Bias: {svm_classifier.get_bias()}")
    print(f"Variance: {svm_classifier.get_variance()}")
    svm_classifier.compute_confusion_matrix(plot=True)
    print(f"Accuracy: {svm_classifier.print_accuracy()}")


if __name__ == "__main__":
    main()
