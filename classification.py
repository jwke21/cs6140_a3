from regression import *
from sklearn import svm


# SVM
def svm_classifier(X_train: pd.DataFrame, y_train: pd.Series):
    svmcl = svm.SVC()
    svmcl.fit(X_train, y_train)
    return svmcl


