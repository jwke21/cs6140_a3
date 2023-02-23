"""
CS6140 Project 3
Yihan Xu
Jake Van Meter
"""

from utils import *
from regression import *
from consts import *
from pca import *
from sklearn.ensemble import RandomForestClassifier
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

mpl.use('TkAgg')


# Function to plot features in pairs
def plot_scatter_matrix(data: pd.DataFrame) -> None:
    data = pd.DataFrame(data)
    plt.figure()
    plt.title('Scatter matrix of All Independent Features')
    pd.plotting.scatter_matrix(data, figsize=(35, 35), marker='o')
    plt.savefig('images/scatter_matrix.png')


# Function to plot heatmap for all features
def plot_heatmap(data: pd.DataFrame) -> None:
    plt.figure()
    plt.title('Heatmap of All Independent Features')
    sns.heatmap(data.corr())
    plt.subplots_adjust(bottom=0.2, left=0.18)
    plt.savefig('images/heatmap_of_all_features.png')
    plt.show()


# Function to plot boxplot for all features
def boxplot_all_features(data: pd.DataFrame) -> None:
    plt.figure()
    sns.boxplot(data=data, width=0.5)
    plt.xticks(rotation=45, ha='right')
    plt.title('Boxplot For All Features')
    plt.savefig('images/boxplot_of_all_features.png')
    plt.show()


# Function to rank features using random forest classifier
def rank_feature_with_rfc(X_train: pd.DataFrame, y_train: pd.Series) -> None:
    rfc = RandomForestClassifier()
    rfc.fit(X_train, y_train)
    feature_importances = rfc.feature_importances_
    formatted_feature_importances = {}
    for i in range(len(feature_importances)):
        formatted_feature_importances[X_train.columns[i]] = {'importance': feature_importances[i]}
    sorted_importances = sorted(formatted_feature_importances.items(), key=lambda x: x[1]['importance'], reverse=True)

    for i in range(len(sorted_importances)):
        print(sorted_importances[i])


# Function to normalize dataset
def normalize_data(data: pd.DataFrame) -> pd.DataFrame:
    A = data.values
    m = np.mean(A, axis=0)
    D = A - m
    std = np.std(D, axis=0)
    D = D / std
    D = pd.DataFrame(D, columns=data.columns)
    return D


def main():
    # Read dataset
    raw_df = open_csv_as_df(RAW_CSV_PATH)
    trn_df = open_csv_as_df(TRN_CSV_PATH)
    tst_df = open_csv_as_df(TST_CSV_PATH)

    # Convert categorical data to binary
    raw_df = format_df_columns(raw_df)
    trn_df = format_df_columns(trn_df)
    tst_df = format_df_columns(tst_df)

    # Get X_train, y_train, X_test, y_test
    X_train = trn_df.drop([DEP_FEATURE], axis=1)
    y_train = trn_df[DEP_FEATURE]
    X_test = tst_df.drop([DEP_FEATURE], axis=1)
    y_test = tst_df[DEP_FEATURE]

    # Normalize the dataset
    X_train_numeric_normalized = normalize_data(X_train[NUMERIC_IND_FEATURES])
    X_test_numeric_normalized = normalize_data(X_test[NUMERIC_IND_FEATURES])
    X_train = pd.concat([X_train[CATEGORICAL_IND_FEATURES], X_train_numeric_normalized], axis=1)
    X_test = pd.concat([X_test[CATEGORICAL_IND_FEATURES], X_test_numeric_normalized], axis=1)

    # Visualize the data by plotting and creating a heatmap for all features in training dataset
    boxplot_all_features(X_train)
    plot_scatter_matrix(X_train)
    plot_heatmap(X_train)

    # Print out feature importance using rfc
    rank_feature_with_rfc(X_train, y_train)

    # Use Logistic regression with lasso regularization to find significant features
    sel_ = logistic_regression_with_lasso(X_train, y_train)
    coefficients = sel_.coef_[0]
    formatted_coefficients = {}
    for i in range(len(coefficients)):
        formatted_coefficients[X_train.columns[i]] = {'coefficient': coefficients[i]}
    for x, y in formatted_coefficients.items():
        print(x)
        print(y)

    #################### Simple Linear Regression ####################

    print("\n--------------------SIMPLE LINEAR REGRESSION WITH ALL FEATURES--------------------\n")
    # Conduct simple linear regression using all features
    model = linear_regression(X_train, y_train)
    print_linear_reg_model_metrics(model, X_train, y_train, X_test, y_test)

    # List containing tuples of (score, feature)
    top_scoring_features = []

    # Conduct linear regression on each of the individual features
    for i, feat in enumerate(DUMMY_IND_FEATURES):
        model = linear_regression(X_train[feat], y_train)
        r2_score = model.score(X_test[feat], y_test)
        # Push model's score and associated feature onto the heap
        top_scoring_features.append((-r2_score, feat))
        print_linear_reg_model_metrics(model, X_train[feat], y_train, X_test[feat], y_test)
    top_scoring_features.sort()

    # Get the top N scoring features
    N = 4
    top_features = []
    for i in range(N):
        _, feat = top_scoring_features[i]
        top_features += feat
    print(f"\n--------------------SIMPLE LINEAR REGRESSION WITH TOP {N} SCORING FEATURES--------------------\n")
    # Build a linear model out of the top N scoring features
    model = linear_regression(X_train[top_features], y_train)
    # Print the model's metrics
    print_linear_reg_model_metrics(model, X_train[top_features], y_train, X_test[top_features], y_test)

    #################### Non-Linear Feature Transformations ####################

    ### Try polynomial regression ###
    DEGREE = 3
    print("\n--------------------POLYNOMIAL REGRESSION WITH ALL FEATURES--------------------\n")
    models = polynomial_regression(X_train[DUMMY_IND_FEATURES_NONLIST], y_train, DEGREE)
    _, best_degree = top_scoring_polynomial_regression_model(models, X_test[DUMMY_IND_FEATURES_NONLIST], y_test)
    print(f"\nBest fit polynomial for all features: degree {best_degree}\n")

    top_scoring_degrees = []
    # Try polynomial regression for each feature individually
    for _, feat in enumerate(NUMERIC_IND_FEATURES):
        print(f"\n--------------------POLYNOMIAL REGRESSION FOR FEATURE '{feat}'--------------------\n")
        # Get higher-order degree models
        models = polynomial_regression(X_train[feat], y_train, DEGREE)
        # Get highest scoring model
        _, best_degree = top_scoring_polynomial_regression_model(models, X_test[feat], y_test)
        top_scoring_degrees.append(best_degree)
        # Print best fit polynomial
        print(f"\nBest fit polynomial for feature {feat}: degree {best_degree}\n")

    transformed_numeric_data_train = np.power(X_train[NUMERIC_IND_FEATURES], top_scoring_degrees)
    transformed_data_train = pd.concat([X_train[CATEGORICAL_IND_FEATURES], transformed_numeric_data_train], axis=1)
    model = linear_regression(transformed_data_train, y_train)
    transformed_numeric_data_test = np.power(X_test[NUMERIC_IND_FEATURES], top_scoring_degrees)
    transformed_data_test = pd.concat([X_test[CATEGORICAL_IND_FEATURES], transformed_numeric_data_test], axis=1)
    print_linear_reg_model_metrics(model, transformed_data_train, y_train, transformed_data_test, y_test)

    ### Try regression with square root of input ###
    print("\n--------------------REGRESSION WITH SQUARE ROOT TRANSFORMATIONS ON ALL FEATURES--------------------\n")
    models = sqrt_regression(X_train, y_train)
    top_transformation = top_scoring_sqrt_model(models, X_test, y_test)
    print(f"\nBest transformation for all features (i.e. 'non-transformed' vs. 'square root'): {top_transformation}\n")

    # Try taking the square root for each feature individually
    for _, feat in enumerate(NUMERIC_IND_FEATURES):
        print(
            f"\n--------------------REGRESSION WITH SQUARE ROOT TRANSFORMATIONS FOR FEATURE '{feat}'--------------------\n")
        # Get the models
        models = sqrt_regression(X_train[feat], y_train)
        # Evaluate the better performing model
        top_transformation = top_scoring_sqrt_model(models, X_test[feat], y_test)
        print(
            f"\nBest transformation for feature '{feat}' (i.e. 'non-transformed' vs. 'square root'): {top_transformation}\n")

    ### Try regression with cosine transformation ###
    print("\n--------------------REGRESSION WITH COSINE TRANSFORMATIONS ON ALL FEATURES--------------------\n")
    models = cosine_regression(X_train, y_train)
    top_transformation = top_scoring_cosine_model(models, X_test, y_test)
    print(f"\nBest transformation for all features (i.e. 'non-transformed' vs. 'square root'): {top_transformation}\n")

    # Try taking the cosine for each feature individually
    for _, feat in enumerate(NUMERIC_IND_FEATURES):
        print(
            f"\n--------------------REGRESSION WITH COSINE TRANSFORMATIONS FOR FEATURE '{feat}'--------------------\n")
        # Get the models
        models = cosine_regression(X_train[feat], y_train)
        # Evaluate the better performing model
        top_transformation = top_scoring_sqrt_model(models, X_test[feat], y_test)
        print(
            f"\nBest transformation for feature '{feat}' (i.e. 'non-transformed' vs. 'cosine'): {top_transformation}\n")

    # Conduct PCA on the numeric independent features of the data set
    means, stds, eigvals, eigvecs, projected_data = pca(X_train[NUMERIC_IND_FEATURES], normalize=False,
                                                        print_results=False)
    print(f"Eigenvalues:\n{eigvals}\n")
    print(f"Eigenvectors:\n{eigvecs}\n")

    totals = [0 for _ in range(len(NUMERIC_IND_FEATURES))]
    for i, vector in enumerate(eigvecs):
        eigval = eigvals[i]
        for j in range(len(vector)):
            # Multiply feature's value in eigenvector by the eigenvector's value
            totals[j] += vector[j] * eigval
    for i, feat in enumerate(NUMERIC_IND_FEATURES):
        print(f"Feature '{feat}' dot product between eigenvectors and eigenvalues: {totals[i]}")
    print("")

    # Write formatted data frames to new CSVs
    # raw_df.to_csv(FRMT_RAW_CSV_PATH, index=False)
    # trn_df.to_csv(FRMT_TRN_CSV_PATH, index=False)
    # tst_df.to_csv(FRMT_TST_CSV_PATH, index=False)

    # we need to drop the insignificant features before saving for later use
    X_train.drop(['RestingBP', 'Normal', 'ST'], inplace=True, axis=1)
    X_test.drop(['RestingBP', 'Normal', 'ST'], inplace=True, axis=1)

    # Write formatted data frames to new CSVs
    X_train.to_csv(X_TRAIN_CSV_PATH, index=False)
    X_test.to_csv(X_TEST_CSV_PATH, index=False)
    y_train.to_csv(Y_TRAIN_CSV_PATH, index=False)
    y_test.to_csv(Y_TEST_CSV_PATH, index=False)


if __name__ == "__main__":
    main()
