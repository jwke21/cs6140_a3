"""
CS6140 Project 3
Yihan Xu
Jake Van Meter
"""

from utils import *
from clustering import *
from regression import *
from consts import *
from pca import *


def main():
    raw_df = open_csv_as_df(RAW_CSV_PATH)
    trn_df = open_csv_as_df(TRN_CSV_PATH)
    tst_df = open_csv_as_df(TST_CSV_PATH)

    raw_df = format_df_columns(raw_df)
    trn_df = format_df_columns(trn_df)
    tst_df = format_df_columns(tst_df)

    # Get X_train, y_train, X_test, y_test
    X_train = trn_df.drop([DEP_FEATURE], axis=1)
    y_train = trn_df[DEP_FEATURE]
    X_test = tst_df.drop([DEP_FEATURE], axis=1)
    y_test = tst_df[DEP_FEATURE]

    #################### Simple Linear Regression ####################

    print("\n--------------------SIMPLE LINEAR REGRESSION WITH ALL FEATURES--------------------\n")
    # Conduct simple linear regression using all features
    model = linear_regression(X_train, y_train)
    print_linear_reg_model_metrics(model, X_train, y_train, X_test, y_test)

    # # List containing tuples of (score, feature)
    # top_scoring_features = []

    # # Conduct linear regression on each of the individual features
    # for i, feat in enumerate(DUMMY_IND_FEATURES):
    #     model = linear_regression(X_train[feat], y_train)
    #     r2_score = model.score(X_test[feat], y_test)
    #     # Push model's score and associated feature onto the heap
    #     top_scoring_features.append((-r2_score, feat))
    #     print_linear_reg_model_metrics(model, X_train[feat], y_train, X_test[feat], y_test)
    # top_scoring_features.sort()

    # # Get the top N scoring features
    # N = 4
    # top_features = []
    # for i in range(N):
    #     _, feat = top_scoring_features[i]
    #     top_features += feat
    # print(f"\n--------------------SIMPLE LINEAR REGRESSION WITH TOP {N} SCORING FEATURES--------------------\n")
    # # Build a linear model out of the top N scoring features
    # model = linear_regression(X_train[top_features], y_train)
    # # Print the model's metrics
    # print_linear_reg_model_metrics(model, X_train[top_features], y_train, X_test[top_features], y_test)

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

    ### Try regression with power transformations of 1/2 (i.e. taking sqrt of input) ###
    print("\n--------------------REGRESSION WITH SQUARE ROOT TRANSFORMATIONS ON ALL FEATURES--------------------\n")
    models = sqrt_regression(X_train, y_train)
    top_transformation = top_scoring_sqrt_model(models, X_test[DUMMY_IND_FEATURES_NONLIST], y_test)
    print(f"\nBest transformation for all features (i.e. 'non-transformed' vs. 'square root'): {top_transformation}\n")

    # Try taking the square root for each feature individually
    for _, feat in enumerate(NUMERIC_IND_FEATURES):
        print(f"\n--------------------REGRESSION WITH SQUARE ROOT TRANSFORMATIONS FOR FEATURE '{feat}'--------------------\n")
        # Get the models
        models = sqrt_regression(X_train[feat], y_train)
        # Evaluate the better performing model
        top_transformation = top_scoring_sqrt_model(models, X_test[feat], y_test)
        print(f"\nBest transformation for feature '{feat}' (i.e. 'non-transformed' vs. 'square root'): {top_transformation}\n")

    ### Try regression with cosine transformation ###
    print("\n--------------------REGRESSION WITH COSINE TRANSFORMATIONS ON ALL FEATURES--------------------\n")
    models = cosine_regression(X_train, y_train)
    top_transformation = top_scoring_cosine_model(models, X_test[DUMMY_IND_FEATURES_NONLIST], y_test)
    print(f"\nBest transformation for all features (i.e. 'non-transformed' vs. 'square root'): {top_transformation}\n")

    # Try taking the cosine for each feature individually
    for _, feat in enumerate(NUMERIC_IND_FEATURES):
        print(f"\n--------------------REGRESSION WITH COSINE TRANSFORMATIONS FOR FEATURE '{feat}'--------------------\n")
        # Get the models
        models = cosine_regression(X_train[feat], y_train)
        # Evaluate the better performing model
        top_transformation = top_scoring_sqrt_model(models, X_test[feat], y_test)
        print(f"\nBest transformation for feature '{feat}' (i.e. 'non-transformed' vs. 'cosine'): {top_transformation}\n")

    # Conduct PCA on the numeric independent features of the data set
    # means, stds, eigvals, eigvecs, projected_data = pca(X_train[NUMERIC_IND_FEATURES], normalize=True, print_results=True)

    # Write formatted data frames to new CSVs
    # raw_df.to_csv(FRMT_RAW_CSV_PATH, index=False)
    # trn_df.to_csv(FRMT_TRN_CSV_PATH, index=False)
    # tst_df.to_csv(FRMT_TST_CSV_PATH, index=False)

    # print pairs
    # plot_pairs(tst_df)
    # cluster_features(tst_df)


if __name__ == "__main__":
    main()
