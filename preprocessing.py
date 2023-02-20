'''
CS6140 Project 3
Yihan Xu
Jake Van Meter
'''

from utils import *
from regression import *
from consts import *
from pca import *
import heapq as hq


def main():
    raw_df = open_csv_as_df(RAW_CSV_PATH)
    trn_df = open_csv_as_df(TRN_CSV_PATH)
    tst_df = open_csv_as_df(TST_CSV_PATH)

    # print(raw_df)
    raw_df = format_df_columns(raw_df)
    # print(raw_df)
    # print(trn_df)
    trn_df = format_df_columns(trn_df)
    # print(trn_df)
    # print(tst_df)
    tst_df = format_df_columns(tst_df)
    # print(tst_df)

    # Get X_train, y_train, X_test, y_test
    X_train = trn_df.drop([DEP_FEATURE], axis=1)
    y_train = trn_df[DEP_FEATURE]
    X_test = tst_df.drop([DEP_FEATURE], axis=1)
    y_test = tst_df[DEP_FEATURE]

    # Max heap containing tuples of (score, feature)
    by_score_heap = []

    # Conduct linear regression on each of the individual features
    for i, feat in enumerate(DUMMY_IND_FEATURES):
        model = linear_regression(X_train[feat], y_train)
        r2_score = model.score(X_test[feat], y_test)
        # Push model's score and associated feature onto the heap
        hq.heappush(by_score_heap, (-r2_score, feat)) # Python's heap is a min heap
        print_linear_reg_model_metrics(model, X_train[feat], y_train, X_test[feat], y_test)

    # Get the top N scoring features
    N = 5
    top_features = []
    for _ in range(N):
        _, feat = hq.heappop(by_score_heap)
        top_features += feat

    # Build a linear model out of the top N scoring features
    model = linear_regression(X_train[top_features], y_train)
    # Print the model's metrics
    print_linear_reg_model_metrics(model, X_train[top_features], y_train, X_test[top_features], y_test)

    # Conduct PCA on the dataset
    means, stds, eigvals, eigvecs, projected_data = pca(X_train[NUMERIC_IND_FEATURES], normalize=True, print_results=True)

    # Write formatted data frames to new CSVs
    # raw_df.to_csv(FRMT_RAW_CSV_PATH, index=False)
    # trn_df.to_csv(FRMT_TRN_CSV_PATH, index=False)
    # tst_df.to_csv(FRMT_TST_CSV_PATH, index=False)

if __name__ == "__main__":
    main()
