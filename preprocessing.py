"""
CS6140 Project 3
Yihan Xu
Jake Van Meter
"""

from utils import *
from regression import *
from consts import *
# from pca import *
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import heapq as hq
import pandas as pd

mpl.use('TkAgg')


# Function to plot features in pairs
def plot_scatter_matrix(data: pd.DataFrame) -> None:
    data = pd.DataFrame(data)
    Y = data.iloc[:, 1]
    pd.plotting.scatter_matrix(data, figsize=(35, 35), marker='o')
    plt.figure()
    plt.title('Scatter matrix of All Independent Features')
    plt.savefig('images/scatter_matrix.png')


# Function to plot heatmap for all features
def plot_heatmap(data: pd.DataFrame) -> None:
    plt.figure()
    plt.title('Heatmap of All Independent Features')
    sns.heatmap(data.corr())
    plt.subplots_adjust(bottom=0.2, left=0.18)
    plt.savefig('images/heatmap_of_all_features.png')
    plt.show()


def boxplot_all_features(data: pd.DataFrame) -> None:
    plt.figure()
    sns.boxplot(data=data, width=0.5)
    plt.xticks(rotation=45, ha='right')
    plt.title('Boxplot For All Features')
    plt.savefig('images/boxplot_of_all_features.png')
    plt.show()


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

    # Normalize the dataset
    raw_df = normalize_data(raw_df)
    trn_df = normalize_data(trn_df)
    tst_df = normalize_data(tst_df)

    # Get X_train, y_train, X_test, y_test
    X_train = trn_df.drop([DEP_FEATURE], axis=1)
    y_train = trn_df[DEP_FEATURE]
    X_test = tst_df.drop([DEP_FEATURE], axis=1)
    y_test = tst_df[DEP_FEATURE]

    # Visualize the data by plotting and creating a heatmap for all features in training dataset
    boxplot_all_features(X_train)
    plot_scatter_matrix(X_train)
    plot_heatmap(X_train)

    # Max heap containing tuples of (score, feature)
    # by_score_heap = []

    # Conduct linear regression on each of the individual features
    # for i, feat in enumerate(DUMMY_IND_FEATURES):
    #     model = linear_regression(X_train[feat], y_train)
    #     r2_score = model.score(X_test[feat], y_test)
    #     # Push model's score and associated feature onto the heap
    #     hq.heappush(by_score_heap, (-r2_score, feat))  # Python's heap is a min heap
    #     print_linear_reg_model_metrics(model, X_train[feat], y_train, X_test[feat], y_test)

    # Get the top N scoring features
    # N = 4
    # top_features = []
    # for _ in range(N):
    #     _, feat = hq.heappop(by_score_heap)
    #     top_features += feat

    # Build a linear model out of the top N scoring features
    # model = linear_regression(X_train[top_features], y_train)
    # Print the model's metrics
    # print_linear_reg_model_metrics(model, X_train[top_features], y_train, X_test[top_features], y_test)

    # Conduct PCA on the numeric independent features of the data set means, stds, eigvals, eigvecs, projected_data =
    # pca(X_train[NUMERIC_IND_FEATURES], normalize=True, print_results=True)

    # Write formatted data frames to new CSVs
    # raw_df.to_csv(FRMT_RAW_CSV_PATH, index=False)
    # trn_df.to_csv(FRMT_TRN_CSV_PATH, index=False)
    # tst_df.to_csv(FRMT_TST_CSV_PATH, index=False)


if __name__ == "__main__":
    main()
