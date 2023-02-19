"""
CS6140 Project 3
Yihan Xu
Jake Van Meter
<<<<<<< Updated upstream
'''

from utils import *
from regression import *
from consts import *
from pca import *
import heapq as hq
=======
"""
from utils import *
from regression import *
from consts import *
from pca import *
import heapq as hq
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.use('TkAgg')

RAW_CSV_PATH = "./data/heart.csv"
TRN_CSV_PATH = "./data/heart_train_718.csv"
TST_CSV_PATH = "./data/heart_test_200.csv"
FRMT_RAW_CSV_PATH = "./data/formatted_heart.csv"
FRMT_TRN_CSV_PATH = "./data/formatted_heart_train_718.csv"
FRMT_TST_CSV_PATH = "./data/formatted_heart_test_200.csv"

IND_FEATURES = [
    "Age",  # Numeric
    "Sex",  # Categorical: [F=1, M=0]
    ["ATA", "NAP", "TA"],
    # (orig='ChestPainType') Categorical: [ATA=(1, 0, 0), NAP=(0, 1, 0), TA=(0, 0, 1), ASY=(0, 0, 0)]
    "RestingBP",  # Numeric
    "Cholesterol",  # Numeric
    "FastingBS",  # Categorical: [Y=1, N=0]
    ["Normal", "ST"],  # (orig='RestingECG') Categorical: [Normal=(1, 0), ST=(0, 1), LVH=(0, 0)]
    "MaxHR",  # Numeric
    "ExerciseAngina",  # Categorical: [Y=1, N=0]
    "Oldpeak",  # Numeric
    ["Flat", "Up"],  # (orig='ST_Slope') Categorical: [Flat=(1, 0), Up=(0, 1), Down=(0, 0)]
]

DEP_FEATURE = "HeartDisease"  # Categorical: [Y=1, N=0]


def print_pairs(data: pd.DataFrame) -> None:
    data = pd.DataFrame(data)
    Y = data.iloc[:, 1]
    pd.plotting.scatter_matrix(data, c=Y, figsize=(50, 50), marker='o')
    plt.savefig('images/scatter_matrix.png')

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
    N = 4
    top_features = []
    for _ in range(N):
        _, feat = hq.heappop(by_score_heap)
        top_features += feat

    # Build a linear model out of the top N scoring features
    model = linear_regression(X_train[top_features], y_train)
    # Print the model's metrics
    print_linear_reg_model_metrics(model, X_train[top_features], y_train, X_test[top_features], y_test)

    # Conduct PCA on the dataset
    means, stds, eigvals, eigvecs, projected_data = pca(X_train, normalize=True, print_results=True)

    # Write formatted data frames to new CSVs
    # raw_df.to_csv(FRMT_RAW_CSV_PATH, index=False)
    # trn_df.to_csv(FRMT_TRN_CSV_PATH, index=False)
    # tst_df.to_csv(FRMT_TST_CSV_PATH, index=False)

    # print pairs
    print_pairs(tst_df)


if __name__ == "__main__":
    main()
