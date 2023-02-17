'''
CS6140 Project 3
Yihan Xu
Jake Van Meter
'''
from utils import *

RAW_CSV_PATH = "./data/heart.csv"
TRN_CSV_PATH = "./data/heart_train_718.csv"
TST_CSV_PATH = "./data/heart_test_200.csv"
FRMT_RAW_CSV_PATH = "./data/formatted_heart.csv"
FRMT_TRN_CSV_PATH = "./data/formatted_heart_train_718.csv"
FRMT_TST_CSV_PATH = "./data/formatted_heart_test_200.csv"

IND_FEATURES = [
    "Age",                  # Numeric
    "Sex",                  # Categorical: [F=1, M=0]
    ["ATA", "NAP", "TA"],   # (orig='ChestPainType') Categorical: [ATA=(1, 0, 0), NAP=(0, 1, 0), TA=(0, 0, 1), ASY=(0, 0, 0)]
    "RestingBP",            # Numeric
    "Cholesterol",          # Numeric
    "FastingBS",            # Categorical: [Y=1, N=0]
    ["Normal", "ST"],       # (orig='RestingECG') Categorical: [Normal=(1, 0), ST=(0, 1), LVH=(0, 0)]
    "MaxHR",                # Numeric
    "ExerciseAngina",       # Categorical: [Y=1, N=0]
    "Oldpeak",              # Numeric
    ["Flat", "Up"],         # (orig='ST_Slope') Categorical: [Flat=(1, 0), Up=(0, 1), Down=(0, 0)]
]

DEP_FEATURE = "HeartDisease" # Categorical: [Y=1, N=0]


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

    # Conduct linear regression on each of the columns

    # Write formatted data frames to new CSVs
    # raw_df.to_csv(FRMT_RAW_CSV_PATH, index=False)
    # trn_df.to_csv(FRMT_TRN_CSV_PATH, index=False)
    # tst_df.to_csv(FRMT_TST_CSV_PATH, index=False)

if __name__ == "__main__":
    main()
