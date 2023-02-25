"""
CS6140 Project 3
Yihan Xu
Jake Van Meter
"""

RAW_CSV_PATH = "./data/heart.csv"
TRN_CSV_PATH = "./data/heart_train_718.csv"
TST_CSV_PATH = "./data/heart_test_200.csv"
FRMT_RAW_CSV_PATH = "./data/formatted_heart.csv"
FRMT_TRN_CSV_PATH = "./data/formatted_heart_train_718.csv"
FRMT_TST_CSV_PATH = "./data/formatted_heart_test_200.csv"
X_TRAIN_CSV_PATH = "./data/X_train.csv"
Y_TRAIN_CSV_PATH = "./data/y_train.csv"
X_TEST_CSV_PATH = "./data/X_test.csv"
Y_TEST_CSV_PATH = "./data/y_test.csv"
X_TRAIN_NO_DROP_CSV_PATH = "./data/X_train_no_drop.csv"
X_TEST_NO_DROP_CSV_PATH = "./data/X_test_no_drop.csv"

IND_FEATURES = [
    "Age",              # Numeric
    "Sex",              # Categorical
    "ChestPainType",    # Categorical
    "RestingBP",        # Numeric
    "Cholesterol",      # Numeric
    "FastingBS",        # Categorical
    "RestingECG",       # Categorical
    "MaxHR",            # Numeric
    "ExerciseAngina",   # Categorical
    "Oldpeak",          # Numeric
    "ST_Slope"          # Categorical
]

NUMERIC_IND_FEATURES = [
    "Age",
    "RestingBP",
    "Cholesterol",
    "MaxHR",
    "Oldpeak",
]

CATEGORICAL_IND_FEATURES = [
    "Sex",
    "ATA", "NAP", "TA",
    "FastingBS",
    "Normal", "ST",
    "ExerciseAngina",
    "Flat", "Up",
]

DUMMY_IND_FEATURES = [
    ["Age"],                # Numeric
    ["Sex"],                # Categorical: [F=1, M=0]
    ["ATA", "NAP", "TA"],   # (orig='ChestPainType') Categorical: [ATA=(1, 0, 0), NAP=(0, 1, 0), TA=(0, 0, 1), ASY=(0, 0, 0)]
    ["RestingBP"],          # Numeric
    ["Cholesterol"],        # Numeric
    ["FastingBS"],          # Categorical: [Y=1, N=0]
    ["Normal", "ST"],       # (orig='RestingECG') Categorical: [Normal=(1, 0), ST=(0, 1), LVH=(0, 0)]
    ["MaxHR"],              # Numeric
    ["ExerciseAngina"],     # Categorical: [Y=1, N=0]
    ["Oldpeak"],            # Numeric
    ["Flat", "Up"],         # (orig='ST_Slope') Categorical: [Flat=(1, 0), Up=(0, 1), Down=(0, 0)]
]

DUMMY_IND_FEATURES_NONLIST = [
    "Age",                # Numeric
    "Sex",                # Categorical: [F=1, M=0]
    "ATA", "NAP", "TA",   # (orig='ChestPainType') Categorical: [ATA=(1, 0, 0), NAP=(0, 1, 0), TA=(0, 0, 1), ASY=(0, 0, 0)]
    "RestingBP",          # Numeric
    "Cholesterol",        # Numeric
    "FastingBS",          # Categorical: [Y=1, N=0]
    "Normal", "ST",       # (orig='RestingECG') Categorical: [Normal=(1, 0), ST=(0, 1), LVH=(0, 0)]
    "MaxHR",              # Numeric
    "ExerciseAngina",     # Categorical: [Y=1, N=0]
    "Oldpeak",            # Numeric
    "Flat", "Up",         # (orig='ST_Slope') Categorical: [Flat=(1, 0), Up=(0, 1), Down=(0, 0)]
]

DEP_FEATURE = "HeartDisease" # Categorical: [Y=1, N=0]