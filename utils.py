"""
CS6140 Project 3
Jake Van Meter
Yihan Xu
"""
import pandas as pd
import numpy as np
from typing import *


def open_csv_as_df(path: str) -> pd.DataFrame:
    print(f"fetching csv from {path}")
    return pd.read_csv(path)


def format_df_columns(df: pd.DataFrame) -> pd.DataFrame:
    ret = pd.DataFrame()
    for i, val in enumerate(df.columns):
        col = df[val]
        # If the values of the column are strings, get dummy values
        if str(col.dtype) == "object":
            # Check if the values are F/M
            if "F" in col.unique():
                # Replace all F/M with 1/0 respectively
                col.replace(to_replace=["F", "M"], value=[1, 0], inplace=True)
            # Check if the values are Y/N
            elif "Y" in col.unique():
                # Replace all Y/N with 1/0 respectively
                col.replace(to_replace=["Y", "N"], value=[1, 0], inplace=True)
            else:
                # Get k-1 dummy columns to avoid the 'dummy variable trap'
                col = pd.get_dummies(col, drop_first=True)
        ret = pd.concat([ret, col], axis=1)
    return ret
