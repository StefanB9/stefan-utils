"""module for dataframe optimizations"""
import numpy as np
import pandas as pd


# pylint: disable-next=too-many-branches
def df_optimize(df: pd.DataFrame, object_option: bool = False) -> pd.DataFrame:
    """Reduce the size of the input dataframe
    Parameters
    ----------
    df: pd.DataFrame
        input DataFrame
    object_option : bool, default=False
        if true, try to convert object to category
    Returns
    -------
    df: pd.DataFrame
        data type optimized output dataframe
    """

    # loop columns in the dataframe to downcast the dtype
    for col in df.columns:
        # process the int columns
        if df[col].dtype == "int64":
            col_min = df[col].min()
            col_max = df[col].max()
            # if all are non-negative, change to uint
            if col_min >= 0:
                if col_max < np.iinfo(np.uint8).max:
                    df[col] = df[col].astype(np.uint8)
                elif col_max < np.iinfo(np.uint16).max:
                    df[col] = df[col].astype(np.uint16)
                elif col_max < np.iinfo(np.uint32).max:
                    df[col] = df[col].astype(np.uint32)
                else:
                    df[col] = df[col]
            else:
                # if it has negative values, downcast based on the min and max
                if col_max < np.iinfo(np.int8).max and\
                        col_min > np.iinfo(np.int8).min:
                    df[col] = df[col].astype(np.int8)
                elif (
                    col_max < np.iinfo(np.int16).max
                    and col_min > np.iinfo(np.int16).min
                ):
                    df[col] = df[col].astype(np.int16)
                elif (
                    col_max < np.iinfo(np.int32).max
                    and col_min > np.iinfo(np.int32).min
                ):
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col]

        # process the float columns
        elif df[col].dtype == "float":
            col_min = df[col].min()
            col_max = df[col].max()
            # downcast based on the min and max
            if (
                col_min > np.finfo(np.float32).min
                and col_max < np.finfo(np.float32).max
            ):
                df[col] = df[col].astype(np.float32)
            else:
                df[col] = df[col]

        if object_option:
            if df[col].dtype == "object":
                if len(df[col].value_counts()) < 0.5 * len(df):
                    df[col] = df[col].astype("category")

    return df
