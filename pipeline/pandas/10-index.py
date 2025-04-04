#!/usr/bin/env python3
"""
    Converts a numpy array to a pandas DataFrame
"""


def index(df):
    """
    Sets the Timestamp column as the index of the DataFrame.

    Parameters:
    df (pd.DataFrame): Input DataFrame with a 'Timestamp' column.

    Returns:
    pd.DataFrame: Modified DataFrame with 'Timestamp' set as index.
    """
    return df.set_index('Timestamp')
