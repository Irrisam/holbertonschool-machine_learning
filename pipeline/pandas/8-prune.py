#!/usr/bin/env python3
"""
    Converts a numpy array to a pandas DataFrame
"""


def prune(df):
    """
    Removes any entries where Close has NaN values.

    Parameters:
    df (pd.DataFrame): Input DataFrame with a 'Close' column.

    Returns:
    pd.DataFrame: Modified DataFrame without NaN values in 'Close'.
    """
    return df.dropna(subset=['Close'])
