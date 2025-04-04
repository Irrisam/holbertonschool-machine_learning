#!/usr/bin/env python3
"""
    Converts a numpy array to a pandas DataFrame
"""


def high(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sorts the DataFrame by the High price in descending order.

    Parameters:
    df (pd.DataFrame): Input DataFrame with a 'High' column.

    Returns:
    pd.DataFrame: Sorted DataFrame.
    """
    return df.sort_values(by='High', ascending=False)
