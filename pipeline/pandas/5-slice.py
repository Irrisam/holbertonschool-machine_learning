#!/usr/bin/env python3
"""
    Converts a numpy array to a pandas DataFrame
"""


def slice(df):
    """
        Slices a DataFrame to keep only the columns
        'High', 'Low', 'Close', and 'Volume_(BTC)'.
        It also takes every 60th row of the DataFrame.

        :param df: pandas.DataFrame
            The DataFrame to slice.

        :return: pandas.DataFrame
            The sliced DataFrame.
    """
    df = df[['High', 'Low', 'Close', 'Volume_(BTC)']].iloc[:: 60]
    return (df)
