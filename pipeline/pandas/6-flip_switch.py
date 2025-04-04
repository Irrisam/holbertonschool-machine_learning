#!/usr/bin/env python3
"""
    Converts a numpy array to a pandas DataFrame
"""


def flip_switch(df):
    """
        Converts a numpy array to a pandas DataFrame.

        :param df: pandas.DataFrame
            The DataFrame to convert.

        :return: pandas.DataFrame
            The converted DataFrame.
    """
    if df.index.inferred_type == 'datetime':
        df_sorted = df.sort_index()
    else:
        df_sorted = df.sort_values(by=df.columns[0])
    return df_sorted.T
