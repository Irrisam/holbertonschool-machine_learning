#!/usr/bin/env python3
"""
    Converts a numpy array to a pandas DataFrame
"""

import pandas as pd


def rename(df):
    """
        Renames the columns of a DataFrame.

        :param df: pandas.DataFrame
            The DataFrame to rename.

        :return: pandas.DataFrame
            The renamed DataFrame.
    """
    df.rename(columns={'Timestamp': 'Datetime'}, inplace=True)

    df['Datetime'] = pd.to_datetime(df['Datetime'], unit='s')

    df = df[['Datetime', 'Close']]

    return df
