#!/usr/bin/env python3
"""
    Converts a numpy array to a pandas DataFrame
"""

import pandas as pd


def concat(df1,df2):
    """
    Indexes both dataframes on their Timestamp columns,
    selects relevant rows from df2,
    concatenates them to df1, and labels the sources.

    Parameters:
    df1 (pd.DataFrame): Coinbase DataFrame.
    df2 (pd.DataFrame): Bitstamp DataFrame.

    Returns:
    pd.DataFrame: Concatenated DataFrame with labeled sources.
    """
    index = __import__('10-index').index
    df1 = index(df1)
    df2 = index(df2)
    df2_filtered = df2.loc[df2.index <= 1417411920]
    
    return pd.concat({'bitstamp': df2_filtered, 'coinbase': df1}, names=['Source'])
