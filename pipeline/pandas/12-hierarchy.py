#!/usr/bin/env python3
"""
    Converts a numpy array to a pandas DataFrame
"""

import pandas as pd

def hierarchy(df1, df2):
    """
    Rearranges the MultiIndex so that Timestamp
    is the first level.
    Concatenates the bitstamp and coinbase
    tables from timestamps 1417411980 to 1417417980, inclusive.
    Adds keys to the data, labeling rows
    from df2 as bitstamp and rows from df1 as coinbase.
    Ensures the data is displayed in chronological order.
    
    Parameters:
    df1 (pd.DataFrame): Coinbase DataFrame.
    df2 (pd.DataFrame): Bitstamp DataFrame.
    
    Returns:
    pd.DataFrame: Concatenated DataFrame with labeled sources, sorted chronologically.
    """
    
    index = __import__('10-index').index
    df1 = index(df1)
    df2 = index(df2)
    df1_filtered = df1.loc[(df1.index >= 1417411980) & (df1.index <= 1417417980)]
    df2_filtered = df2.loc[(df2.index >= 1417411980) & (df2.index <= 1417417980)]
    
    df_combined = pd.concat({'bitstamp': df2_filtered, 'coinbase': df1_filtered}, names=['Source'])
    return df_combined.sort_index()
