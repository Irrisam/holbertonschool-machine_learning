#!/usr/bin/env python3
"""
    Converts a numpy array to a pandas DataFrame
"""


def fill(df):
    """
    Removes the Weighted_Price column and
    fills missing values in the DataFrame.

    Parameters:
    df (pd.DataFrame): Input DataFrame with financial data.

    Returns:
    pd.DataFrame: Modified DataFrame with missing values handled.
    """
    df = df.drop(columns=['Weighted_Price'], errors='ignore')
    df['Close'] = df['Close'].fillna(method='ffill')
    for col in ['High', 'Low', 'Open']:
        df[col] = df[col].fillna(df['Close'])
    df[['Volume_(BTC)',
        'Volume_(Currency)']] = df[['Volume_(BTC)',
                                    'Volume_(Currency)']].fillna(0)
    return df
