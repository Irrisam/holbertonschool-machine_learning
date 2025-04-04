#!/usr/bin/env python3
"""
    Converts a numpy array to a pandas DataFrame
"""


def analyze(df):
    """
        Analyzes the DataFrame and returns descriptive statistics.
        :param df: pandas.DataFrame
            The DataFrame to analyze.
        :return: pandas.DataFrame
            The DataFrame with descriptive statistics.
    """
    if 'Timestamp' in df.columns:
        df = df.drop(columns=['Timestamp'])

    descriptive_stats = df.describe()

    return descriptive_stats
