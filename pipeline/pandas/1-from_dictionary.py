#!/usr/bin/env python3
"""
    Converts a numpy array to a pandas DataFrame
"""
import pandas as pd


def dict_df_builder():
    """
        Converts a dictionary to a pandas DataFrame.

        :return: pandas.DataFrame
            The converted DataFrame.
    """
    dict_df = pd.DataFrame({'First': [0.0, 0.5, 1.0, 1.5], 'Second': ["one", "two", "three", "four"]}, index=["A", "B", "C", "D"])

    return dict_df

df = dict_df_builder()
