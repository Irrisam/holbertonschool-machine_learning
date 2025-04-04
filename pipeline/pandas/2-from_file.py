#!/usr/bin/env python3
"""
    Converts a numpy array to a pandas DataFrame
"""

import pandas as pd


def from_file(filename, delimiter):
    """
        Converts a numpy array to a pandas DataFrame.

        :param filename: str
            The name of the file to convert.
        :param delimiter: str
            The delimiter used in the file.

        :return: pandas.DataFrame
            The converted DataFrame.
    """
    df = pd.read_csv(filename, delimiter=delimiter)
    return df
