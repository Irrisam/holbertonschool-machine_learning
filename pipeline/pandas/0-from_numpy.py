#!/usr/bin/env python3
"""
    Converts a numpy array to a pandas DataFrame
"""
import pandas as pd


def from_numpy(array):
    """
        Converts a numpy array to a pandas DataFrame.

        :param array: numpy.ndarray
            The numpy array to convert.

        :return: pandas.DataFrame
            The converted DataFrame.
    """
    df = pd.DataFrame(array)
    num_col = array.shape[1]
    letters = []
    for i in range(num_col):
        letters.append(chr(ord('A') + i))
    df = pd.DataFrame(array, columns=letters)
    return df
