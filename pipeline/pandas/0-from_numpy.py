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

    return pd.DataFrame(array)