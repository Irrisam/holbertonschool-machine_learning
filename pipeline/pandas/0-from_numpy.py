import pandas as pd
import numpy as np


def from_numpy(array):
    """
        Converts a numpy array to a pandas DataFrame.

        :param array: numpy.ndarray
            The numpy array to convert.

        :return: pandas.DataFrame
            The converted DataFrame.
    """

    return pd.DataFrame(array)