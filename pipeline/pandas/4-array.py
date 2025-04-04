#!/usr/bin/env python3
"""
    Converts a numpy array to a pandas DataFrame
"""


def array(df):
    """
        Converts a numpy array to a pandas DataFrame.

        :param df: pandas.DataFrame
            The DataFrame to convert.

        :return: numpy.ndarray
            The converted numpy array.
    """

    arrayed_tails = df[['High', 'Close']].tail(10)

    arrayed_tails = arrayed_tails.to_numpy()
    return (arrayed_tails)
