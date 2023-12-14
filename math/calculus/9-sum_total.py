#!/usr/bin/env python3
"""function that calculates sigma with n as interative"""


def summation_i_squared(n):
    """function that calculates sigma with n as interative"""
    if type(n) is not int or n < 1:
        return None
    """apply natural number formula"""
    sum = (n * (n + 1) * (2 * n + 1)) // 6
    return sum
