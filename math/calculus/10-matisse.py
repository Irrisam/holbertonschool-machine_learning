#!/usr/bin/env python3
"""function that returns derivatives list"""


def poly_derivative(poly):
    """function that returns derivatives list"""
    if not isinstance(poly, list):
        return None

    lenpoly = len(poly)

    if lenpoly == 0:
        return None

    if lenpoly == 1:
        return [0]

    derivative = [0] * (lenpoly - 1)

    for i in range(1, lenpoly):
        derivative[i - 1] = poly[i] * i

    return derivative
