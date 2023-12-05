#!/usr/bin/env python3
"""function that concatenates along specified axis"""


def cat_matrices2D(mat1, mat2, axis=0):
    """function that concatenates along specified axis"""
    if axis == 0 and len(mat1[0]) != len(mat2[0]):
        """checks for same number of columns"""
        return None
    elif axis == 1 and len(mat1) != len(mat2):
        """checks for same number of rows"""
        return None
    result = []
    if axis == 0:
        """concatenates row, aka vertical axis"""
        result = mat1 + mat2
    else:
        """concatenates columns aka honrizontal axis"""
        for i in range(len(mat1)):
            result.append(mat1[i] + mat2[i])
    return result
