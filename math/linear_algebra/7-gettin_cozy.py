#!/usr/bin/env python3
"""function that concatenates along specified axis"""


def cat_matrices2D(mat1, mat2, axis=0):
    """function that concatenates along specified axis"""
    if len(mat1) != len(mat2):
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
