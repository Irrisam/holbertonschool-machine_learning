#!/usr/bin/env python3
"""function that add two arrays element wise"""


def add_arrays(arr1, arr2):
    """checks for different sizes"""
    if len(arr1) != len(arr2):
        return None

    transformed_matrix = []
    for i, j in zip(arr1, arr2):
        """zip builds tupples with each element and give them to i and j"""
        result = i + j
        transformed_matrix.append(result)

    return transformed_matrix
