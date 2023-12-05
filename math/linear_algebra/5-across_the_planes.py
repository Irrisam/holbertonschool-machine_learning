#!/usr/bin/env python3
"""function that adds two matrices element wise"""


def add_matrices2D(mat1, mat2):
    if len(mat1) != len(mat2):
        """checks for shape"""
        return None
    transformed_matrix = [[] for _ in range(len(mat1))]
    """builds the receiving matrice with the size of the original matrices"""
    for counter, (i, j) in enumerate(zip(mat1, mat2)):
        """adds a counter in order to insert new values in new matrice"""
        for x, y in zip(i, j):
            """digs within each new tuple to get new values"""
            result = x + y
            transformed_matrix[counter].append(result)

    return transformed_matrix
