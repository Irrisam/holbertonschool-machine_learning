#!/usr/bin/env python3
"""function that returns operations on matrices"""


def np_elementwise(mat1, mat2):
    """function that returns operations on matrices"""
    if mat1 is not int or mat2 is not int:
        if len(mat1) != len(mat2):
            """checks for shape"""
            return None
        for row in mat2:
            if len(row) != len(mat1[0]):
                return None

        sum_matrix = [[] for _ in range(len(mat1))]
        diff_matrix = [[] for _ in range(len(mat1))]
        mul_matrix = [[] for _ in range(len(mat1))]
        div_matrix = [[] for _ in range(len(mat1))]

        for counter, (i, j) in enumerate(zip(mat1, mat2)):
            """adds a counter in order to insert new values in new matrice"""
            for x, y in zip(i, j):
                """digs within each new tuple to get new values"""
                sum_matrix[counter].append(x + y)
                diff_matrix[counter].append(x - y)
                mul_matrix[counter].append(x * y)
                div_matrix[counter].append(x / y)
        return sum_matrix, diff_matrix, mul_matrix, div_matrix
