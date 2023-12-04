#!/usr/bin/env python3
"""Function that calculates the shape of a matrix"""


def matrix_shape(matrix):
    """Function that calculates the shape of a matrice"""
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        matrix = matrix[0]
    return(shape)
