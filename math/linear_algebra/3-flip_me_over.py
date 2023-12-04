#!/usr/bin/env python3
"""function that transposes a matrix"""
# import numpy


# def matrix_transpose(matrix):
#    numpy_matrix = numpy.array(matrix)
#    transposed_matrix = numpy.transpose(numpy_matrix)#
#   return transposed_matrix.tolist()

def matrix_transpose(matrix):
    """function that transposes a matrix"""
    transposed_matrix = []
    """save the size of the matrix (has to not change)"""
    cols = len(matrix[0])
    rows = len(matrix)
    for i in range(cols):
        """set up for the number of top layer(nmb of columns)"""
        new_rows = []
        for j in range(rows):
            """copies and add the transposed rows into the new matrixe"""
            new_rows.append(matrix[j][i])
            """reversing matrix by using j first"""
        transposed_matrix.append(new_rows)
    return(transposed_matrix)
