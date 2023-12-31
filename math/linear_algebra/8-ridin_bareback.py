def mat_mul(mat1, mat2):
    # Check if the matrices can be multiplied
    if len(mat1[0]) != len(mat2):
        return None

    # Initialize the result matrix with zeros
    result = []
    for _ in range(len(mat1)):
        result.append([0] * len(mat2[0]))

    # Perform matrix multiplication
    for i in range(len(mat1)):
        for j in range(len(mat2[0])):
            for k in range(len(mat2)):
                result[i][j] += mat1[i][k] * mat2[k][j]

    return result
