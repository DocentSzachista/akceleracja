import numpy as np
import ray
ray.init(address='auto')

from python_impl import permutation_init, permutation_permute


def det(matrix: np.array) -> int:
    size = matrix.shape[0]
    if size == 1:
        return matrix[0, 0]
    if size == 2:
        return matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]
    
    p, v = permutation_init(size)
    sign = 1
    futures = []

    while True:
        futures.append(
            compute_diagonal.remote(
                matrix,
                p,
                sign 
            )
        )
        sign = -sign
        if not permutation_permute(size, p, v):
            break
    return sum(ray.get(futures))

@ray.remote
def compute_diagonal(matrix: np.ndarray, p: list[int], sign: int) -> int:
    prod = 1
    for i in range(matrix.shape[0]):
        prod *= matrix[i, p[i]]
    return prod * sign 