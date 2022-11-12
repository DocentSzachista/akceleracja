import numpy as np

def permutation_init(size: int) -> tuple[list[int], list[bool]]:
    p = [x for x in range(size)]
    v = [False for x in range(size)]
    return p, v

def permuatation_position(size: int, i: int, p: list[int]) -> int:
    for j in range(size):
        if p[j] == i:
            return j
    return 0

def permutation_permute(size: int, p: list[int], v: list[bool]) -> bool:
    i = size
    while i:
        i -= 1
        position = permuatation_position(size, i, p)
        if position == 0 and not v[position]:
            continue
        if position == size - 1 and v[position]:
            continue
        if not v[position] and p[position - 1] > i:
            continue
        if v[position] and p[position + 1] > i:
            continue
        sp = position + 1 if v[position] else position - 1

        tmp = p[position]
        p[position] = p[sp]
        p[sp] = tmp

        tmp = v[position]
        v[position] = v[sp]
        v[sp] = tmp

        for j in range(i + 1, size):
            position = permuatation_position(size, j, p)
            v[position] = not v[position]
        return True
    return False

def det(matrix: np.array) -> int:
    size = matrix.shape[0]
    if size == 1:
        return matrix[0, 0]
    if size == 2:
        return matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]
    
    p, v = permutation_init(size)
    det_sum = 0
    sign = 1
    
    # for _ in range(math.factorial(size)):
    #     prod = 1
    #     for i in range(size):
    #         prod *= matrix[i, p[i]]
    #     det_sum += prod * sign
    #     sign = -sign
    #     permutation_permute(size, p, v)

    while True:
        prod = 1
        for i in range(size):
            prod *= matrix[i, p[i]]
        det_sum += prod * sign
        sign = -sign
        if not permutation_permute(size, p, v):
            break
    return det_sum