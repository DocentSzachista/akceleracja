# from cProfile import label
import numpy as np 
# from matplotlib import pyplot as plt
from time import perf_counter_ns
def det(arr): 
    def count_det(arr, i, j):
        if arr.shape == (3,3): 
            return (-1) **( i+j ) * ( 
                  arr[0][0] * arr[1][1] * arr[2][2]  
                + arr[0][1] * arr[1][2] * arr[2][0]
                + arr[0][2] * arr[1][0] * arr[2][1]
                - arr[0][2] * arr[1][1] * arr[2][0]
                - arr[0][0] * arr[1][2] * arr[2][1]
                - arr[0][1] * arr[1][0] * arr[2][2]
            )
        cut_matrix = np.delete(arr, 0, 0)
        det = 0 
        for elem in range(arr.shape[1]):
            det += (-1) **(i+j) * arr[0][elem] * count_det(np.delete(cut_matrix, elem, 1), 0, elem) 
        return det    
    
    if arr.shape == (1, 1):
        return arr[0][0]
    if arr.shape == (2, 2):
        return arr[0][0] * arr[1][1] - arr[1][0] * arr[0][1]
    return count_det(arr, 0, 0)

def det_iterative(arr): 
    pass

# def det_iterative(arr):

#     rows, columns = arr.shape
#     det_positive_part = 0 
    
#     for column_index in range(columns):
#         mul_op = 1 
#         for row_index in range(rows):
#             dupa_zmienna = column_index+row_index
#             mul_op *= arr[row_index][dupa_zmienna if dupa_zmienna  < columns else columns - dupa_zmienna] 
#         det_positive_part += mul_op
            
def divide_matrix(arr: np.array):
    assert arr.shape[0] > 2, "This method only works for 3x3 matrices and greater"
    matrices = []
    values = []

    def divide_inner(matrix: np.array, i: int, j: int, current_value: int):
        if matrix.shape == (3, 3):
            matrices.append(matrix)
            values.append(current_value)
            return

        cut_matrix = np.delete(arr, 0, 0)
        for elem in range(matrix.shape[1]):
            divide_inner(np.delete(cut_matrix, elem, 1), 0, elem, current_value * ((-1)**(i+j) * matrix[0, elem]))

    divide_inner(arr, 0, 0, 1)
    return (matrices, values)


def test_divide():
    SIZE = 4
    m, v = divide_matrix(np.random.randint(10, size=(SIZE, SIZE)))
    print(m)
    print(v)



def test_det():
    for i in range(1, 11):
        random_matrix = np.random.randint(10, size=(i, i))
        our_impl =  det(random_matrix)
        numpy_impl = round(np.linalg.det(random_matrix))
        if our_impl != numpy_impl:
            print(f"{our_impl} != {numpy_impl} size: {i}")

            print(random_matrix)
        else:
            print("Guccis")

def measure_time(det_func, size)->list:
    scores = []
    for i in range(1, size):
        random_matrix = np.random.randint(10, size=(i, i))
        start = perf_counter_ns()
        det_func(random_matrix)
        end = perf_counter_ns()
        score = end - start 
        scores.append(score)
    return scores



if __name__ == "__main__":
    test_divide()
    # test_det()
    # our_det_scores = measure_time(det, 11)
    # numpy_scores = measure_time(np.linalg.det, 11)

    # print(numpy_scores)
    # print(our_det_scores)
    # plt.title("Time of counting determinant according to matrix size ")
    # plt.plot(range(1 ,11), our_det_scores, label="our implementation" )
    # plt.plot(range(1 ,11), numpy_scores, label="numpy det")
    # plt.xlabel("Size XnX of matrix")
    # plt.ylabel("Time passed to count determinant (ns)")
    # plt.grid()
    # plt.legend()
    # plt.savefig("example.png")




