import numpy as np 

def det(arr): 
    def count_det(arr, i, j):
        if arr.shape == (1,1):
            return arr[0]
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

def test_naive():
    for i in range(1, 10):
        random_matrix = np.random.randint(10, size=(i, i))
        our_impl =  det(random_matrix)
        numpy_impl = round(np.linalg.det(random_matrix))
        if our_impl != numpy_impl:
            print(f"{our_impl} != {numpy_impl} size: {i}")

            print(random_matrix)
        else:
            print("Guccis")


if __name__ == "__main__":
    test_naive()



