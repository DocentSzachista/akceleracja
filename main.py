import os
import numpy as np 
from matplotlib import pyplot as plt
from time import perf_counter_ns

import python_impl
import numba_impl

MATRIX_MIN_VALUE = -10
MATRIX_MAX_VALUE = 10

def test_correctness(method, max_size):
    for i in range(max_size):
        matrix = np.random.randint(MATRIX_MIN_VALUE, MATRIX_MAX_VALUE, size=(i, i))
        np_res = round(np.linalg.det(matrix))
        our_res = round(method(matrix))
        assert np_res == our_res
        print(f'Run {i + 1}/{max_size} compleated')

def measure_time(method, max_size):
    times = []
    for i in range(max_size):
        matrix = np.random.randint(MATRIX_MIN_VALUE, MATRIX_MAX_VALUE, size=(i, i))
        start = perf_counter_ns()
        method(matrix)
        end = perf_counter_ns()
        times.append(end - start)
    return times

def make_graph(name, times, max_size):
    plt.cla()
    plt.title(f"Time to count a matrix determinant using '{name}'")
    plt.plot(range(1, max_size + 1), times)
    plt.xlabel("Size of the square matrix")
    plt.ylabel("Time [ns]")
    plt.grid()
    plt.savefig(f'plots/{name}.png')

def make_combined_graph(names, all_times, max_size):
    plt.cla()
    plt.title(f"Time to count a matrix determinant using different methods")
    for (times, name) in zip(all_times, names):
        plt.plot(range(1, max_size + 1), times, label=name)
    plt.xlabel("Size of the square matrix")
    plt.ylabel("Time [ns]")
    plt.grid()
    plt.legend()
    plt.savefig(f'plots/combined.png')

if __name__ == "__main__":
    if not os.path.exists('plots'):
        os.mkdir('plots')

    PERF_MAX_SIZE = 10
    TIME_MAX_SIZE = 10
    methods = [python_impl.det, numba_impl.det]
    names = ["Python", "Numba"]
    all_times = []
    for (method, name) in zip(methods, names):
        print(name, ":")
        test_correctness(method, PERF_MAX_SIZE)
        times = measure_time(method, TIME_MAX_SIZE)
        all_times.append(times)
        make_graph(name, times, TIME_MAX_SIZE)
    make_combined_graph(names, all_times, TIME_MAX_SIZE)

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




