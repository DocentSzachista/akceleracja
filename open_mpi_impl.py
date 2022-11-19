from python_impl import permutation_init, permutation_permute
from mpi4py import MPI 
import numpy as np 
from main import measure_time, dump_times


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
workers = comm.Get_size() - 1


def distribute_tasks(matrix, size):
    tasks = [] 
    p, v = permutation_init(size)
    tasks.append( (p, matrix) )

    id = 1
    mark = 1 
    while permutation_permute(size, p, v):
        comm.send( (p, matrix) , pid = id , tag = 1) 
        comm.send( (size, mark), pid = id, tag =  2)
        mark *= -1

def assemble_tasks():

    pid = 1
    det = 0
    for _ in range(workers): 
        det += comm.recv(source =pid, tag=pid )
        pid += 1 
    return det

def slave_labour():
    p, matrix = comm.recv(source=0, tag=1)
    size, sign = comm.recv(source=0, tag=2)
    prod = 1
    for i in range(size):
        prod *= matrix[i, p[i]]
    det_sum = prod * sign
    comm.send(det_sum, dest=0, tag=rank)

def det(matrix: np.array) -> int:
    size = matrix.shape[0]
    if size == 1:
        return matrix[0, 0]
    if size == 2:
        return matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]
    distribute_tasks(matrix, size )
    return assemble_tasks()




if __name__ == "__main__": 
    if rank == 0: 
        times = measure_time(det, 10)
        dump_times("open_mpi", times)
    else: 
        slave_labour()