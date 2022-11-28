import numpy as np
import pyopencl as cl
import math

from python_impl import permutation_init, permutation_permute

KERNEL = """
__kernel void det(__global const long* matrix, __global const long* all_p, 
                  __global long* result, __global long* s)
{
    long size = s[0];

    long gid = get_global_id(0);
    const long* p = all_p[size * gid];
    long prod = 1;
    long sign = gid % 2 == 0 ? 1 : -1;
    for(long i = 0; i < size; i++) {
        prod *= matrix[(size * i) + p[i]];
    }
    result[gid] = prod * sign;
}
"""

def det(matrix: np.array) -> int:
    size = matrix.shape[0]
    if size == 1:
        return matrix[0, 0]
    if size == 2:
        return matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]
    
    matrix = matrix.astype(np.int64)
    
    p, v = permutation_init(size)
    
    result_size = 0
    all_p = np.empty(0, dtype=np.int64)

    while True:
        result_size += 1
        all_p = np.append(all_p, p)
        if not permutation_permute(size, p, v):
            break
    
    result_cpu = np.zeros(result_size, dtype=np.int64)
    
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    mf = cl.mem_flags
    matrix_gpu = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=matrix.flatten())
    all_p_gpu = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=all_p)
    size_gpu = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.array([size], dtype=np.int64))
    result_gpu = cl.Buffer(ctx, mf.WRITE_ONLY, result_cpu.nbytes)

    program = cl.Program(ctx, KERNEL).build()
    knl = program.det
    knl(queue, [size], None, matrix_gpu, all_p_gpu, result_gpu, size_gpu)

    cl.enqueue_copy(queue, result_cpu, result_gpu)

    return sum(result_cpu)
