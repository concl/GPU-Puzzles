
from numba.cuda import jit
from numba import cuda
import numba
import numpy as np
import time

@jit
def add_arrays(a, b, c):
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if i < a.size:
        c[i] = a[i] + b[i]
        
def add_arrays_numpy(a, b):
    return a + b

TPB = 32

@jit
def matrix_multiply(a, b, out):
    
    size_a_rows, size_a_cols = a.shape
    size_b_rows, size_b_cols = b.shape
    
    a_shared = cuda.shared.array((TPB, TPB), numba.float32)
    b_shared = cuda.shared.array((TPB, TPB), numba.float32)
    out_shared = cuda.shared.array((TPB, TPB), numba.float32)
    
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    
    local_i = cuda.threadIdx.x
    local_j = cuda.threadIdx.y

    out_shared[local_i, local_j] = 0.0
    for curr_block in range(min((size_a_cols + TPB - 1) // TPB, (size_b_rows + TPB - 1) // TPB)):
        if i < size_a_rows and (j % TPB + curr_block * TPB) < size_b_cols:
            a_shared[local_i, local_j] = a[i, j % TPB + curr_block * TPB]
        else:
            a_shared[local_i, local_j] = 0
            
        if j < size_b_rows and (i % TPB + curr_block * TPB) < size_a_cols:
            b_shared[local_i, local_j] = b[i % TPB + curr_block * TPB, j]
        else:
            b_shared[local_i, local_j] = 0
        cuda.syncthreads()
        
        if i < size_a_rows and j < size_b_cols:
            for k in range(TPB):
                out_shared[local_i, local_j] += a_shared[local_i, k] * b_shared[k, local_j]
        
        cuda.syncthreads()
    
    if i < size_a_rows and j < size_b_cols:
        out[i, j] = out_shared[local_i, local_j]
    
    
def matmul(a, b, out):
    size_a_rows, size_a_cols = a.shape
    size_b_rows, size_b_cols = b.shape
    
    threads_per_block = (TPB, TPB)
    blocks_per_grid_x = (size_a_rows + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (size_b_cols + threads_per_block[1] - 1) // threads_per_block[1]
    
    matrix_multiply[(blocks_per_grid_x, blocks_per_grid_y), threads_per_block](a, b, out)
        
def test_add_arrays():            
    print("Testing add_arrays...")
    
    n = 10**8
    a = np.random.rand(n).astype(np.float32)
    b = np.random.rand(n).astype(np.float32)
    c = np.empty_like(a)

    
    with cuda.gpus[0]:
        threads_per_block = 1024
        blocks = n // threads_per_block + 1
        
        # warmup computation
        d_a = cuda.to_device(a)
        d_b = cuda.to_device(b)
        d_c = cuda.to_device(c)

        add_arrays[blocks, threads_per_block](d_a, d_b, d_c)
        cuda.synchronize()
        
        # main computation
        d_a = cuda.to_device(a)
        d_b = cuda.to_device(b)
        d_c = cuda.to_device(c)
        
        start_time = time.perf_counter()
        add_arrays[blocks, threads_per_block](d_a, d_b, d_c)
        cuda.synchronize()
        end_time = time.perf_counter()
        
        c = d_c.copy_to_host()
    
        print(f"Numba CUDA kernel time: {end_time - start_time}s")
    
    
    start_time = time.perf_counter()
    result_numpy = add_arrays_numpy(a, b)
    end_time = time.perf_counter()
    
    print(f"Numpy time: {end_time - start_time}s")
    
    assert np.allclose(c, result_numpy), "Results do not match!"
    
def test_matrix_multiply():
    print("Testing matrix_multiply...")
    
    size_a_rows, size_a_cols = 1024, 1024
    size_b_rows, size_b_cols = 1024, 1024
    
    a = np.random.rand(size_a_rows, size_a_cols).astype(np.float32)
    b = np.random.rand(size_b_rows, size_b_cols).astype(np.float32)
    
    with cuda.gpus[0]:
        # warmup computation
        d_a = cuda.to_device(a)
        d_b = cuda.to_device(b)
        d_out = cuda.device_array((size_a_rows, size_b_cols), dtype=np.float32)
        matmul(d_a, d_b, d_out)
        cuda.synchronize()
        
        # main computation
        start_time = time.perf_counter()
        matmul(d_a, d_b, d_out)
        cuda.synchronize()
        end_time = time.perf_counter()
        
        print(f"Numba CUDA matrix multiplication time: {end_time - start_time}s")
        out_gpu = d_out.copy_to_host()
    
    start_time = time.perf_counter()
    out_numpy = np.dot(a, b)
    end_time = time.perf_counter()
    
    print(f"Numpy matrix multiplication time: {end_time - start_time}s")
    
    assert np.allclose(out_gpu, out_numpy), "Results do not match!"

def main():
    
    test_add_arrays()
    test_matrix_multiply()

    
if __name__ == "__main__":
    main()
