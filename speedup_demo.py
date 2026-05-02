
from numba.cuda import jit
from numba import cuda
import numpy as np
import time

@jit
def add_arrays(a, b, c):
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if i < a.size:
        c[i] = a[i] + b[i]
        
def add_arrays_numpy(a, b):
    return a + b

def main():
    
    n = 10**9
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
    
if __name__ == "__main__":
    main()
