from numba import cuda
import numpy as np
from time import perf_counter

@cuda.jit
def add_kernel(a, b, c):
    idx = cuda.grid(1)
    # Use a.shape[0] directly – no extra parameter
    if idx < a.shape[0]:
        c[idx] = a[idx] + b[idx]





def time_cuda_kernel():
    # Host code
    n = 1000000000
    a = np.arange(n, dtype=np.float32)
    b = np.arange(n, dtype=np.float32)
    c = np.zeros_like(a)

    threads = 256
    blocks = (n + threads - 1) // threads
    
    time_start = perf_counter()
    add_kernel[blocks, threads](a, b, c)
    cuda.synchronize()  # Ensure kernel execution is complete
    time_end = perf_counter()
    return time_end - time_start

def time_numpy():
    n = 1000000000
    a = np.arange(n, dtype=np.float32)
    b = np.arange(n, dtype=np.float32)
    
    time_start = perf_counter()
    c = a + b
    time_end = perf_counter()
    return time_end - time_start


def main():
    cuda_time = time_cuda_kernel()
    numpy_time = time_numpy()
    
    print(f"CUDA kernel execution time: {cuda_time:.6f} seconds")
    print(f"NumPy execution time: {numpy_time:.6f} seconds")
    
if __name__ == "__main__":
    main()