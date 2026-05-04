#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <stdexcept>
#include <vector>
#include <thread>
#include <algorithm>

namespace py = pybind11;

// Naive triple loop matrix multiplication
py::array_t<float> matmul_naive(py::array_t<float> A, py::array_t<float> B) {
    py::buffer_info bufA = A.request();
    py::buffer_info bufB = B.request();

    if (bufA.ndim != 2 || bufB.ndim != 2) {
        throw std::runtime_error("Number of dimensions must be two");
    }

    if (bufA.shape[1] != bufB.shape[0]) {
        throw std::runtime_error("Matrix inner dimensions must agree");
    }

    size_t M = bufA.shape[0];
    size_t K = bufA.shape[1];
    size_t N = bufB.shape[1];

    auto result = py::array_t<float>({M, N});
    py::buffer_info bufC = result.request();

    float *ptrA = static_cast<float *>(bufA.ptr);
    float *ptrB = static_cast<float *>(bufB.ptr);
    float *ptrC = static_cast<float *>(bufC.ptr);

    // Initialize C with zeros
    for (size_t i = 0; i < M * N; i++) {
        ptrC[i] = 0.0f;
    }

    // Naive triple loop
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; k++) {
                sum += ptrA[i * K + k] * ptrB[k * N + j];
            }
            ptrC[i * N + j] = sum;
        }
    }

    return result;
}

// Tiled matrix multiplication (single-threaded)
py::array_t<float> matmul_tiled(py::array_t<float> A, py::array_t<float> B) {
    py::buffer_info bufA = A.request();
    py::buffer_info bufB = B.request();

    if (bufA.ndim != 2 || bufB.ndim != 2) {
        throw std::runtime_error("Number of dimensions must be two");
    }

    if (bufA.shape[1] != bufB.shape[0]) {
        throw std::runtime_error("Matrix inner dimensions must agree");
    }

    size_t M = bufA.shape[0];
    size_t K = bufA.shape[1];
    size_t N = bufB.shape[1];

    auto result = py::array_t<float>({M, N});
    py::buffer_info bufC = result.request();

    float *ptrA = static_cast<float *>(bufA.ptr);
    float *ptrB = static_cast<float *>(bufB.ptr);
    float *ptrC = static_cast<float *>(bufC.ptr);

    // Initialize C with zeros
    std::fill(ptrC, ptrC + (M * N), 0.0f);

    // Block size for cache tiling
    const size_t BLOCK_SIZE = 64; 

    // Tiled block traversal
    for (size_t bi = 0; bi < M; bi += BLOCK_SIZE) {
        for (size_t bk = 0; bk < K; bk += BLOCK_SIZE) {
            for (size_t bj = 0; bj < N; bj += BLOCK_SIZE) {
                
                // Boundaries for this block
                size_t i_end = std::min(bi + BLOCK_SIZE, M);
                size_t k_end = std::min(bk + BLOCK_SIZE, K);
                size_t j_end = std::min(bj + BLOCK_SIZE, N);

                // Compute block
                for (size_t i = bi; i < i_end; i++) {
                    for (size_t k = bk; k < k_end; k++) {
                        // Extract invariant out of the inner loop
                        float a_ik = ptrA[i * K + k];
                        for (size_t j = bj; j < j_end; j++) {
                            ptrC[i * N + j] += a_ik * ptrB[k * N + j];
                        }
                    }
                }
            }
        }
    }

    return result;
}

// Tiled and Multi-threaded matrix multiplication
py::array_t<float> matmul_optimized(py::array_t<float> A, py::array_t<float> B) {
    py::buffer_info bufA = A.request();
    py::buffer_info bufB = B.request();

    if (bufA.ndim != 2 || bufB.ndim != 2) {
        throw std::runtime_error("Number of dimensions must be two");
    }

    if (bufA.shape[1] != bufB.shape[0]) {
        throw std::runtime_error("Matrix inner dimensions must agree");
    }

    size_t M = bufA.shape[0];
    size_t K = bufA.shape[1];
    size_t N = bufB.shape[1];

    auto result = py::array_t<float>({M, N});
    py::buffer_info bufC = result.request();

    float *ptrA = static_cast<float *>(bufA.ptr);
    float *ptrB = static_cast<float *>(bufB.ptr);
    float *ptrC = static_cast<float *>(bufC.ptr);

    // Initialize C with zeros
    std::fill(ptrC, ptrC + (M * N), 0.0f);

    // Block size for cache tiling
    const size_t BLOCK_SIZE = 64; 
    
    // Automatically detect processor core count
    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4; // fallback

    // Worker function for a thread to process a subset of rows
    auto worker = [&](size_t start_row, size_t end_row) {
        // Tiled block traversal
        for (size_t bi = start_row; bi < end_row; bi += BLOCK_SIZE) {
            for (size_t bk = 0; bk < K; bk += BLOCK_SIZE) {
                for (size_t bj = 0; bj < N; bj += BLOCK_SIZE) {
                    
                    // Boundaries for this block
                    size_t i_end = std::min(bi + BLOCK_SIZE, end_row);
                    size_t k_end = std::min(bk + BLOCK_SIZE, K);
                    size_t j_end = std::min(bj + BLOCK_SIZE, N);

                    // Compute block
                    for (size_t i = bi; i < i_end; i++) {
                        for (size_t k = bk; k < k_end; k++) {
                            // Extract invariant out of the inner loop
                            float a_ik = ptrA[i * K + k];
                            for (size_t j = bj; j < j_end; j++) {
                                ptrC[i * N + j] += a_ik * ptrB[k * N + j];
                            }
                        }
                    }
                }
            }
        }
    };

    // Spin up threads
    std::vector<std::thread> threads;
    size_t rows_per_thread = (M + num_threads - 1) / num_threads;

    for (unsigned int t = 0; t < num_threads; ++t) {
        size_t start_row = t * rows_per_thread;
        size_t end_row = std::min(start_row + rows_per_thread, M);
        if (start_row < M) {
            threads.emplace_back(worker, start_row, end_row);
        }
    }

    // Wait for all threads to finish
    for (auto &th : threads) {
        th.join();
    }

    return result;
}

PYBIND11_MODULE(naive_cpp_matmul, m) {
    m.doc() = "C++ matrix multiplication module"; 
    m.def("matmul_naive", &matmul_naive, "A naive function that multiplies two matrices");
    m.def("matmul_tiled", &matmul_tiled, "A function that multiplies two matrices using tiling (single-threaded)");
    m.def("matmul_optimized", &matmul_optimized, "A function that multiplies two matrices using threading and tiling");
    m.def("matmul", &matmul_optimized, "Alias for matmul_optimized so existing code still works");
}
