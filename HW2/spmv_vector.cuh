#ifndef SPMV_CUH
#define SPMV_CUH

#include "common.h"
#include "CSR.hpp"


namespace spmv {

/**
 * @brief CUDA kernel for performing sparse matrix-vector multiplication (SpMV) using the CSR format.
 *
 * This kernel computes the product of a sparse matrix \( A \) (stored in compressed sparse row (CSR) format) 
 * and a dense vector \( x \), storing the result in the output vector \( y \).
 *
 * @param m        Number of rows in the sparse matrix.
 * @param n        Number of columns in the sparse matrix.
 * @param d_vals   Device pointer to the nonzero values of the sparse matrix (CSR format).
 * @param d_colinds Device pointer to the column indices corresponding to the nonzero values (CSR format).
 * @param d_rowptrs Device pointer to the row pointers defining the matrix structure (CSR format).
 * @param d_x      Device pointer to the input dense vector.
 * @param d_y      Device pointer to the output dense vector.
 */
__global__ void SpMV(const size_t m, const size_t n,
                    float * d_vals, uint32_t * d_colinds, uint32_t * d_rowptrs,
                    __constant__ float * d_x, float * d_y)
{
    int t = threadIdx.x;                         // Thread ID in block
    int lane = t & (warpSize - 1);               // Thread index within the warp (0-31)
    int warpsPerBlock = blockDim.x / warpSize;   // Number of warps in this block
    int row = (blockIdx.x * warpsPerBlock) + (t / warpSize);  // One warp per row

    __shared__ volatile float vals[1024]; // Shared memory buffer (assuming blockDim.x = 1024)

    if (row < m) {
        int rowStart = d_rowptrs[row];
        int rowEnd = d_rowptrs[row + 1];
        float sum = 0.0f;

        // Each thread in the warp processes a subset of the row's non-zeros
        for (int j = rowStart + lane; j < rowEnd; j += warpSize) {
            int col = d_colinds[j];
            sum += d_vals[j] * d_x[col];
        }

        vals[t] = sum;

        // Warp-level reduction (manual unrolling)
        if (lane < 16) vals[t] += vals[t + 16];
        if (lane < 8)  vals[t] += vals[t + 8];
        if (lane < 4)  vals[t] += vals[t + 4];
        if (lane < 2)  vals[t] += vals[t + 2];
        if (lane < 1)  vals[t] += vals[t + 1];

        // Only the first thread in the warp writes the final result
        if (lane == 0) {
            d_y[row] = vals[t];
        }
    }
}

/**
 * @brief Launches a CUDA kernel to perform Sparse Matrix-Vector Multiplication (SpMV).
 * 
 * This function wraps the SpMV kernel, configuring the execution parameters 
 * and ensuring the computation completes before returning. The sparse matrix 
 * is stored in Compressed Sparse Row (CSR) format.
 * Feel free to change this function however you'd like. 
 * Just ensure that the `d_y` array stores the correct output vector
 * when the function terminates.
 * 
 * @param A The sparse matrix in CSR format.
 * @param d_x Device pointer to the input vector.
 * @param d_y Device pointer to the output vector, where the result is stored.
 */
void SpMV_wrapper(CSR<float, uint32_t>& A, float * d_x, float * d_y)
{
    //**** CHANGE THESE VALUES ****//
    uint32_t threads_per_block = 1;
    uint32_t blocks = 1; 

    // Call the kernel
    SpMV<<<blocks, threads_per_block>>>(A.get_rows(), A.get_cols(),
                   A.get_vals(), A.get_colinds(), A.get_rowptrs(),
                   d_x, d_y);

    // Sync w/ the host
    CUDA_CHECK(cudaDeviceSynchronize());
}

}
#endif
