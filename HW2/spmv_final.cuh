
// READ THIS FOR PROCESS OF FIXES/DEBUGGING

// #ifndef SPMV_CUH
// #define SPMV_CUH

// #include "common.h"

// #include "CSR.hpp"

// // vectorized implementation
// namespace spmv {

// /**
//  * @brief CUDA kernel for performing sparse matrix-vector multiplication (SpMV) using the CSR format.
//  *
//  * This kernel computes the product of a sparse matrix \( A \) (stored in compressed sparse row (CSR) format) 
//  * and a dense vector \( x \), storing the result in the output vector \( y \).
//  *
//  * @param m        Number of rows in the sparse matrix.
//  * @param n        Number of columns in the sparse matrix.
//  * @param d_vals   Device pointer to the nonzero values of the sparse matrix (CSR format).
//  * @param d_colinds Device pointer to the column indices corresponding to the nonzero values (CSR format).
//  * @param d_rowptrs Device pointer to the row pointers defining the matrix structure (CSR format).
//  * @param d_x      Device pointer to the input dense vector.
//  * @param d_y      Device pointer to the output dense vector.
//  */
// __global__ void SpMV(const size_t m, const size_t n,
//                     float * d_vals, uint32_t * d_colinds, uint32_t * d_rowptrs,
//                     const float * d_x, float * d_y)
// {
//     int t = threadIdx.x;                         // Thread ID in block
//     int lane = t & (warpSize - 1);               // Thread index within the warp (0-31)
//     int warpsPerBlock = blockDim.x / warpSize;   // Number of warps in this block

//     // FIX 3: Here, we assigned one warp (32 threads) to compute ONE matrix row
//     int warp_id = t / warpSize;
//     int row = (blockIdx.x * warpsPerBlock) + warp_id;  // One warp per row

//     // __shared__ volatile float vals[1024]; // Shared memory buffer (assuming blockDim.x = 1024)
    
//     // FIX 1: We need 2D shared memory buffer since we're using more threads and warps. 
//     // a possible reason why we had cases where we don't pass the correctness test everytime before
//     // -> is because we had one shared memory buffer for all warps to write into -> causes data race, instability 
//     // We make this change so that, each warp gets its own row in shared memory, Each thread writes to vals[warp_id][lane]
//     // Theoretically, now there should be no interference between warps
//     __shared__ float vals[8][32]; // 8 warps * 32 threads, Change based on the num of warps and threads

//     if (row < m) {
//         int rowStart = d_rowptrs[row];
//         int rowEnd = d_rowptrs[row + 1];

//         // initialize sum
//         float sum = 0.0f;

//         // Each thread in the warp processes a subset of the row's non-zeros
//         for (int j = rowStart + lane; j < rowEnd; j += warpSize) {
//             int col = d_colinds[j];
//             sum += d_vals[j] * d_x[col];
//         }

        

//         // FIX 1: Accordingly change vals since now its 2D
//         // vals[t] = sum;
//         vals[warp_id][lane] = sum;

//         // FIX 2: no need to do this, because only threads in the warp are reducing for the following if statements
//         // but we're calling __syncthreads(), which is a block-wide sync.
//         // __syncthreads(); 
        
//         // FIX 2: Reduce within the warp 
//         // if (lane < 16) vals[warp_id][lane] += vals[warp_id][lane + 16];
//         // if (lane < 8)  vals[warp_id][lane] += vals[warp_id][lane + 8];
//         // if (lane < 4)  vals[warp_id][lane] += vals[warp_id][lane + 4];
//         // if (lane < 2)  vals[warp_id][lane] += vals[warp_id][lane + 2];
//         // if (lane < 1)  vals[warp_id][lane] += vals[warp_id][lane + 1];


//         for (int offset = 16; offset > 0; offset /= 2) {
//             sum += __shfl_down_sync(0xffffffff, sum, offset);
//         }
        
//         // Fix 1: Change accordingly because of 2D shared memory buffer. 
//         // Only lane 0 writes result
//         if (lane == 0) {
//             d_y[row] = vals[warp_id][0];
//         }

//     }
// }

// /**
//  * @brief Launches a CUDA kernel to perform Sparse Matrix-Vector Multiplication (SpMV).
//  * 
//  * This function wraps the SpMV kernel, configuring the execution parameters 
//  * and ensuring the computation completes before returning. The sparse matrix 
//  * is stored in Compressed Sparse Row (CSR) format.
//  * Feel free to change this function however you'd like. 
//  * Just ensure that the `d_y` array stores the correct output vector
//  * when the function terminates.
//  * 
//  * @param A The sparse matrix in CSR format.
//  * @param d_x Device pointer to the input vector.
//  * @param d_y Device pointer to the output vector, where the result is stored.
//  */
// void SpMV_wrapper(CSR<float, uint32_t>& A, float * d_x, float * d_y)
// {
//     //**** CHANGE THESE VALUES ****//

//     // Fix 1: Utilize more warps and threads
//     uint32_t warps_total = A.get_rows(); // Fix 3:  One warp per matrix row, now kernel covers one warp per matrix row and no rows are skipped
//     uint32_t warps_per_block = 8; // each block has 8 warps = 256 threads
//     uint32_t threads_per_block = warps_per_block * 32; // 32 threads per warp
//     uint32_t blocks = (warps_total + warps_per_block - 1) / warps_per_block;

//     // Call the kernel
//     SpMV<<<blocks, threads_per_block>>>(A.get_rows(), A.get_cols(),
//                    A.get_vals(), A.get_colinds(), A.get_rowptrs(),
//                    d_x, d_y);

//     // Sync w/ the host
//     CUDA_CHECK(cudaDeviceSynchronize());
// }

// }
// #endif

// Final implementation

#ifndef SPMV_CUH
#define SPMV_CUH

#include "common.h"
#include "CSR.hpp"

// vectorized implementation
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
                    const float * d_x, float * d_y)
{
    int warp_id = threadIdx.x / warpSize;
    int lane = threadIdx.x % warpSize;
    int warpsPerBlock = blockDim.x / warpSize;
    int row = blockIdx.x * warpsPerBlock + warp_id;

    if (row < m) {
        float sum = 0.0f;
        int rowStart = d_rowptrs[row];
        int rowEnd = d_rowptrs[row + 1];

        for (int j = rowStart + lane; j < rowEnd; j += warpSize) {
            sum += d_vals[j] * d_x[d_colinds[j]];
        }

        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }

        if (lane == 0) {
            d_y[row] = sum;
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

    // Utilize more warps and threads
    uint32_t warps_total = A.get_rows(); 
    uint32_t warps_per_block = 8; // each block has 8 warps = 256 threads
    uint32_t threads_per_block = warps_per_block * 32; // 32 threads per warp
    uint32_t blocks = (warps_total + warps_per_block - 1) / warps_per_block;

    // Call the kernel
    SpMV<<<blocks, threads_per_block>>>(A.get_rows(), A.get_cols(),
                   A.get_vals(), A.get_colinds(), A.get_rowptrs(),
                   d_x, d_y);

    // Sync w/ the host
    CUDA_CHECK(cudaDeviceSynchronize());
}

}
#endif