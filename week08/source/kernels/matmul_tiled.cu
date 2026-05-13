/*
 * matmul_tiled.cu
 * Tiled matrix multiplication using shared memory.
 * C = A * B  where A, B, C are N x N row-major float matrices.
 * Week 8 -- CUDA Kernel Optimization
 *
 * Students: complete the sections marked TODO.
 */

#include "matmul_tiled.h"

#ifndef TILE_SIZE
#define TILE_SIZE 16
#endif

/*
 * Tiled GEMM kernel.
 *
 * Each block computes a TILE_SIZE x TILE_SIZE sub-matrix of C.
 * The computation is split into (N / TILE_SIZE) phases.
 * In each phase, a tile of A and a tile of B are loaded into
 * shared memory, and the partial dot-products are accumulated.
 *
 * Global memory traffic: reduced by a factor of TILE_SIZE
 * compared to the naive implementation.
 */
__global__ void matmul_tiled_kernel(const float *A, const float *B,
                                    float *C, int N)
{
    /* TODO 1: Declare shared memory tiles for A and B.
     *         Both are TILE_SIZE x TILE_SIZE floats.
     *         Add a +1 padding column to each to avoid bank conflicts. */
    __shared__ float As[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE + 1];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    float sum = 0.0f;

    /* TODO 2: Loop over tiles along the K dimension.
     *         In each iteration t:
     *           (a) Load As[ty][tx] = A[row][t*TILE_SIZE + tx]  (guard bounds)
     *           (b) Load Bs[ty][tx] = B[t*TILE_SIZE + ty][col]  (guard bounds)
     *           (c) __syncthreads()
     *           (d) Accumulate the partial dot-product
     *           (e) __syncthreads() before loading the next tile */
    int num_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; t++) {
        /* (a) Load tile of A */
        int a_col = t * TILE_SIZE + tx;
        As[ty][tx] = (row < N && a_col < N) ? A[row * N + a_col] : 0.0f;

        /* (b) Load tile of B */
        int b_row = t * TILE_SIZE + ty;
        Bs[ty][tx] = (b_row < N && col < N) ? B[b_row * N + col] : 0.0f;

        /* (c) Synchronize: all threads must finish loading before computing */
        __syncthreads();

        /* (d) Accumulate partial dot-product */
        for (int k = 0; k < TILE_SIZE; k++)
            sum += As[ty][k] * Bs[k][tx];

        /* (e) Synchronize before loading the next tile */
        __syncthreads();
    }

    /* TODO 3: Write the result, guarding against out-of-bound writes. */
    if (row < N && col < N)
        C[row * N + col] = sum;
}

void launch_matmul_tiled(const float *d_A, const float *d_B,
                          float *d_C, int N)
{
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE,
              (N + TILE_SIZE - 1) / TILE_SIZE);
    matmul_tiled_kernel<<<grid, block>>>(d_A, d_B, d_C, N);
}
