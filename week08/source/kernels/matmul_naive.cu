/*
 * matmul_naive.cu
 * Baseline matrix multiplication kernel (no shared memory).
 * C = A * B  where A, B, C are N x N row-major float matrices.
 * Week 8 -- CUDA Kernel Optimization
 */

#include "matmul_naive.h"

__global__ void matmul_naive_kernel(const float *A, const float *B,
                                    float *C, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++)
            sum += A[row * N + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}

void launch_matmul_naive(const float *d_A, const float *d_B,
                          float *d_C, int N)
{
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (N + 15) / 16);
    matmul_naive_kernel<<<grid, block>>>(d_A, d_B, d_C, N);
}
