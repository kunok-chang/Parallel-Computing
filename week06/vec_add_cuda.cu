#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while (0)

__global__ void vec_add_kernel(const float* A, const float* B, float* C, long N) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main(int argc, char** argv) {
    long N = 10000000;
    if (argc > 1) N = atol(argv[1]);

    size_t bytes = (size_t)N * sizeof(float);
    float* A = (float*)malloc(bytes);
    float* B = (float*)malloc(bytes);
    float* C = (float*)malloc(bytes);
    float *d_A = NULL, *d_B = NULL, *d_C = NULL;

    if (!A || !B || !C) {
        fprintf(stderr, "Host allocation failed\n");
        free(A);
        free(B);
        free(C);
        return 1;
    }

    for (long i = 0; i < N; i++) {
        A[i] = 1.0f + 0.001f * (float)(i % 1000);
        B[i] = 2.0f + 0.002f * (float)(i % 1000);
    }

    CUDA_CHECK(cudaMalloc((void**)&d_A, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_B, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_C, bytes));

    cudaEvent_t start_total, stop_total, start_kernel, stop_kernel;
    CUDA_CHECK(cudaEventCreate(&start_total));
    CUDA_CHECK(cudaEventCreate(&stop_total));
    CUDA_CHECK(cudaEventCreate(&start_kernel));
    CUDA_CHECK(cudaEventCreate(&stop_kernel));

    int block = 256;
    int grid = (int)((N + block - 1) / block);

    CUDA_CHECK(cudaEventRecord(start_total));
    CUDA_CHECK(cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, bytes, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaEventRecord(start_kernel));
    vec_add_kernel<<<grid, block>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop_kernel));
    CUDA_CHECK(cudaEventSynchronize(stop_kernel));

    CUDA_CHECK(cudaMemcpy(C, d_C, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(stop_total));
    CUDA_CHECK(cudaEventSynchronize(stop_total));

    float kernel_ms = 0.0f, total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, start_kernel, stop_kernel));
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start_total, stop_total));

    double checksum = 0.0;
    long stride = (N / 16 > 0) ? (N / 16) : 1;
    for (long i = 0; i < N; i += stride) {
        checksum += C[i];
    }

    printf("[cuda vec_add kernel] N=%ld time=%.6f s checksum=%.6f\n", N, kernel_ms / 1000.0, checksum);
    printf("[cuda vec_add total ] N=%ld time=%.6f s checksum=%.6f\n", N, total_ms / 1000.0, checksum);

    CUDA_CHECK(cudaEventDestroy(start_total));
    CUDA_CHECK(cudaEventDestroy(stop_total));
    CUDA_CHECK(cudaEventDestroy(start_kernel));
    CUDA_CHECK(cudaEventDestroy(stop_kernel));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(A);
    free(B);
    free(C);
    return 0;
}
