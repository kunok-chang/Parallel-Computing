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

__global__ void vecAdd(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int main(int argc, char** argv) {
    int N = 10000000;
    if (argc > 1) {
        N = atoi(argv[1]);
    }

    size_t bytes = (size_t)N * sizeof(float);
    float *A = (float*)malloc(bytes), *B = (float*)malloc(bytes), *C = (float*)malloc(bytes);
    float *d_A = NULL, *d_B = NULL, *d_C = NULL;
    if (!A || !B || !C) {
        fprintf(stderr, "Host allocation failed\n");
        free(A); free(B); free(C);
        return 1;
    }

    for (int i = 0; i < N; i++) {
        A[i] = (float)(i % 1000) * 0.5f;
        B[i] = (float)(i % 1000) * 0.25f;
    }

    CUDA_CHECK(cudaMalloc((void**)&d_A, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_B, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_C, bytes));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, bytes, cudaMemcpyHostToDevice));

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(C, d_C, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    double checksum = 0.0;
    for (int i = 0; i < N; i += (N / 10 > 0 ? N / 10 : 1)) {
        checksum += C[i];
    }

    printf("[CUDA total] N=%d time=%.6f s checksum=%.6f\n", N, ms / 1000.0, checksum);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(A); free(B); free(C);
    return 0;
}
