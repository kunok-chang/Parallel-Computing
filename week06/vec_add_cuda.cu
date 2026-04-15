#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void heavy_compute_kernel(const float* A, const float* B, float* C, long N, int REPEAT) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float x = A[idx];
        float y = B[idx];

        #pragma unroll 4
        for (int r = 0; r < REPEAT; r++) {
            x = x * 1.000001f + y * 0.999999f;
            y = y * 1.000000f - x * 0.000001f;
            x = x + 0.000001f;
            y = y - 0.000001f;
        }

        C[idx] = x + y;
    }
}

static void check_cuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}

int main(int argc, char** argv) {
    long N = 1000000;
    int REPEAT = 1000;

    if (argc > 1) N = atol(argv[1]);
    if (argc > 2) REPEAT = atoi(argv[2]);

    size_t bytes = (size_t)N * sizeof(float);

    float* A = (float*)malloc(bytes);
    float* B = (float*)malloc(bytes);
    float* C = (float*)malloc(bytes);

    if (!A || !B || !C) {
        fprintf(stderr, "Host allocation failed\n");
        free(A); free(B); free(C);
        return 1;
    }

    for (long i = 0; i < N; i++) {
        A[i] = 1.0f + (float)(i % 1000) * 0.001f;
        B[i] = 0.5f + (float)(i % 1000) * 0.002f;
    }

    float *d_A = NULL, *d_B = NULL, *d_C = NULL;
    check_cuda(cudaMalloc((void**)&d_A, bytes), "cudaMalloc d_A");
    check_cuda(cudaMalloc((void**)&d_B, bytes), "cudaMalloc d_B");
    check_cuda(cudaMalloc((void**)&d_C, bytes), "cudaMalloc d_C");

    cudaEvent_t e_start_total, e_stop_total, e_start_kernel, e_stop_kernel;
    check_cuda(cudaEventCreate(&e_start_total), "event create total start");
    check_cuda(cudaEventCreate(&e_stop_total), "event create total stop");
    check_cuda(cudaEventCreate(&e_start_kernel), "event create kernel start");
    check_cuda(cudaEventCreate(&e_stop_kernel), "event create kernel stop");

    int blockSize = 256;
    int gridSize = (int)((N + blockSize - 1) / blockSize);

    check_cuda(cudaEventRecord(e_start_total), "record total start");

    check_cuda(cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice), "memcpy A H2D");
    check_cuda(cudaMemcpy(d_B, B, bytes, cudaMemcpyHostToDevice), "memcpy B H2D");

    check_cuda(cudaEventRecord(e_start_kernel), "record kernel start");

    heavy_compute_kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, N, REPEAT);

    check_cuda(cudaEventRecord(e_stop_kernel), "record kernel stop");
    check_cuda(cudaGetLastError(), "kernel launch");
    check_cuda(cudaEventSynchronize(e_stop_kernel), "sync kernel stop");

    check_cuda(cudaMemcpy(C, d_C, bytes, cudaMemcpyDeviceToHost), "memcpy C D2H");

    check_cuda(cudaEventRecord(e_stop_total), "record total stop");
    check_cuda(cudaEventSynchronize(e_stop_total), "sync total stop");

    float kernel_ms = 0.0f, total_ms = 0.0f;
    check_cuda(cudaEventElapsedTime(&kernel_ms, e_start_kernel, e_stop_kernel), "elapsed kernel");
    check_cuda(cudaEventElapsedTime(&total_ms, e_start_total, e_stop_total), "elapsed total");

    double checksum = 0.0;
    for (long i = 0; i < N; i += (N / 10 > 0 ? N / 10 : 1)) {
        checksum += C[i];
    }

    printf("[CUDA kernel-only] N=%ld REPEAT=%d time=%.6f s checksum=%.6f\n",
           N, REPEAT, kernel_ms / 1000.0, checksum);
    printf("[CUDA total]       N=%ld REPEAT=%d time=%.6f s checksum=%.6f\n",
           N, REPEAT, total_ms / 1000.0, checksum);

    cudaEventDestroy(e_start_total);
    cudaEventDestroy(e_stop_total);
    cudaEventDestroy(e_start_kernel);
    cudaEventDestroy(e_stop_kernel);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(A);
    free(B);
    free(C);
    return 0;
}
