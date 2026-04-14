#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while (0)

__global__ void stencil2D(const float* oldA, float* newA, int Nx, int Ny) {
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int i = blockIdx.y * blockDim.y + threadIdx.y + 1;

    if (i < Nx - 1 && j < Ny - 1) {
        int id = i * Ny + j;
        newA[id] = 0.25f * (
            oldA[(i - 1) * Ny + j] + oldA[(i + 1) * Ny + j] +
            oldA[i * Ny + (j - 1)] + oldA[i * Ny + (j + 1)]);
    }
}

static void initialize(float* a, int Nx, int Ny) {
    for (int i = 0; i < Nx; i++) {
        for (int j = 0; j < Ny; j++) {
            a[i * Ny + j] = (float)((i + j) % 100) * 0.01f;
        }
    }
}

int main(int argc, char** argv) {
    int Nx = 2048, Ny = 2048, iters = 200;
    if (argc > 1) Nx = atoi(argv[1]);
    if (argc > 2) Ny = atoi(argv[2]);
    if (argc > 3) iters = atoi(argv[3]);

    size_t bytes = (size_t)Nx * Ny * sizeof(float);
    float* h_old = (float*)malloc(bytes);
    float* h_new = (float*)malloc(bytes);
    float *d_old = NULL, *d_new = NULL;
    if (!h_old || !h_new) {
        fprintf(stderr, "Host allocation failed\n");
        free(h_old); free(h_new);
        return 1;
    }

    initialize(h_old, Nx, Ny);
    memcpy(h_new, h_old, bytes);

    CUDA_CHECK(cudaMalloc((void**)&d_old, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_new, bytes));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(cudaMemcpy(d_old, h_old, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_new, h_new, bytes, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((Ny - 2 + block.x - 1) / block.x, (Nx - 2 + block.y - 1) / block.y);

    for (int step = 0; step < iters; step++) {
        stencil2D<<<grid, block>>>(d_old, d_new, Nx, Ny);
        CUDA_CHECK(cudaGetLastError());
        float* tmp = d_old;
        d_old = d_new;
        d_new = tmp;
    }

    CUDA_CHECK(cudaMemcpy(h_old, d_old, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    double checksum = 0.0;
    int total = Nx * Ny;
    for (int i = 0; i < total; i += (total / 16 > 0 ? total / 16 : 1)) {
        checksum += h_old[i];
    }

    printf("[CUDA stencil total] Nx=%d Ny=%d iters=%d time=%.6f s checksum=%.6f\n",
           Nx, Ny, iters, ms / 1000.0, checksum);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_old));
    CUDA_CHECK(cudaFree(d_new));
    free(h_old);
    free(h_new);
    return 0;
}
