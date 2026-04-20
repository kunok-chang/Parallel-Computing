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

static void initialize(float* a, int Nx, int Ny) {
    for (int i = 0; i < Nx; i++) {
        for (int j = 0; j < Ny; j++) {
            if (i == 0 || j == 0 || i == Nx - 1 || j == Ny - 1) {
                a[(size_t)i * Ny + j] = 1.0f;
            } else {
                a[(size_t)i * Ny + j] = (float)((i + j) & 15) * 0.001f;
            }
        }
    }
}

template <int BX, int BY>
__global__ void stencil2d_shared(const float* __restrict__ oldA,
                                 float* __restrict__ newA,
                                 int Nx, int Ny) {
    __shared__ float tile[BY + 2][BX + 2];

    int j = blockIdx.x * BX + threadIdx.x;
    int i = blockIdx.y * BY + threadIdx.y;
    int tx = threadIdx.x + 1;
    int ty = threadIdx.y + 1;

    if (i < Nx && j < Ny) {
        tile[ty][tx] = oldA[(size_t)i * Ny + j];
    }

    if (threadIdx.x == 0 && j > 0 && i < Nx) {
        tile[ty][0] = oldA[(size_t)i * Ny + (j - 1)];
    }
    if (threadIdx.x == BX - 1 && j + 1 < Ny && i < Nx) {
        tile[ty][BX + 1] = oldA[(size_t)i * Ny + (j + 1)];
    }
    if (threadIdx.y == 0 && i > 0 && j < Ny) {
        tile[0][tx] = oldA[(size_t)(i - 1) * Ny + j];
    }
    if (threadIdx.y == BY - 1 && i + 1 < Nx && j < Ny) {
        tile[BY + 1][tx] = oldA[(size_t)(i + 1) * Ny + j];
    }

    if (threadIdx.x == 0 && threadIdx.y == 0 && i > 0 && j > 0) {
        tile[0][0] = oldA[(size_t)(i - 1) * Ny + (j - 1)];
    }
    if (threadIdx.x == BX - 1 && threadIdx.y == 0 && i > 0 && j + 1 < Ny) {
        tile[0][BX + 1] = oldA[(size_t)(i - 1) * Ny + (j + 1)];
    }
    if (threadIdx.x == 0 && threadIdx.y == BY - 1 && i + 1 < Nx && j > 0) {
        tile[BY + 1][0] = oldA[(size_t)(i + 1) * Ny + (j - 1)];
    }
    if (threadIdx.x == BX - 1 && threadIdx.y == BY - 1 && i + 1 < Nx && j + 1 < Ny) {
        tile[BY + 1][BX + 1] = oldA[(size_t)(i + 1) * Ny + (j + 1)];
    }

    __syncthreads();

    if (i > 0 && i < Nx - 1 && j > 0 && j < Ny - 1) {
        newA[(size_t)i * Ny + j] = 0.2f * (
            tile[ty][tx] +
            tile[ty - 1][tx] + tile[ty + 1][tx] +
            tile[ty][tx - 1] + tile[ty][tx + 1]);
    }
}

int main(int argc, char** argv) {
    int Nx = 4096, Ny = 4096, iters = 400;
    if (argc > 1) Nx = atoi(argv[1]);
    if (argc > 2) Ny = atoi(argv[2]);
    if (argc > 3) iters = atoi(argv[3]);

    size_t n = (size_t)Nx * Ny;
    size_t bytes = n * sizeof(float);

    float* h_old = (float*)malloc(bytes);
    float* h_new = (float*)malloc(bytes);
    float *d_old = NULL, *d_new = NULL;
    if (!h_old || !h_new) {
        fprintf(stderr, "Host allocation failed\n");
        free(h_old);
        free(h_new);
        return 1;
    }

    initialize(h_old, Nx, Ny);
    memcpy(h_new, h_old, bytes);

    CUDA_CHECK(cudaMalloc((void**)&d_old, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_new, bytes));
    CUDA_CHECK(cudaMemcpy(d_old, h_old, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_new, h_new, bytes, cudaMemcpyHostToDevice));

    dim3 block(32, 8);
    dim3 grid((unsigned int)((Ny + block.x - 1) / block.x),
              (unsigned int)((Nx + block.y - 1) / block.y));

    stencil2d_shared<32, 8><<<grid, block>>>(d_old, d_new, Nx, Ny);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    for (int step = 0; step < iters; step++) {
        stencil2d_shared<32, 8><<<grid, block>>>(d_old, d_new, Nx, Ny);
    
        CUDA_CHECK(cudaGetLastError());
        float* tmp = d_old;
        d_old = d_new;
        d_new = tmp;
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaMemcpy(h_old, d_old, bytes, cudaMemcpyDeviceToHost));

    double checksum = 0.0;
    size_t stride = (n / 32 > 0) ? (n / 32) : 1;
    for (size_t i = 0; i < n; i += stride) {
        checksum += h_old[i];
    }

    printf("[cuda stencil compute] Nx=%d Ny=%d iters=%d time=%.6f s checksum=%.6f\n",
           Nx, Ny, iters, ms / 1000.0, checksum);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_old));
    CUDA_CHECK(cudaFree(d_new));
    free(h_old);
    free(h_new);
    return 0;
}
