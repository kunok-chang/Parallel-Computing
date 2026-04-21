/*
 * stencil_bench.cu
 *
 * 2D 5-point stencil benchmark.
 * Sweeps over block sizes (bx x by) and grid sizes to study
 * the effect of occupancy on kernel performance.
 *
 * Build:
 *   nvcc -O2 -arch=sm_75 stencil_bench.cu -o stencil_bench
 *
 * Run:
 *   ./stencil_bench
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "timing_harness.cuh"

#define NRUNS 20

// ─────────────────────────────────────────────────────────────
// Kernel: 2D 5-point stencil (average of 4 neighbors)
// ─────────────────────────────────────────────────────────────
__global__ void stencil2D(const float *oldA, float *newA, int Nx, int Ny)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int i = blockIdx.y * blockDim.y + threadIdx.y + 1;

    if (i < Nx - 1 && j < Ny - 1) {
        int id = i * Ny + j;
        newA[id] = 0.25f * (oldA[(i-1)*Ny + j]
                          + oldA[(i+1)*Ny + j]
                          + oldA[i*Ny + (j-1)]
                          + oldA[i*Ny + (j+1)]);
    }
}

// ─────────────────────────────────────────────────────────────
// Benchmark one (Nx, Ny, bx, by) combination
// ─────────────────────────────────────────────────────────────
BenchResult benchmark_stencil(int Nx, int Ny, int bx, int by)
{
    size_t bytes = (size_t)Nx * Ny * sizeof(float);

    float *h_old, *h_new;
    CUDA_CHECK(cudaMallocHost(&h_old, bytes));
    CUDA_CHECK(cudaMallocHost(&h_new, bytes));
    for (int k = 0; k < Nx * Ny; k++) h_old[k] = (float)k * 0.001f;

    float *d_old, *d_new;
    CUDA_CHECK(cudaMalloc(&d_old, bytes));
    CUDA_CHECK(cudaMalloc(&d_new, bytes));

    CudaTimer t = timer_create();
    BenchResult res = {0};

    // H2D
    timer_start(&t);
    CUDA_CHECK(cudaMemcpy(d_old, h_old, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_new, h_new, bytes, cudaMemcpyHostToDevice));
    res.h2d_ms = timer_stop_ms(&t);

    dim3 block(bx, by);
    dim3 grid((Ny - 2 + bx - 1) / bx, (Nx - 2 + by - 1) / by);

    // warm-up
    stencil2D<<<grid, block>>>(d_old, d_new, Nx, Ny);
    CUDA_CHECK(cudaDeviceSynchronize());

    // timed runs
    float total_ms = 0.0f;
    for (int r = 0; r < NRUNS; r++) {
        timer_start(&t);
        stencil2D<<<grid, block>>>(d_old, d_new, Nx, Ny);
        total_ms += timer_stop_ms(&t);
    }
    res.kernel_ms = total_ms / NRUNS;

    // D2H
    timer_start(&t);
    CUDA_CHECK(cudaMemcpy(h_new, d_new, bytes, cudaMemcpyDeviceToHost));
    res.d2h_ms = timer_stop_ms(&t);

    // bandwidth: 4 reads (neighbors) + 1 write per interior point
    long long interior = (long long)(Nx - 2) * (Ny - 2);
    size_t bytes_moved = 5ULL * interior * sizeof(float);
    res.bw_GB_s = (float)bytes_moved / (res.kernel_ms * 1e6f);

    timer_destroy(&t);
    CUDA_CHECK(cudaFreeHost(h_old)); CUDA_CHECK(cudaFreeHost(h_new));
    CUDA_CHECK(cudaFree(d_old));     CUDA_CHECK(cudaFree(d_new));
    return res;
}

// ─────────────────────────────────────────────────────────────
// Suggest optimal block size using the occupancy API
// ─────────────────────────────────────────────────────────────
static void print_occupancy_suggestion(void)
{
    int minGridSize, blockSize;
    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(
        &minGridSize, &blockSize, stencil2D, 0, 0));
    printf("\n[Occupancy API] Suggested 1D block size: %d  |  Min grid size: %d\n\n",
           blockSize, minGridSize);
}

// ─────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────
int main(void)
{
    print_occupancy_suggestion();

    // Block size configurations to test
    int bx[] = {8,  16, 32, 16, 32};
    int by[] = {8,  16,  8, 32, 32};
    int nconf = 5;

    // Grid sizes to test
    int grids[][2] = {{1024, 1024}, {2048, 2048}, {4096, 4096}};
    int ngrids = 3;

    for (int g = 0; g < ngrids; g++) {
        int Nx = grids[g][0], Ny = grids[g][1];
        printf("=== Grid %dx%d ===\n", Nx, Ny);
        printf("%-10s  %-10s  %-9s  %-9s\n",
               "Block", "Kernel(ms)", "BW(GB/s)", "Total(ms)");
        printf("%-10s  %-10s  %-9s  %-9s\n",
               "----------", "----------", "---------", "---------");

        for (int c = 0; c < nconf; c++) {
            char label[16];
            snprintf(label, sizeof(label), "%dx%d", bx[c], by[c]);
            BenchResult r = benchmark_stencil(Nx, Ny, bx[c], by[c]);
            float total = r.h2d_ms + r.kernel_ms + r.d2h_ms;
            printf("%-10s  %10.3f  %9.1f  %9.3f\n",
                   label, r.kernel_ms, r.bw_GB_s, total);
        }
        printf("\n");
    }
    return 0;
}
