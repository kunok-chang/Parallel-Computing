/*
 * bandwidth_bench.cu
 *
 * Sweeps vector-addition over several problem sizes and reports:
 *   - H2D copy time
 *   - kernel execution time (averaged over NRUNS)
 *   - D2H copy time
 *   - achieved memory bandwidth
 *
 * Build:
 *   nvcc -O2 -arch=sm_75 bandwidth_bench.cu -o bandwidth_bench
 *
 * Run:
 *   ./bandwidth_bench
 *   ./bandwidth_bench > results.csv   # redirect for plotting
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "timing_harness.cuh"

#define NRUNS        20       /* number of timed kernel repetitions */
#define THREADS      256      /* threads per block                  */

// ─────────────────────────────────────────────────────────────
// Kernel: C[i] = A[i] + B[i]
// ─────────────────────────────────────────────────────────────
__global__ void vecAdd(const float *A, const float *B, float *C, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];
}

// ─────────────────────────────────────────────────────────────
// Benchmark one problem size, return BenchResult
// ─────────────────────────────────────────────────────────────
BenchResult benchmark_vecadd(int N)
{
    size_t bytes = N * sizeof(float);

    // --- allocate host memory (page-locked for accurate copy timing) ---
    float *h_A, *h_B, *h_C;
    CUDA_CHECK(cudaMallocHost(&h_A, bytes));
    CUDA_CHECK(cudaMallocHost(&h_B, bytes));
    CUDA_CHECK(cudaMallocHost(&h_C, bytes));
    for (int i = 0; i < N; i++) { h_A[i] = 1.0f; h_B[i] = 2.0f; }

    // --- allocate device memory ---
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    CudaTimer t = timer_create();
    BenchResult res = {0};

    // --- H2D copy (both arrays) ---
    timer_start(&t);
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));
    res.h2d_ms = timer_stop_ms(&t);

    // --- warm-up run (not timed) ---
    int blocks = (N + THREADS - 1) / THREADS;
    vecAdd<<<blocks, THREADS>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // --- timed runs ---
    float total_ms = 0.0f;
    for (int r = 0; r < NRUNS; r++) {
        timer_start(&t);
        vecAdd<<<blocks, THREADS>>>(d_A, d_B, d_C, N);
        total_ms += timer_stop_ms(&t);
    }
    res.kernel_ms = total_ms / NRUNS;

    // --- D2H copy ---
    timer_start(&t);
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));
    res.d2h_ms = timer_stop_ms(&t);

    // --- achieved bandwidth: 2 reads + 1 write, averaged over timed runs ---
    size_t bytes_moved = 3ULL * bytes;
    res.bw_GB_s = (float)bytes_moved / (res.kernel_ms * 1e6f);

    // --- verify (spot check) ---
    for (int i = 0; i < N; i++) {
        if (fabsf(h_C[i] - 3.0f) > 1e-4f) {
            fprintf(stderr, "Verification FAILED at i=%d\n", i);
            break;
        }
    }

    timer_destroy(&t);
    CUDA_CHECK(cudaFreeHost(h_A)); CUDA_CHECK(cudaFreeHost(h_B));
    CUDA_CHECK(cudaFreeHost(h_C));
    CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return res;
}

// ─────────────────────────────────────────────────────────────
// Main: sweep over problem sizes
// ─────────────────────────────────────────────────────────────
int main(void)
{
    int sizes[] = {100000, 500000, 1000000, 5000000,
                   10000000, 50000000, 100000000};
    int nsizes  = (int)(sizeof(sizes) / sizeof(sizes[0]));

    printf("%-10s  %-7s  %-10s  %-7s  %-7s  %-9s\n",
           "N", "H2D(ms)", "Kernel(ms)", "D2H(ms)", "BW(GB/s)", "Total(ms)");
    printf("%-10s  %-7s  %-10s  %-7s  %-7s  %-9s\n",
           "----------", "-------", "----------", "-------", "-------", "---------");

    for (int k = 0; k < nsizes; k++) {
        int N = sizes[k];
        BenchResult r = benchmark_vecadd(N);
        float total = r.h2d_ms + r.kernel_ms + r.d2h_ms;
        printf("%-10d  %7.3f  %10.3f  %7.3f  %7.1f  %9.3f\n",
               N, r.h2d_ms, r.kernel_ms, r.d2h_ms, r.bw_GB_s, total);
    }
    return 0;
}
