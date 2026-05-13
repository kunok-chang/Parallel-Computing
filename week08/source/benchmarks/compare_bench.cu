/*
 * compare_bench.cu
 * Compares naive vs tiled kernels for stencil and GEMM.
 *
 * Usage:
 *   ./compare_bench stencil [N]       (default N=1024)
 *   ./compare_bench matmul  [N]       (default N=1024)
 *   ./compare_bench all     [N]
 *
 * Output: CSV line appended to results.csv for plot_speedup.py
 * Week 8 -- CUDA Kernel Optimization
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

#include "timing_harness.h"
#include "../kernels/stencil_naive.h"
#include "../kernels/stencil_tiled.h"
#include "../kernels/matmul_naive.h"
#include "../kernels/matmul_tiled.h"

#define NRUNS 20

/* ------------------------------------------------------------------ */
/* Stencil argument bundle (used with timing_harness generic pointer)  */
/* ------------------------------------------------------------------ */
typedef struct { const float *in; float *out; int Nx, Ny; } StencilArgs;

static void run_stencil_naive_wrap(void *vargs) {
    StencilArgs *a = (StencilArgs *)vargs;
    dim3 block(16, 16);
    dim3 grid((a->Nx + 15)/16, (a->Ny + 15)/16);
    launch_stencil_naive(a->in, a->out, a->Nx, a->Ny, block, grid);
}
static void run_stencil_tiled_wrap(void *vargs) {
    StencilArgs *a = (StencilArgs *)vargs;
    launch_stencil_tiled(a->in, a->out, a->Nx, a->Ny);
}

/* ------------------------------------------------------------------ */
/* GEMM argument bundle                                                */
/* ------------------------------------------------------------------ */
typedef struct { const float *A, *B; float *C; int N; } GemmArgs;

static void run_matmul_naive_wrap(void *vargs) {
    GemmArgs *a = (GemmArgs *)vargs;
    launch_matmul_naive(a->A, a->B, a->C, a->N);
}
static void run_matmul_tiled_wrap(void *vargs) {
    GemmArgs *a = (GemmArgs *)vargs;
    launch_matmul_tiled(a->A, a->B, a->C, a->N);
}

/* ------------------------------------------------------------------ */
/* Benchmark: stencil                                                  */
/* ------------------------------------------------------------------ */
void bench_stencil(int N, FILE *csv)
{
    size_t bytes = (size_t)N * N * sizeof(float);

    float *h_in  = (float *)malloc(bytes);
    float *h_out = (float *)malloc(bytes);
    for (int i = 0; i < N*N; i++) h_in[i] = (float)(i % 100) * 0.01f;

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in,  bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));

    /* H2D */
    float h2d_ms = time_memcpy_ms(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    /* Naive kernel */
    StencilArgs args = { d_in, d_out, N, N };
    float naive_ms = time_kernel_ms(run_stencil_naive_wrap, &args, NRUNS);
    /* Bandwidth: 2 reads + 1 write per interior cell (approx N^2) */
    float naive_bw = (3.0f * bytes) / (naive_ms * 1e6f);

    /* Tiled kernel */
    float tiled_ms = time_kernel_ms(run_stencil_tiled_wrap, &args, NRUNS);
    float tiled_bw = (3.0f * bytes) / (tiled_ms * 1e6f);

    /* D2H */
    float d2h_ms = time_memcpy_ms(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    BenchResult r_naive = { h2d_ms, naive_ms, d2h_ms, naive_bw };
    BenchResult r_tiled = { h2d_ms, tiled_ms, d2h_ms, tiled_bw };

    printf("\n=== Stencil (N=%d) ===\n", N);
    print_result("stencil_naive", r_naive);
    print_result("stencil_tiled", r_tiled);
    printf("  Speedup: %.2fx\n", naive_ms / tiled_ms);

    if (csv)
        fprintf(csv, "stencil,%d,naive,%.4f,%.2f\n"
                     "stencil,%d,tiled,%.4f,%.2f\n",
                N, naive_ms, naive_bw,
                N, tiled_ms, tiled_bw);

    cudaFree(d_in); cudaFree(d_out);
    free(h_in); free(h_out);
}

/* ------------------------------------------------------------------ */
/* Benchmark: GEMM                                                     */
/* ------------------------------------------------------------------ */
void bench_matmul(int N, FILE *csv)
{
    size_t bytes = (size_t)N * N * sizeof(float);

    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C = (float *)malloc(bytes);
    for (int i = 0; i < N*N; i++) {
        h_A[i] = (float)(rand() % 100) / 100.0f;
        h_B[i] = (float)(rand() % 100) / 100.0f;
    }

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    /* H2D (both A and B) */
    float h2d_ms  = time_memcpy_ms(d_A, h_A, bytes, cudaMemcpyHostToDevice);
           h2d_ms += time_memcpy_ms(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    /* FLOPs: 2 * N^3 (multiply-add for each element) */
    float flops = 2.0f * (float)N * (float)N * (float)N;

    /* Naive */
    GemmArgs args = { d_A, d_B, d_C, N };
    float naive_ms = time_kernel_ms(run_matmul_naive_wrap, &args, NRUNS);
    float naive_gflops = flops / (naive_ms * 1e6f);
    /* Bandwidth (A+B reads, C write) */
    float naive_bw = (3.0f * bytes) / (naive_ms * 1e6f);

    /* Tiled */
    float tiled_ms = time_kernel_ms(run_matmul_tiled_wrap, &args, NRUNS);
    float tiled_gflops = flops / (tiled_ms * 1e6f);
    float tiled_bw     = (3.0f * bytes) / (tiled_ms * 1e6f);

    /* D2H */
    float d2h_ms = time_memcpy_ms(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    printf("\n=== GEMM (N=%d) ===\n", N);
    printf("%-30s  kernel=%6.3f ms  BW=%6.1f GB/s  %6.1f GFLOP/s\n",
           "matmul_naive", naive_ms, naive_bw, naive_gflops);
    printf("%-30s  kernel=%6.3f ms  BW=%6.1f GB/s  %6.1f GFLOP/s\n",
           "matmul_tiled", tiled_ms, tiled_bw, tiled_gflops);
    printf("  Speedup: %.2fx\n", naive_ms / tiled_ms);
    (void)h2d_ms; (void)d2h_ms;

    if (csv)
        fprintf(csv, "matmul,%d,naive,%.4f,%.2f\n"
                     "matmul,%d,tiled,%.4f,%.2f\n",
                N, naive_ms, naive_gflops,
                N, tiled_ms, tiled_gflops);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
}

/* ------------------------------------------------------------------ */
/* main                                                                */
/* ------------------------------------------------------------------ */
int main(int argc, char **argv)
{
    const char *mode = (argc >= 2) ? argv[1] : "all";
    int N = (argc >= 3) ? atoi(argv[2]) : 1024;

    printf("CUDA Week 8 -- Optimization Benchmark\n");
    printf("GPU: ");
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("%s  (SM %d.%d)\n\n", prop.name,
           prop.major, prop.minor);

    FILE *csv = fopen("results.csv", "a");
    if (csv) fprintf(csv, "kernel,N,variant,time_ms,metric\n");

    if (strcmp(mode, "stencil") == 0 || strcmp(mode, "all") == 0)
        bench_stencil(N, csv);

    if (strcmp(mode, "matmul") == 0 || strcmp(mode, "all") == 0)
        bench_matmul(N, csv);

    if (csv) fclose(csv);
    return 0;
}
