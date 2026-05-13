/*
 * timing_harness.cu
 * CUDA event-based timing utilities (carried over from Week 7).
 * Week 8 -- CUDA Kernel Optimization
 */

#include <stdio.h>
#include "timing_harness.h"

/* ------------------------------------------------------------------ */
/* Helper: check CUDA errors                                           */
/* ------------------------------------------------------------------ */
void cuda_check(cudaError_t err, const char *file, int line)
{
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d: %s\n",
                file, line, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

/* ------------------------------------------------------------------ */
/* Time a single CUDA kernel launch (averaged over nruns warm runs).   */
/* A single warm-up run is performed first and discarded.              */
/* ------------------------------------------------------------------ */
float time_kernel_ms(cuda_kernel_fn fn, void *args,
                     int nruns)
{
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    /* Warm-up */
    fn(args);
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Timed runs */
    float total_ms = 0.0f;
    for (int r = 0; r < nruns; r++) {
        CUDA_CHECK(cudaEventRecord(start));
        fn(args);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        total_ms += ms;
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return total_ms / (float)nruns;
}

/* ------------------------------------------------------------------ */
/* Time a cudaMemcpy call.                                             */
/* ------------------------------------------------------------------ */
float time_memcpy_ms(void *dst, const void *src,
                     size_t bytes, cudaMemcpyKind kind)
{
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(cudaMemcpy(dst, src, bytes, kind));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return ms;
}

/* ------------------------------------------------------------------ */
/* Print a BenchResult to stdout.                                      */
/* ------------------------------------------------------------------ */
void print_result(const char *label, BenchResult r)
{
    printf("%-30s  H2D=%6.3f ms  kernel=%6.3f ms  "
           "D2H=%6.3f ms  BW=%6.1f GB/s\n",
           label, r.h2d_ms, r.kernel_ms, r.d2h_ms, r.bw_GBs);
}
