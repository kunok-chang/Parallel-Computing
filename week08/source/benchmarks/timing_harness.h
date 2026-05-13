#pragma once
#include <cuda_runtime.h>
#include <stdlib.h>

/* ------------------------------------------------------------------ */
/* Macro: check every CUDA API call                                    */
/* ------------------------------------------------------------------ */
#define CUDA_CHECK(call) cuda_check((call), __FILE__, __LINE__)
void cuda_check(cudaError_t err, const char *file, int line);

/* ------------------------------------------------------------------ */
/* Benchmark result structure (same as Week 7)                         */
/* ------------------------------------------------------------------ */
typedef struct {
    float h2d_ms;    /* host-to-device copy time */
    float kernel_ms; /* average kernel execution time */
    float d2h_ms;    /* device-to-host copy time */
    float bw_GBs;    /* achieved memory bandwidth (GB/s) */
} BenchResult;

/* ------------------------------------------------------------------ */
/* Generic kernel function pointer type                                */
/* ------------------------------------------------------------------ */
typedef void (*cuda_kernel_fn)(void *args);

/* ------------------------------------------------------------------ */
/* API                                                                 */
/* ------------------------------------------------------------------ */
float time_kernel_ms(cuda_kernel_fn fn, void *args, int nruns);
float time_memcpy_ms(void *dst, const void *src,
                     size_t bytes, cudaMemcpyKind kind);
void  print_result(const char *label, BenchResult r);
