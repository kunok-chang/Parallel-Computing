#pragma once
#include <stdio.h>
#include <cuda_runtime.h>

// ─────────────────────────────────────────────────────────────
// Macro: check every CUDA call and abort on error
// ─────────────────────────────────────────────────────────────
#define CUDA_CHECK(call)                                                \
    do {                                                                \
        cudaError_t err = (call);                                       \
        if (err != cudaSuccess) {                                       \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",              \
                    __FILE__, __LINE__, cudaGetErrorString(err));       \
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (0)

// ─────────────────────────────────────────────────────────────
// Result struct returned by every benchmark function
// ─────────────────────────────────────────────────────────────
typedef struct {
    float h2d_ms;       // host-to-device copy time
    float kernel_ms;    // kernel execution time (averaged)
    float d2h_ms;       // device-to-host copy time
    float bw_GB_s;      // achieved memory bandwidth  [GB/s]
} BenchResult;

// ─────────────────────────────────────────────────────────────
// Print a result row to stdout
// ─────────────────────────────────────────────────────────────
static inline void print_result(const char *label, BenchResult r)
{
    printf("%-30s  H2D=%7.3f ms  kernel=%7.3f ms  D2H=%7.3f ms  BW=%6.1f GB/s\n",
           label, r.h2d_ms, r.kernel_ms, r.d2h_ms, r.bw_GB_s);
}

// ─────────────────────────────────────────────────────────────
// RAII-style CUDA event pair for convenience
// ─────────────────────────────────────────────────────────────
typedef struct {
    cudaEvent_t start;
    cudaEvent_t stop;
} CudaTimer;

static inline CudaTimer timer_create(void)
{
    CudaTimer t;
    CUDA_CHECK(cudaEventCreate(&t.start));
    CUDA_CHECK(cudaEventCreate(&t.stop));
    return t;
}

static inline void timer_destroy(CudaTimer *t)
{
    cudaEventDestroy(t->start);
    cudaEventDestroy(t->stop);
}

static inline void timer_start(CudaTimer *t)
{
    CUDA_CHECK(cudaEventRecord(t->start));
}

static inline float timer_stop_ms(CudaTimer *t)
{
    float ms = 0.0f;
    CUDA_CHECK(cudaEventRecord(t->stop));
    CUDA_CHECK(cudaEventSynchronize(t->stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, t->start, t->stop));
    return ms;
}
