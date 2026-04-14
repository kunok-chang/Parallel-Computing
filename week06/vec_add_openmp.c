#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char** argv) {
    long N = 10000000;
    if (argc > 1) {
        N = atol(argv[1]);
    }

    size_t bytes = (size_t)N * sizeof(float);
    float* A = (float*)malloc(bytes);
    float* B = (float*)malloc(bytes);
    float* C = (float*)malloc(bytes);
    if (!A || !B || !C) {
        fprintf(stderr, "Allocation failed\n");
        free(A); free(B); free(C);
        return 1;
    }

    #pragma omp parallel for schedule(static)
    for (long i = 0; i < N; i++) {
        A[i] = (float)(i % 1000) * 0.5f;
        B[i] = (float)(i % 1000) * 0.25f;
    }

    double t0 = now_sec();
    #pragma omp parallel for schedule(static)
    for (long i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }
    double t1 = now_sec();

    double checksum = 0.0;
    for (long i = 0; i < N; i += (N / 10 > 0 ? N / 10 : 1)) {
        checksum += C[i];
    }

    printf("[OpenMP] N=%ld threads=%d time=%.6f s checksum=%.6f\n",
           N, omp_get_max_threads(), t1 - t0, checksum);

    free(A);
    free(B);
    free(C);
    return 0;
}
