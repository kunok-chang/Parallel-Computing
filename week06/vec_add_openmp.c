#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main(int argc, char** argv) {
    long N = 10000000;
    if (argc > 1) N = atol(argv[1]);

    size_t bytes = (size_t)N * sizeof(float);
    float* A = (float*)malloc(bytes);
    float* B = (float*)malloc(bytes);
    float* C = (float*)malloc(bytes);

    if (!A || !B || !C) {
        fprintf(stderr, "Allocation failed\n");
        free(A);
        free(B);
        free(C);
        return 1;
    }

    #pragma omp parallel for schedule(static)
    for (long i = 0; i < N; i++) {
        A[i] = 1.0f + 0.001f * (float)(i % 1000);
        B[i] = 2.0f + 0.002f * (float)(i % 1000);
    }

    double t0 = omp_get_wtime();
    #pragma omp parallel for schedule(static)
    for (long i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }
    double t1 = omp_get_wtime();

    double checksum = 0.0;
    long stride = (N / 16 > 0) ? (N / 16) : 1;
    for (long i = 0; i < N; i += stride) {
        checksum += C[i];
    }

    printf("[openmp vec_add] N=%ld threads=%d time=%.6f s checksum=%.6f\n",
           N, omp_get_max_threads(), t1 - t0, checksum);

    free(A);
    free(B);
    free(C);
    return 0;
}
