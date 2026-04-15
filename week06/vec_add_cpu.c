#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char** argv) {
    long N = 1000000;
    int REPEAT = 1000;

    if (argc > 1) N = atol(argv[1]);
    if (argc > 2) REPEAT = atoi(argv[2]);

    size_t bytes = (size_t)N * sizeof(float);

    float* A = (float*)malloc(bytes);
    float* B = (float*)malloc(bytes);
    float* C = (float*)malloc(bytes);

    if (!A || !B || !C) {
        fprintf(stderr, "Allocation failed\n");
        free(A); free(B); free(C);
        return 1;
    }

    for (long i = 0; i < N; i++) {
        A[i] = 1.0f + (float)(i % 1000) * 0.001f;
        B[i] = 0.5f + (float)(i % 1000) * 0.002f;
    }

    double t0 = now_sec();

    for (long i = 0; i < N; i++) {
        float x = A[i];
        float y = B[i];

        for (int r = 0; r < REPEAT; r++) {
            x = x * 1.000001f + y * 0.999999f;
            y = y * 1.000000f - x * 0.000001f;
            x = x + 0.000001f;
            y = y - 0.000001f;
        }

        C[i] = x + y;
    }

    double t1 = now_sec();

    double checksum = 0.0;
    for (long i = 0; i < N; i += (N / 10 > 0 ? N / 10 : 1)) {
        checksum += C[i];
    }

    printf("[CPU serial] N=%ld REPEAT=%d time=%.6f s checksum=%.6f\n",
           N, REPEAT, t1 - t0, checksum);

    free(A);
    free(B);
    free(C);
    return 0;
}
