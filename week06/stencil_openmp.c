#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void initialize(float* a, int Nx, int Ny) {
    for (int i = 0; i < Nx; i++) {
        for (int j = 0; j < Ny; j++) {
            a[i * Ny + j] = (float)((i + j) % 100) * 0.01f;
        }
    }
}

int main(int argc, char** argv) {
    int Nx = 2048, Ny = 2048, iters = 200;
    if (argc > 1) Nx = atoi(argv[1]);
    if (argc > 2) Ny = atoi(argv[2]);
    if (argc > 3) iters = atoi(argv[3]);

    size_t bytes = (size_t)Nx * Ny * sizeof(float);
    float* oldA = (float*)malloc(bytes);
    float* newA = (float*)malloc(bytes);
    if (!oldA || !newA) {
        fprintf(stderr, "Allocation failed\n");
        free(oldA); free(newA);
        return 1;
    }

    initialize(oldA, Nx, Ny);
    memcpy(newA, oldA, bytes);

    double t0 = now_sec();
    for (int step = 0; step < iters; step++) {
        #pragma omp parallel for collapse(2) schedule(static)
        for (int i = 1; i < Nx - 1; i++) {
            for (int j = 1; j < Ny - 1; j++) {
                newA[i * Ny + j] = 0.25f * (
                    oldA[(i - 1) * Ny + j] + oldA[(i + 1) * Ny + j] +
                    oldA[i * Ny + (j - 1)] + oldA[i * Ny + (j + 1)]);
            }
        }
        float* tmp = oldA;
        oldA = newA;
        newA = tmp;
    }
    double t1 = now_sec();

    double checksum = 0.0;
    for (int i = 0; i < Nx * Ny; i += ((Nx * Ny) / 16 > 0 ? (Nx * Ny) / 16 : 1)) {
        checksum += oldA[i];
    }

    printf("[OpenMP stencil] Nx=%d Ny=%d iters=%d threads=%d time=%.6f s checksum=%.6f\n",
           Nx, Ny, iters, omp_get_max_threads(), t1 - t0, checksum);

    free(oldA);
    free(newA);
    return 0;
}
