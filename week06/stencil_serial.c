#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void initialize(float* a, int Nx, int Ny) {
    for (int i = 0; i < Nx; i++) {
        for (int j = 0; j < Ny; j++) {
            if (i == 0 || j == 0 || i == Nx - 1 || j == Ny - 1) {
                a[(size_t)i * Ny + j] = 1.0f;
            } else {
                a[(size_t)i * Ny + j] = (float)((i + j) & 15) * 0.001f;
            }
        }
    }
}

int main(int argc, char** argv) {
    int Nx = 4096, Ny = 4096, iters = 400;
    if (argc > 1) Nx = atoi(argv[1]);
    if (argc > 2) Ny = atoi(argv[2]);
    if (argc > 3) iters = atoi(argv[3]);

    size_t n = (size_t)Nx * Ny;
    size_t bytes = n * sizeof(float);
    float* oldA = (float*)malloc(bytes);
    float* newA = (float*)malloc(bytes);
    if (!oldA || !newA) {
        fprintf(stderr, "Allocation failed\n");
        free(oldA);
        free(newA);
        return 1;
    }

    initialize(oldA, Nx, Ny);
    memcpy(newA, oldA, bytes);

    double t0 = now_sec();
    for (int step = 0; step < iters; step++) {
        for (int i = 1; i < Nx - 1; i++) {
            size_t row = (size_t)i * Ny;
            size_t row_up = row - Ny;
            size_t row_dn = row + Ny;
            for (int j = 1; j < Ny - 1; j++) {
                newA[row + j] = 0.2f * (
                    oldA[row + j] +
                    oldA[row_up + j] + oldA[row_dn + j] +
                    oldA[row + j - 1] + oldA[row + j + 1]);
            }
        }
        float* tmp = oldA;
        oldA = newA;
        newA = tmp;
    }
    double t1 = now_sec();

    double checksum = 0.0;
    size_t stride = (n / 32 > 0) ? (n / 32) : 1;
    for (size_t i = 0; i < n; i += stride) {
        checksum += oldA[i];
    }

    printf("[serial stencil] Nx=%d Ny=%d iters=%d time=%.6f s checksum=%.6f\n",
           Nx, Ny, iters, t1 - t0, checksum);

    free(oldA);
    free(newA);
    return 0;
}
