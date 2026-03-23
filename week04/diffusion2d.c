// diffusion2d.c
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

static void initialize(double *a, int Nx, int Ny) {
    for (int i = 0; i < Nx; i++) {
        for (int j = 0; j < Ny; j++) {
            if (i == 0 || i == Nx - 1 || j == 0 || j == Ny - 1) {
                a[i * Ny + j] = 1.0;   // boundary
            } else {
                a[i * Ny + j] = 0.0;
            }
        }
    }
}

static double checksum(double *a, int Nx, int Ny) {
    double s = 0.0;
    for (long long i = 0; i < (long long)Nx * Ny; i++) s += a[i];
    return s;
}

int main(int argc, char *argv[]) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s Nx Ny steps\n", argv[0]);
        return 1;
    }

    int Nx = atoi(argv[1]);
    int Ny = atoi(argv[2]);
    int steps = atoi(argv[3]);

    double *old = (double *)malloc(sizeof(double) * (long long)Nx * Ny);
    double *newa = (double *)malloc(sizeof(double) * (long long)Nx * Ny);

    if (!old || !newa) {
        fprintf(stderr, "Allocation failed\n");
        free(old); free(newa);
        return 1;
    }

    initialize(old, Nx, Ny);
    initialize(newa, Nx, Ny);

    double t0 = omp_get_wtime();

    for (int step = 0; step < steps; step++) {
        #pragma omp parallel for collapse(2) schedule(static)
        for (int i = 1; i < Nx - 1; i++) {
            for (int j = 1; j < Ny - 1; j++) {
                newa[i * Ny + j] =
                    0.25 * (old[(i - 1) * Ny + j] +
                            old[(i + 1) * Ny + j] +
                            old[i * Ny + (j - 1)] +
                            old[i * Ny + (j + 1)]);
            }
        }

        double *tmp = old;
        old = newa;
        newa = tmp;
    }

    double t1 = omp_get_wtime();
    double sum = checksum(old, Nx, Ny);

    printf("Nx=%d Ny=%d steps=%d threads=%d time=%.6f s checksum=%.6f\n",
           Nx, Ny, steps, omp_get_max_threads(), t1 - t0, sum);

    free(old);
    free(newa);
    return 0;
}
