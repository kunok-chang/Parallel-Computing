#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

static void init_arrays(double *B, double *C, int Nx, int Ny) {
    long long N = (long long)Nx * Ny;
    for (long long i = 0; i < N; i++) {
        B[i] = 1.0 + (double)(i % 100) * 0.001;
        C[i] = 2.0 + (double)(i % 200) * 0.001;
    }
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s Nx Ny [collapse]\n", argv[0]);
        return 1;
    }

    int Nx = atoi(argv[1]);
    int Ny = atoi(argv[2]);
    int use_collapse = (argc >= 4 && strcmp(argv[3], "collapse") == 0);

    long long N = (long long)Nx * Ny;
    double *A = (double *)malloc(sizeof(double) * N);
    double *B = (double *)malloc(sizeof(double) * N);
    double *C = (double *)malloc(sizeof(double) * N);

    if (!A || !B || !C) {
        fprintf(stderr, "Allocation failed\n");
        free(A); free(B); free(C);
        return 1;
    }

    init_arrays(B, C, Nx, Ny);

    double t0 = omp_get_wtime();

    if (use_collapse) {
        #pragma omp parallel for collapse(2) schedule(static)
        for (int i = 0; i < Nx; i++) {
            for (int j = 0; j < Ny; j++) {
                long long idx = (long long)i * Ny + j;
                double x = B[idx];
                double y = C[idx];

                for (int k = 0; k < 20; k++) {
                    x = x * 1.0000001 + y * 0.9999999;
                }

                A[idx] = x;
            }
        }
    } else {
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < Nx; i++) {
            for (int j = 0; j < Ny; j++) {
                long long idx = (long long)i * Ny + j;
                double x = B[idx];
                double y = C[idx];

                for (int k = 0; k < 20; k++) {
                    x = x * 1.0000001 + y * 0.9999999;
                }

                A[idx] = x;
            }
        }
    }

    double t1 = omp_get_wtime();

    double checksum = 0.0;
    for (long long i = 0; i < N; i++) checksum += A[i];

    printf("Nx=%d Ny=%d mode=%s threads=%d time=%.6f s checksum=%.6f\n",
           Nx, Ny,
           use_collapse ? "collapse(2)" : "no-collapse",
           omp_get_max_threads(),
           t1 - t0, checksum);

    free(A);
    free(B);
    free(C);
    return 0;
}
