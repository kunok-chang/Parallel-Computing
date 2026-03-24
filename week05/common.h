#ifndef COMMON_H
#define COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static inline double *alloc_grid(int Nx, int Ny)
{
    double *a = (double *)malloc((size_t)Nx * (size_t)Ny * sizeof(double));
    if (!a) {
        fprintf(stderr, "Allocation failed for %d x %d grid\n", Nx, Ny);
        exit(1);
    }
    return a;
}

static inline void init_grid(double *a, int Nx, int Ny)
{
    for (int i = 0; i < Nx; i++) {
        for (int j = 0; j < Ny; j++) {
            double x = (double)i / (double)(Nx > 1 ? Nx - 1 : 1);
            double y = (double)j / (double)(Ny > 1 ? Ny - 1 : 1);
            a[(size_t)i * Ny + j] = sin(2.0 * M_PI * x) * cos(2.0 * M_PI * y);
        }
    }
}

static inline void zero_grid(double *a, int Nx, int Ny)
{
    memset(a, 0, (size_t)Nx * (size_t)Ny * sizeof(double));
}

static inline void apply_dirichlet_bc(double *a, int Nx, int Ny, double value)
{
    for (int i = 0; i < Nx; i++) {
        a[(size_t)i * Ny + 0] = value;
        a[(size_t)i * Ny + (Ny - 1)] = value;
    }
    for (int j = 0; j < Ny; j++) {
        a[(size_t)0 * Ny + j] = value;
        a[(size_t)(Nx - 1) * Ny + j] = value;
    }
}

static inline double checksum_grid(const double *a, int Nx, int Ny)
{
    double s = 0.0;
    for (int i = 0; i < Nx; i++) {
        for (int j = 0; j < Ny; j++) {
            s += a[(size_t)i * Ny + j];
        }
    }
    return s;
}

static inline double wall_seconds(void)
{
    return omp_get_wtime();
}

#endif
