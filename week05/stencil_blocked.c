#include "common.h"

static void stencil_plain(double *newa, const double *olda, int Nx, int Ny)
{
    #pragma omp parallel for schedule(static)
    for (int i = 1; i < Nx - 1; i++) {
        for (int j = 1; j < Ny - 1; j++) {
            size_t idx = (size_t)i * Ny + j;
            newa[idx] = 0.25 * (
                olda[(size_t)(i - 1) * Ny + j] + olda[(size_t)(i + 1) * Ny + j] +
                olda[(size_t)i * Ny + (j - 1)] + olda[(size_t)i * Ny + (j + 1)]);
        }
    }
}

static void stencil_blocked(double *newa, const double *olda, int Nx, int Ny, int B)
{
    #pragma omp parallel for collapse(2) schedule(static)
    for (int ii = 1; ii < Nx - 1; ii += B) {
        for (int jj = 1; jj < Ny - 1; jj += B) {
            int iend = (ii + B < Nx - 1) ? ii + B : Nx - 1;
            int jend = (jj + B < Ny - 1) ? jj + B : Ny - 1;
            for (int i = ii; i < iend; i++) {
                for (int j = jj; j < jend; j++) {
                    size_t idx = (size_t)i * Ny + j;
                    newa[idx] = 0.25 * (
                        olda[(size_t)(i - 1) * Ny + j] + olda[(size_t)(i + 1) * Ny + j] +
                        olda[(size_t)i * Ny + (j - 1)] + olda[(size_t)i * Ny + (j + 1)]);
                }
            }
        }
    }
}

int main(int argc, char **argv)
{
    int Nx = (argc > 1) ? atoi(argv[1]) : 2048;
    int Ny = (argc > 2) ? atoi(argv[2]) : 2048;
    int steps = (argc > 3) ? atoi(argv[3]) : 100;
    int B = (argc > 4) ? atoi(argv[4]) : 32;

    double *a = alloc_grid(Nx, Ny);
    double *b = alloc_grid(Nx, Ny);
    init_grid(a, Nx, Ny);
    zero_grid(b, Nx, Ny);

    double t0 = wall_seconds();
    for (int t = 0; t < steps; t++) {
        apply_dirichlet_bc(b, Nx, Ny, 0.0);
        stencil_plain(b, a, Nx, Ny);
        double *tmp = a; a = b; b = tmp;
    }
    double t1 = wall_seconds();
    double sum_plain = checksum_grid(a, Nx, Ny);

    init_grid(a, Nx, Ny);
    zero_grid(b, Nx, Ny);
    double t2 = wall_seconds();
    for (int t = 0; t < steps; t++) {
        apply_dirichlet_bc(b, Nx, Ny, 0.0);
        stencil_blocked(b, a, Nx, Ny, B);
        double *tmp = a; a = b; b = tmp;
    }
    double t3 = wall_seconds();
    double sum_blocked = checksum_grid(a, Nx, Ny);

    printf("plain   time=%.6f checksum=%.12e\n", t1 - t0, sum_plain);
    printf("blocked time=%.6f checksum=%.12e block=%d\n", t3 - t2, sum_blocked, B);

    free(a);
    free(b);
    return 0;
}
