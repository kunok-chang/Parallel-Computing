#include "common.h"

static void good_order(double *newa, const double *olda, int Nx, int Ny)
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

static void bad_order(double *newa, const double *olda, int Nx, int Ny)
{
    #pragma omp parallel for schedule(static)
    for (int j = 1; j < Ny - 1; j++) {
        for (int i = 1; i < Nx - 1; i++) {
            size_t idx = (size_t)i * Ny + j;
            newa[idx] = 0.25 * (
                olda[(size_t)(i - 1) * Ny + j] + olda[(size_t)(i + 1) * Ny + j] +
                olda[(size_t)i * Ny + (j - 1)] + olda[(size_t)i * Ny + (j + 1)]);
        }
    }
}

int main(int argc, char **argv)
{
    int Nx = (argc > 1) ? atoi(argv[1]) : 2048;
    int Ny = (argc > 2) ? atoi(argv[2]) : 2048;
    int steps = (argc > 3) ? atoi(argv[3]) : 100;

    double *a = alloc_grid(Nx, Ny);
    double *b = alloc_grid(Nx, Ny);
    init_grid(a, Nx, Ny);
    zero_grid(b, Nx, Ny);

    double t0 = wall_seconds();
    for (int t = 0; t < steps; t++) {
        apply_dirichlet_bc(b, Nx, Ny, 0.0);
        good_order(b, a, Nx, Ny);
        double *tmp = a; a = b; b = tmp;
    }
    double t1 = wall_seconds();
    double sum_good = checksum_grid(a, Nx, Ny);

    init_grid(a, Nx, Ny);
    zero_grid(b, Nx, Ny);
    double t2 = wall_seconds();
    for (int t = 0; t < steps; t++) {
        apply_dirichlet_bc(b, Nx, Ny, 0.0);
        bad_order(b, a, Nx, Ny);
        double *tmp = a; a = b; b = tmp;
    }
    double t3 = wall_seconds();
    double sum_bad = checksum_grid(a, Nx, Ny);

    printf("good_order time=%.6f checksum=%.12e\n", t1 - t0, sum_good);
    printf("bad_order  time=%.6f checksum=%.12e\n", t3 - t2, sum_bad);

    free(a);
    free(b);
    return 0;
}
