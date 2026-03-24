#include "common.h"

static void neumann_bad(double *newa, const double *olda, int Nx, int Ny)
{
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < Nx; i++) {
        for (int j = 0; j < Ny; j++) {
            int up    = (i == 0)    ? i : i - 1;
            int down  = (i == Nx-1) ? i : i + 1;
            int left  = (j == 0)    ? j : j - 1;
            int right = (j == Ny-1) ? j : j + 1;
            newa[(size_t)i * Ny + j] = 0.25 * (
                olda[(size_t)up * Ny + j] + olda[(size_t)down * Ny + j] +
                olda[(size_t)i * Ny + left] + olda[(size_t)i * Ny + right]);
        }
    }
}

static void neumann_split(double *newa, const double *olda, int Nx, int Ny)
{
    /* Interior only */
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 1; i < Nx - 1; i++) {
        for (int j = 1; j < Ny - 1; j++) {
            newa[(size_t)i * Ny + j] = 0.25 * (
                olda[(size_t)(i - 1) * Ny + j] + olda[(size_t)(i + 1) * Ny + j] +
                olda[(size_t)i * Ny + (j - 1)] + olda[(size_t)i * Ny + (j + 1)]);
        }
    }

    for (int i = 1; i < Nx - 1; i++) {
        newa[(size_t)i * Ny + 0]      = newa[(size_t)i * Ny + 1];
        newa[(size_t)i * Ny + (Ny-1)] = newa[(size_t)i * Ny + (Ny-2)];
    }
    for (int j = 0; j < Ny; j++) {
        newa[(size_t)0 * Ny + j]      = newa[(size_t)1 * Ny + j];
        newa[(size_t)(Nx-1) * Ny + j] = newa[(size_t)(Nx-2) * Ny + j];
    }
}

int main(int argc, char **argv)
{
    int Nx = (argc > 1) ? atoi(argv[1]) : 1024;
    int Ny = (argc > 2) ? atoi(argv[2]) : 1024;
    int steps = (argc > 3) ? atoi(argv[3]) : 200;

    double *a = alloc_grid(Nx, Ny);
    double *b = alloc_grid(Nx, Ny);
    init_grid(a, Nx, Ny);
    zero_grid(b, Nx, Ny);

    double t0 = wall_seconds();
    for (int t = 0; t < steps; t++) {
        neumann_bad(b, a, Nx, Ny);
        double *tmp = a; a = b; b = tmp;
    }
    double t1 = wall_seconds();
    double sum_bad = checksum_grid(a, Nx, Ny);

    init_grid(a, Nx, Ny);
    zero_grid(b, Nx, Ny);
    double t2 = wall_seconds();
    for (int t = 0; t < steps; t++) {
        neumann_split(b, a, Nx, Ny);
        double *tmp = a; a = b; b = tmp;
    }
    double t3 = wall_seconds();
    double sum_split = checksum_grid(a, Nx, Ny);

    printf("neumann_bad   time=%.6f checksum=%.12e\n", t1 - t0, sum_bad);
    printf("neumann_split time=%.6f checksum=%.12e\n", t3 - t2, sum_split);
    return 0;
}
