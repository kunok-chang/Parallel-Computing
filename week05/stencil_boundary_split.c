#include "common.h"

static void stencil_boundary_split(double *newa, const double *olda, int Nx, int Ny)
{
    apply_dirichlet_bc(newa, Nx, Ny, 0.0);

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 1; i < Nx - 1; i++) {
        for (int j = 1; j < Ny - 1; j++) {
            size_t idx = (size_t)i * Ny + j;
            newa[idx] = 0.25 * (
                olda[(size_t)(i - 1) * Ny + j] +
                olda[(size_t)(i + 1) * Ny + j] +
                olda[(size_t)i * Ny + (j - 1)] +
                olda[(size_t)i * Ny + (j + 1)]);
        }
    }
}

int main(int argc, char **argv)
{
    int Nx = (argc > 1) ? atoi(argv[1]) : 1024;
    int Ny = (argc > 2) ? atoi(argv[2]) : 1024;
    int steps = (argc > 3) ? atoi(argv[3]) : 200;

    double *olda = alloc_grid(Nx, Ny);
    double *newa = alloc_grid(Nx, Ny);
    init_grid(olda, Nx, Ny);
    zero_grid(newa, Nx, Ny);

    double t0 = wall_seconds();
    for (int t = 0; t < steps; t++) {
        stencil_boundary_split(newa, olda, Nx, Ny);
        double *tmp = olda; olda = newa; newa = tmp;
    }
    double t1 = wall_seconds();

    printf("stencil_boundary_split Nx=%d Ny=%d steps=%d threads=%d time=%.6f checksum=%.12e\n",
           Nx, Ny, steps, omp_get_max_threads(), t1 - t0, checksum_grid(olda, Nx, Ny));

    free(olda);
    free(newa);
    return 0;
}
