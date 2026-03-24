#include "common.h"

int main(int argc, char **argv)
{
    int Nx = (argc > 1) ? atoi(argv[1]) : 1024;
    int Ny = (argc > 2) ? atoi(argv[2]) : 1024;
    int steps = (argc > 3) ? atoi(argv[3]) : 500;

    double *olda = alloc_grid(Nx, Ny);
    double *newa = alloc_grid(Nx, Ny);
    init_grid(olda, Nx, Ny);
    zero_grid(newa, Nx, Ny);

    double t0 = wall_seconds();
    #pragma omp parallel
    {
        for (int t = 0; t < steps; t++) {
            #pragma omp single
            {
                apply_dirichlet_bc(newa, Nx, Ny, 0.0);
            }

            #pragma omp for collapse(2) schedule(static)
            for (int i = 1; i < Nx - 1; i++) {
                for (int j = 1; j < Ny - 1; j++) {
                    size_t idx = (size_t)i * Ny + j;
                    newa[idx] = 0.25 * (
                        olda[(size_t)(i - 1) * Ny + j] + olda[(size_t)(i + 1) * Ny + j] +
                        olda[(size_t)i * Ny + (j - 1)] + olda[(size_t)i * Ny + (j + 1)]);
                }
            }

            #pragma omp single
            {
                double *tmp = olda; olda = newa; newa = tmp;
            }

            #pragma omp barrier
        }
    }
    double t1 = wall_seconds();

    printf("diffusion_good_region Nx=%d Ny=%d steps=%d threads=%d time=%.6f checksum=%.12e\n",
           Nx, Ny, steps, omp_get_max_threads(), t1 - t0, checksum_grid(olda, Nx, Ny));

    free(olda);
    free(newa);
    return 0;
}
