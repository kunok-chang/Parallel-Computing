/*
 * stencil_tiled.cu
 * Tiled 2D 5-point stencil using shared memory with halo loading.
 * Week 8 -- CUDA Kernel Optimization
 *
 * Students: complete the sections marked TODO.
 */

#include <stdio.h>
#include <stdlib.h>
#include "stencil_tiled.h"

#define TILE_X 16
#define TILE_Y 16
#define RADIUS 1

/*
 * Tiled stencil kernel.
 *
 * Each block loads a (TILE_X + 2*RADIUS) x (TILE_Y + 2*RADIUS) region
 * (the tile plus its halo) into shared memory, then computes the stencil
 * for the TILE_X x TILE_Y interior cells.
 *
 * Shared memory layout:
 *   s[ty][tx]  where ty, tx in [0, TILE_Y+2*RADIUS) x [0, TILE_X+2*RADIUS)
 *   Interior output cell at s[threadIdx.y + RADIUS][threadIdx.x + RADIUS]
 */
__global__ void stencil_tiled_kernel(const float *in, float *out,
                                     int Nx, int Ny)
{
    /* TODO 1: Declare shared memory with room for the halo.
     *         Size: (TILE_Y + 2*RADIUS) rows x (TILE_X + 2*RADIUS + 1) cols.
     *         The extra +1 column eliminates bank conflicts. */
    __shared__ float s[TILE_Y + 2*RADIUS][TILE_X + 2*RADIUS + 1];  /* +1 padding */

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    /* Global indices including halo offset */
    int gx = blockIdx.x * TILE_X + tx - RADIUS;
    int gy = blockIdx.y * TILE_Y + ty - RADIUS;

    /* TODO 2: Cooperatively load global memory -> shared memory.
     *         Handle out-of-bound indices by loading 0.0f. */
    s[ty][tx] = (gx >= 0 && gx < Nx && gy >= 0 && gy < Ny)
                ? in[gy * Nx + gx]
                : 0.0f;

    /* TODO 3: Synchronize to ensure all threads have finished loading. */
    __syncthreads();

    /* TODO 4: Compute the stencil only for interior threads.
     *         Interior condition: tx in [RADIUS, TILE_X+RADIUS)
     *                             ty in [RADIUS, TILE_Y+RADIUS)
     *         Also check that the output cell is within the grid. */
    if (tx >= RADIUS && tx < TILE_X + RADIUS &&
        ty >= RADIUS && ty < TILE_Y + RADIUS)
    {
        /* Map back to global output coordinates */
        int ox = gx;   /* = blockIdx.x * TILE_X + (tx - RADIUS) */
        int oy = gy;   /* = blockIdx.y * TILE_Y + (ty - RADIUS) */

        if (ox >= 1 && ox < Nx-1 && oy >= 1 && oy < Ny-1) {
            out[oy * Nx + ox] =
                0.2f * (s[ty][tx]
                      + s[ty-1][tx]
                      + s[ty+1][tx]
                      + s[ty][tx-1]
                      + s[ty][tx+1]);
        }
    }
}

void launch_stencil_tiled(const float *d_in, float *d_out,
                           int Nx, int Ny)
{
    /* Block size matches the tile interior; threads also cover the halo. */
    dim3 block(TILE_X + 2*RADIUS, TILE_Y + 2*RADIUS);
    dim3 grid((Nx + TILE_X - 1) / TILE_X,
              (Ny + TILE_Y - 1) / TILE_Y);

    stencil_tiled_kernel<<<grid, block>>>(d_in, d_out, Nx, Ny);
}
