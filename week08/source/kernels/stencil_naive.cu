/*
 * stencil_naive.cu
 * Baseline 2D 5-point stencil kernel (no shared memory).
 * Week 8 -- CUDA Kernel Optimization
 */

#include <stdio.h>
#include <stdlib.h>
#include "stencil_naive.h"

__global__ void stencil_naive_kernel(const float *in, float *out,
                                     int Nx, int Ny)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= 1 && x < Nx-1 && y >= 1 && y < Ny-1) {
        out[y*Nx + x] = 0.2f * (in[y*Nx + x]
                               + in[(y-1)*Nx + x]
                               + in[(y+1)*Nx + x]
                               + in[y*Nx + (x-1)]
                               + in[y*Nx + (x+1)]);
    }
}

void launch_stencil_naive(const float *d_in, float *d_out,
                           int Nx, int Ny,
                           dim3 block, dim3 grid)
{
    stencil_naive_kernel<<<grid, block>>>(d_in, d_out, Nx, Ny);
}
