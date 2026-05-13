#pragma once
#include <cuda_runtime.h>

void launch_stencil_naive(const float *d_in, float *d_out,
                           int Nx, int Ny,
                           dim3 block, dim3 grid);
