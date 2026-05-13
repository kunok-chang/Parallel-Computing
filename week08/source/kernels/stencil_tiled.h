#pragma once
#include <cuda_runtime.h>

void launch_stencil_tiled(const float *d_in, float *d_out,
                           int Nx, int Ny);
