#pragma once
#include <cuda_runtime.h>

void launch_matmul_naive(const float *d_A, const float *d_B,
                          float *d_C, int N);
