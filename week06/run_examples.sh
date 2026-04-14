#!/usr/bin/env bash
set -e

echo "=== Vector addition: serial ==="
./vec_add_cpu 10000000

echo "=== Vector addition: OpenMP ==="
export OMP_NUM_THREADS=8
./vec_add_openmp 10000000

echo "=== Vector addition: CUDA ==="
./vec_add_cuda 10000000

echo "=== Stencil: OpenMP ==="
./stencil_openmp 2048 2048 200

echo "=== Stencil: CUDA ==="
./stencil_cuda 2048 2048 200
