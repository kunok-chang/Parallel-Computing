#!/usr/bin/env bash
set -e

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

echo "=== Vector addition: serial ==="
./vec_add_cpu 10000000

echo "=== Vector addition: OpenMP ==="
./vec_add_openmp 10000000

if [ -x ./vec_add_cuda ]; then
  echo "=== Vector addition: CUDA ==="
  ./vec_add_cuda 10000000
else
  echo "=== Vector addition: CUDA skipped (vec_add_cuda not built) ==="
fi

echo "=== Stencil: serial ==="
./stencil_serial 4096 4096 400

echo "=== Stencil: OpenMP ==="
./stencil_openmp 4096 4096 400

if [ -x ./stencil_cuda ]; then
  echo "=== Stencil: CUDA ==="
  ./stencil_cuda 4096 4096 400
else
  echo "=== Stencil: CUDA skipped (stencil_cuda not built) ==="
fi
