# CUDA Introduction: CPU, OpenMP, and CUDA Comparison

This repository is designed for the first CUDA class.
Students can compile and run simple examples, then compare serial CPU, OpenMP, and CUDA performance.

## Files

- `vec_add_cpu.c`: serial CPU vector addition baseline
- `vec_add_openmp.c`: OpenMP vector addition baseline
- `vec_add_cuda.cu`: CUDA vector addition
- `stencil_openmp.c`: OpenMP 2D stencil (FDM-style update)
- `stencil_cuda.cu`: CUDA 2D stencil
- `Makefile`: build targets
- `run_examples.sh`: example run script

## Learning goals

- Understand host-device workflow in CUDA
- Compare serial CPU, OpenMP, and CUDA execution
- Observe that GPU speedup depends on problem size
- Connect CUDA examples to previous OpenMP/FDM lessons

## Requirements

### For CPU/OpenMP
- GCC with OpenMP support

### For CUDA on WSL
- WSL2 installed on Windows
- NVIDIA Windows driver with WSL CUDA support
- CUDA Toolkit inside Ubuntu

## Build

```bash
make
```

If `nvcc` is not installed yet, CPU/OpenMP examples can still be compiled individually:

```bash
gcc -O3 -march=native vec_add_cpu.c -o vec_add_cpu
gcc -O3 -march=native -fopenmp vec_add_openmp.c -o vec_add_openmp
gcc -O3 -march=native -fopenmp stencil_openmp.c -o stencil_openmp
```

## Run examples

### Vector addition

```bash
./vec_add_cpu 10000000
export OMP_NUM_THREADS=8
./vec_add_openmp 10000000
./vec_add_cuda 10000000
```

Try larger sizes too:

```bash
./vec_add_cpu 100000000
./vec_add_openmp 100000000
./vec_add_cuda 100000000
```

### 2D stencil

```bash
export OMP_NUM_THREADS=8
./stencil_openmp 2048 2048 200
./stencil_cuda 2048 2048 200
```

## What to discuss in class

### Vector addition
- CUDA syntax is simple to understand
- Small problems may not benefit from GPU because copy overhead matters
- Large problems make GPU parallelism more worthwhile

### 2D stencil
- This is closer to finite difference workloads
- OpenMP and CUDA solve the same parallel problem on different hardware
- Compare total runtime, not only kernel time

## Suggested report questions

1. Does CUDA always beat OpenMP?
2. At what problem size does CUDA become advantageous?
3. Why is total CUDA time different from kernel-only time?
4. Why is stencil a better bridge example than vector addition?

## Notes

- CUDA timings in these examples report end-to-end GPU time including memory copies.
- Checksums are printed for quick correctness checks.
- These are intentionally simple first-week teaching codes, not fully optimized production kernels.
