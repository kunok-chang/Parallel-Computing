CC = gcc
NVCC = nvcc
CFLAGS = -O3 -march=native -Wall -Wextra
OMPFLAGS = -fopenmp
NVFLAGS = -O3

TARGETS = vec_add_cpu vec_add_openmp stencil_openmp vec_add_cuda stencil_cuda

all: $(TARGETS)

vec_add_cpu: vec_add_cpu.c
	$(CC) $(CFLAGS) $< -o $@

vec_add_openmp: vec_add_openmp.c
	$(CC) $(CFLAGS) $(OMPFLAGS) $< -o $@

stencil_openmp: stencil_openmp.c
	$(CC) $(CFLAGS) $(OMPFLAGS) $< -o $@

vec_add_cuda: vec_add_cuda.cu
	$(NVCC) $(NVFLAGS) $< -o $@

stencil_cuda: stencil_cuda.cu
	$(NVCC) $(NVFLAGS) $< -o $@

clean:
	rm -f $(TARGETS)
