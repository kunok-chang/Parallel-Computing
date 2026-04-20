CC = gcc
CFLAGS = -O3 -march=native -Wall -Wextra
OMPFLAGS = -fopenmp

CPU_TARGETS = vec_add_cpu vec_add_openmp stencil_serial stencil_openmp
TARGETS = $(CPU_TARGETS)

ifeq ($(shell command -v nvcc >/dev/null 2>&1; echo $$?),0)
NVCC = nvcc
NVFLAGS = -O3
CUDA_TARGETS = vec_add_cuda stencil_cuda
TARGETS += $(CUDA_TARGETS)
endif

all: $(TARGETS)

vec_add_cpu: vec_add_cpu.c
	$(CC) $(CFLAGS) $< -o $@

vec_add_openmp: vec_add_openmp.c
	$(CC) $(CFLAGS) $(OMPFLAGS) $< -o $@

stencil_serial: stencil_serial.c
	$(CC) $(CFLAGS) $< -o $@

stencil_openmp: stencil_openmp.c
	$(CC) $(CFLAGS) $(OMPFLAGS) $< -o $@

vec_add_cuda: vec_add_cuda.cu
	$(NVCC) $(NVFLAGS) $< -o $@

stencil_cuda: stencil_cuda.cu
	$(NVCC) $(NVFLAGS) $< -o $@

cpu: $(CPU_TARGETS)

cuda: $(CUDA_TARGETS)

clean:
	rm -f $(CPU_TARGETS) $(CUDA_TARGETS)
