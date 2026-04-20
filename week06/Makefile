CC      = gcc
CFLAGS  = -O2 -Wall
OMPFLAGS = -fopenmp

# CPU 타깃은 항상 빌드
CPU_TARGETS = vec_add_serial vec_add_openmp stencil_serial stencil_openmp
TARGETS = $(CPU_TARGETS)

# nvcc 가 있으면 CUDA 타깃도 빌드
ifneq ($(shell which nvcc 2>/dev/null),)
  NVCC = nvcc
  NVFLAGS = -O2
  CUDA_TARGETS = vec_add_cuda stencil_cuda
  TARGETS += $(CUDA_TARGETS)
endif

all: $(TARGETS)

vec_add_serial:   vec_add_serial.c
	$(CC) $(CFLAGS) $< -o $@

vec_add_openmp:   vec_add_openmp.c
	$(CC) $(CFLAGS) $(OMPFLAGS) $< -o $@

stencil_serial:   stencil_serial.c
	$(CC) $(CFLAGS) $< -o $@

stencil_openmp:   stencil_openmp.c
	$(CC) $(CFLAGS) $(OMPFLAGS) $< -o $@

vec_add_cuda:     vec_add_cuda.cu
	$(NVCC) $(NVFLAGS) $< -o $@

stencil_cuda:     stencil_cuda.cu
	$(NVCC) $(NVFLAGS) $< -o $@

clean:
	rm -f $(CPU_TARGETS) $(CUDA_TARGETS)
