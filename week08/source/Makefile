# Makefile -- Week 8: CUDA Kernel Optimization
# Builds: compare_bench
#
# Targets:
#   make             -- build everything
#   make clean       -- remove binaries
#   make run         -- build and run all benchmarks (N=1024)
#   make run_stencil -- run stencil benchmark only
#   make run_matmul  -- run GEMM benchmark only
#   make profile     -- profile with Nsight Compute
#
# Override tile size at build time:
#   make TILE=32

NVCC      = nvcc
ARCH      = -arch=sm_75        # Change to match your GPU (sm_80 for A100, sm_86 for RTX 30xx)
OPTFLAGS  = -O2
CFLAGS    = $(OPTFLAGS) $(ARCH)
DBGFLAGS  = -g -G              # Use with 'make debug'

TILE      ?= 16                # Default GEMM tile size (override: make TILE=32)

KERNEL_SRCS = kernels/stencil_naive.cu \
              kernels/stencil_tiled.cu \
              kernels/matmul_naive.cu  \
              kernels/matmul_tiled.cu

BENCH_SRCS  = benchmarks/compare_bench.cu \
              benchmarks/timing_harness.cu

INCLUDES    = -I kernels -I benchmarks

.PHONY: all clean run run_stencil run_matmul profile debug

all: compare_bench

compare_bench: $(KERNEL_SRCS) $(BENCH_SRCS)
	$(NVCC) $(CFLAGS) $(INCLUDES) \
	    -DTILE_SIZE=$(TILE) \
	    $^ -o $@ -lm

debug: $(KERNEL_SRCS) $(BENCH_SRCS)
	$(NVCC) $(DBGFLAGS) $(ARCH) $(INCLUDES) \
	    -DTILE_SIZE=$(TILE) \
	    $^ -o compare_bench_debug -lm

run: compare_bench
	./compare_bench all 1024

run_stencil: compare_bench
	./compare_bench stencil 1024

run_matmul: compare_bench
	./compare_bench matmul 1024

profile: compare_bench
	bash profiling/run_ncu.sh ./compare_bench all 1024

plot: results.csv
	python plot/plot_speedup.py results.csv

clean:
	rm -f compare_bench compare_bench_debug \
	      results.csv *.ncu-rep \
	      speedup_*.png
