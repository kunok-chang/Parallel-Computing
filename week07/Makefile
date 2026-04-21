NVCC      = nvcc
ARCH      = -arch=sm_75        # change to match your GPU (sm_86 for RTX 30xx, etc.)
CFLAGS    = -O2 $(ARCH)
OMP       = -Xcompiler -fopenmp

BENCHDIR  = benchmarks

TARGETS   = bandwidth_bench stencil_bench

.PHONY: all clean

all: $(TARGETS)

bandwidth_bench: $(BENCHDIR)/bandwidth_bench.cu $(BENCHDIR)/timing_harness.cuh
	$(NVCC) $(CFLAGS) $(BENCHDIR)/bandwidth_bench.cu -o $@

stencil_bench: $(BENCHDIR)/stencil_bench.cu $(BENCHDIR)/timing_harness.cuh
	$(NVCC) $(CFLAGS) $(OMP) $(BENCHDIR)/stencil_bench.cu -o $@

clean:
	rm -f $(TARGETS) *.csv *.ncu-rep
