CC=gcc
CFLAGS=-O3 -fopenmp -march=native -Wall -Wextra -std=c11
LDLIBS=-lm

TARGETS = \
	stencil_basic \
	stencil_bad_branch \
	stencil_boundary_split \
	stencil_collapse_compare \
	stencil_static_dynamic \
	stencil_false_sharing \
	stencil_atomic_vs_reduction \
	stencil_blocked \
	stencil_loop_order \
	stencil_neumann_bad \
	diffusion_bad_region \
	diffusion_good_region

all: $(TARGETS)

%: %.c common.h
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

clean:
	rm -f $(TARGETS)
