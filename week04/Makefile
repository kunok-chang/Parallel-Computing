CC=gcc
CFLAGS=-O2 -Wall -Wextra -std=c11 -fopenmp

TARGETS=collapse_2d sections_demo task_demo task_overhead diffusion2d mc_pi

all: $(TARGETS)

collapse_2d: collapse_2d.c
	$(CC) $(CFLAGS) $< -o $@

sections_demo: sections_demo.c
	$(CC) $(CFLAGS) $< -o $@

task_demo: task_demo.c
	$(CC) $(CFLAGS) $< -o $@

task_overhead: task_overhead.c
	$(CC) $(CFLAGS) $< -o $@

diffusion2d: diffusion2d.c
	$(CC) $(CFLAGS) $< -o $@

mc_pi: mc_pi.c
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -f $(TARGETS)

.PHONY: all clean
