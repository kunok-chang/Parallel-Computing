C=gcc
CFLAGS=-O2 -Wall -Wextra -std=c11 -fopenmp

TARGETS=hello_omp pi_integral race_demo

all: $(TARGETS)

hello_omp: hello_omp.c
    $(CC) $(CFLAGS) $< -o $@

pi_integral: pi_integral.c
    $(CC) $(CFLAGS) $< -o $@

race_demo: race_demo.c
    $(CC) $(CFLAGS) $< -o $@

clean:
    rm -f $(TARGETS)

.PHONY: all clean
