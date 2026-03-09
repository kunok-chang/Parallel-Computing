CC=gcc
CFLAGS=-O2 -Wall -Wextra -std=c11 -fopenmp

TARGETS=race_sum reduction_sum private_test firstprivate_test

all: $(TARGETS)

race_sum: race_sum.c
	$(CC) $(CFLAGS) $< -o $@

reduction_sum: reduction_sum.c
	$(CC) $(CFLAGS) $< -o $@

private_test: private_test.c
	$(CC) $(CFLAGS) $< -o $@

firstprivate_test: firstprivate_test.c
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -f $(TARGETS)

.PHONY: all clean
