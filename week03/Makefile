CC=gcc
CFLAGS=-O2 -Wall -Wextra -std=c11 -fopenmp

TARGETS=counter_race counter_critical counter_atomic counter_reduction barrier_test nowait_bug scaling_sync

all: $(TARGETS)

counter_race: counter_race.c
	$(CC) $(CFLAGS) $< -o $@

counter_critical: counter_critical.c
	$(CC) $(CFLAGS) $< -o $@

counter_atomic: counter_atomic.c
	$(CC) $(CFLAGS) $< -o $@

counter_reduction: counter_reduction.c
	$(CC) $(CFLAGS) $< -o $@

barrier_test: barrier_test.c
	$(CC) $(CFLAGS) $< -o $@

nowait_bug: nowait_bug.c
	$(CC) $(CFLAGS) $< -o $@

scaling_sync: scaling_sync.c
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -f $(TARGETS)

.PHONY: all clean
