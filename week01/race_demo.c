#include <omp.h>
#include <stdio.h>
#include <string.h>

int main(int argc, char** argv) {
    const long N = 10000000;
    int fix = (argc >= 2 && strcmp(argv[1], "--fix") == 0);

    double t0 = omp_get_wtime();
    long counter = 0;

    if (!fix) {
        // WRONG: race condition (counter is shared, increment is not atomic)
        #pragma omp parallel for
        for (long i = 0; i < N; i++) {
            counter++; // race
        }
    } else {
        // FIX: reduction
        #pragma omp parallel for reduction(+:counter)
        for (long i = 0; i < N; i++) {
            counter += 1;
        }
    }

    double t1 = omp_get_wtime();

    int nt = 1;
    #pragma omp parallel
    {
        #pragma omp single
        nt = omp_get_num_threads();
    }

    printf("threads=%d, N=%ld\n", nt, N);
    printf("counter=%ld (expected %ld)\n", counter, N);
    printf("time=%.6f sec\n", t1 - t0);

    if (!fix) {
        printf("Hint: run './race_demo --fix'\n");
    }
    return 0;
}
