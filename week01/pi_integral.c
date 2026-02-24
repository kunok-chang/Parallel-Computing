#define _GNU_SOURCE
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

static long parse_long(const char* s) {
    char* end = NULL;
    long v = strtol(s, &end, 10);
    if (!s || *s == '\0' || (end && *end != '\0') || v <= 0) {
        fprintf(stderr, "Invalid N: %s\nUsage: ./pi_integral <N>\n", s ? s : "(null)");
        exit(1);
    }
    return v;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <N>\nExample: %s 200000000\n", argv[0], argv[0]);
        return 1;
    }

    const long N = parse_long(argv[1]);

    double t0 = omp_get_wtime();

    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum) schedule(static)
    for (long i = 0; i < N; i++) {
        double x = (i + 0.5) / (double)N;
        sum += 4.0 / (1.0 + x * x);
    }

    double pi = sum / (double)N;
    double t1 = omp_get_wtime();

    int nt = 1;
    #pragma omp parallel
    {
        #pragma omp single
        nt = omp_get_num_threads();
    }

    printf("N=%ld, threads=%d\n", N, nt);
    printf("pi=%.15f\n", pi);
    printf("time=%.6f sec\n", t1 - t0);

    return 0;
}
