// mc_pi.c
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s Nsamples\n", argv[0]);
        return 1;
    }

    long long N = atoll(argv[1]);
    long long hits = 0;

    double t0 = omp_get_wtime();

    #pragma omp parallel
    {
        unsigned int seed = 1234u + 17u * (unsigned int)omp_get_thread_num();
        long long local_hits = 0;

        #pragma omp for
        for (long long i = 0; i < N; i++) {
            double x = rand_r(&seed) / (double)RAND_MAX;
            double y = rand_r(&seed) / (double)RAND_MAX;
            if (x * x + y * y <= 1.0) {
                local_hits++;
            }
        }

        #pragma omp atomic
        hits += local_hits;
    }

    double t1 = omp_get_wtime();

    double pi = 4.0 * (double)hits / (double)N;
    printf("N=%lld threads=%d pi=%.10f time=%.6f s hits=%lld\n",
           N, omp_get_max_threads(), pi, t1 - t0, hits);

    return 0;
}
