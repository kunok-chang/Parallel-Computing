#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

static inline void heavy_work(long i) {
    if (i % 1000 == 0) {
        volatile long s = 0;
        for (long k = 0; k < 200000; k++) s += k;
    }
}

int main() {
    long N = 20000000;
    double t0 = omp_get_wtime();

    #pragma omp parallel for schedule(guided)
    for (long i = 0; i < N; i++) {
        heavy_work(i);
    }

    double t1 = omp_get_wtime();
    printf("Elapsed = %.6f sec\n", t1 - t0);
    return 0;
}
