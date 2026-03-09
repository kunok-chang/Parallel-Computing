#include <stdio.h>
#include <omp.h>

int main() {
    const long N = 200000000;
    double sum = 0.0;

    double t0 = omp_get_wtime();

    #pragma omp parallel for reduction(+:sum)
    for (long i = 0; i < N; i++) {
        sum += 1.0;
    }

    double t1 = omp_get_wtime();

    printf("Elapsed = %.6f sec\n", t1 - t0);

    return 0;
}
