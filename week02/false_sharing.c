#include <stdio.h>
#include <omp.h>

int main() {
    const long N = 200000000;
    static double partial[64]; // assume <= 64 threads

    double t0 = omp_get_wtime();

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        for (long i = 0; i < N; i++) {
            partial[tid] += 1.0; // may false-share
        }
    }
    double t1 = omp_get_wtime();
    printf("Elapsed = %.6f sec\n", t1 - t0);
    return 0;
}
