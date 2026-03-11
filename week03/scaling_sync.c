#include <stdio.h>
#include <omp.h>

int main() {

    int counter = 0;
    int N = 100000000;

    double start = omp_get_wtime();

    #pragma omp parallel for
    for (int i = 0; i < N; i++) {

        #pragma omp critical
        counter++;
    }

    double end = omp_get_wtime();

    printf("scaling_sync\n");
    printf("threads = %d\n", omp_get_max_threads());
    printf("time    = %f\n", end - start);

    return 0;
}
