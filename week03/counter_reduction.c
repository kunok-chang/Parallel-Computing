#include <stdio.h>
#include <omp.h>

int main() {

    int counter = 0;
    int N = 100000000;

    double start = omp_get_wtime();

    #pragma omp parallel for reduction(+:counter)
    for (int i = 0; i < N; i++)
        counter++;

    double end = omp_get_wtime();

    printf("counter_reduction\n");
    printf("result = %d\n", counter);
    printf("time   = %f\n", end - start);

    return 0;
}
