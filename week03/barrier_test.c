#include <stdio.h>
#include <omp.h>

void heavy_work() {

    volatile double x = 0;

    for (long i = 0; i < 200000000; i++)
        x += i * 0.000001;
}

void light_work() {

    volatile double x = 0;

    for (long i = 0; i < 10000000; i++)
        x += i * 0.000001;
}

int main() {

    double start = omp_get_wtime();

    #pragma omp parallel
    {

        int tid = omp_get_thread_num();

        printf("Thread %d starting heavy work\n", tid);

        heavy_work();

        #pragma omp barrier

        printf("Thread %d passed barrier\n", tid);

        light_work();
    }

    double end = omp_get_wtime();

    printf("Total time = %f\n", end - start);

    return 0;
}
