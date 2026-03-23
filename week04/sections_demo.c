// sections_demo.c
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

static void busy_work(const char *name, long iters) {
    double sum = 0.0;
    int tid = omp_get_thread_num();

    for (long i = 0; i < iters; i++) {
        sum += (i % 100) * 0.000001;
    }

    printf("[%s] done by thread %d, sum=%.6f\n", name, tid, sum);
}

static void compute_flux(void) {
    busy_work("compute_flux", 120000000L);
}

static void update_temperature(void) {
    busy_work("update_temperature", 80000000L);
}

static void write_output(void) {
    busy_work("write_output", 40000000L);
}

int main(void) {
    double t0 = omp_get_wtime();

    #pragma omp parallel sections
    {
        #pragma omp section
        compute_flux();

        #pragma omp section
        update_temperature();

        #pragma omp section
        write_output();
    }

    double t1 = omp_get_wtime();
    printf("sections total time = %.6f s\n", t1 - t0);
    return 0;
}
