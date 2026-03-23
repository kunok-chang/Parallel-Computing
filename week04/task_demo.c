// task_demo.c
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

static void work(const char *name, long iters) {
    double x = 0.0;
    int tid = omp_get_thread_num();

    for (long i = 0; i < iters; i++) {
        x += (i % 97) * 1e-7;
    }

    printf("%s executed by thread %d, x=%.6f\n", name, tid, x);
}

static void finalize(void) {
    printf("finalize() by thread %d\n", omp_get_thread_num());
}

int main(void) {
    double t0 = omp_get_wtime();

    #pragma omp parallel
    {
        #pragma omp single
        {
            #pragma omp task
            work("work1", 100000000L);

            #pragma omp task
            work("work2", 120000000L);

            #pragma omp task
            work("work3", 80000000L);

            #pragma omp taskwait
            finalize();
        }
    }

    double t1 = omp_get_wtime();
    printf("task_demo total time = %.6f s\n", t1 - t0);
    return 0;
}
