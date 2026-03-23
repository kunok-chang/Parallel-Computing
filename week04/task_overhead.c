// task_overhead.c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

static inline double tiny_work(int cost) {
    double x = 0.0;
    for (int i = 0; i < cost; i++) {
        x += (i % 13) * 1e-6;
    }
    return x;
}

int main(int argc, char *argv[]) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s Ntasks cost mode(serial|task|cutoff)\n", argv[0]);
        return 1;
    }

    int Ntasks = atoi(argv[1]);
    int cost   = atoi(argv[2]);
    const char *mode = argv[3];

    double result = 0.0;
    double t0 = omp_get_wtime();

    if (strcmp(mode, "serial") == 0) {
        for (int i = 0; i < Ntasks; i++) {
            result += tiny_work(cost);
        }
    } else if (strcmp(mode, "task") == 0) {
        #pragma omp parallel
        {
            #pragma omp single
            {
                for (int i = 0; i < Ntasks; i++) {
                    #pragma omp task shared(result)
                    {
                        double val = tiny_work(cost);
                        #pragma omp atomic
                        result += val;
                    }
                }
                #pragma omp taskwait
            }
        }
    } else if (strcmp(mode, "cutoff") == 0) {
        int cutoff = 1000;  // example threshold

        #pragma omp parallel
        {
            #pragma omp single
            {
                for (int i = 0; i < Ntasks; i++) {
                    if (cost >= cutoff) {
                        #pragma omp task shared(result)
                        {
                            double val = tiny_work(cost);
                            #pragma omp atomic
                            result += val;
                        }
                    } else {
                        result += tiny_work(cost);
                    }
                }
                #pragma omp taskwait
            }
        }
    } else {
        fprintf(stderr, "Unknown mode: %s\n", mode);
        return 1;
    }

    double t1 = omp_get_wtime();

    printf("mode=%s Ntasks=%d cost=%d threads=%d time=%.6f s result=%.6f\n",
           mode, Ntasks, cost, omp_get_max_threads(), t1 - t0, result);

    return 0;
}
