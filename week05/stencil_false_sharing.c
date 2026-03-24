#include "common.h"

typedef struct {
    double value;
    char pad[64 - sizeof(double)];
} PaddedDouble;

static double bad_false_sharing(long N)
{
    int nt = omp_get_max_threads();
    double *partial = (double *)calloc((size_t)nt, sizeof(double));
    if (!partial) exit(1);

    double t0 = wall_seconds();
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        #pragma omp for
        for (long i = 0; i < N; i++) {
            partial[tid] += 1.0 / (1.0 + (double)(i % 97));
        }
    }
    double t1 = wall_seconds();

    double s = 0.0;
    for (int i = 0; i < nt; i++) s += partial[i];
    free(partial);
    printf("false_sharing_bad   time=%.6f sum=%.12e\n", t1 - t0, s);
    return t1 - t0;
}

static double good_padded(long N)
{
    int nt = omp_get_max_threads();
    PaddedDouble *partial = (PaddedDouble *)calloc((size_t)nt, sizeof(PaddedDouble));
    if (!partial) exit(1);

    double t0 = wall_seconds();
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        #pragma omp for
        for (long i = 0; i < N; i++) {
            partial[tid].value += 1.0 / (1.0 + (double)(i % 97));
        }
    }
    double t1 = wall_seconds();

    double s = 0.0;
    for (int i = 0; i < nt; i++) s += partial[i].value;
    free(partial);
    printf("false_sharing_good  time=%.6f sum=%.12e\n", t1 - t0, s);
    return t1 - t0;
}

int main(int argc, char **argv)
{
    long N = (argc > 1) ? atol(argv[1]) : 200000000;
    bad_false_sharing(N);
    good_padded(N);
    return 0;
}
