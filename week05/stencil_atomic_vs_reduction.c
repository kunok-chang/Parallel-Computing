#include "common.h"

static double residual_atomic(const double *a, const double *b, long N)
{
    double sum = 0.0;
    double t0 = wall_seconds();
    #pragma omp parallel for
    for (long i = 0; i < N; i++) {
        double diff = a[i] - b[i];
        #pragma omp atomic
        sum += diff * diff;
    }
    double t1 = wall_seconds();
    printf("atomic    time=%.6f residual=%.12e\n", t1 - t0, sum);
    return sum;
}

static double residual_reduction(const double *a, const double *b, long N)
{
    double sum = 0.0;
    double t0 = wall_seconds();
    #pragma omp parallel for reduction(+:sum)
    for (long i = 0; i < N; i++) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    double t1 = wall_seconds();
    printf("reduction time=%.6f residual=%.12e\n", t1 - t0, sum);
    return sum;
}

int main(int argc, char **argv)
{
    long N = (argc > 1) ? atol(argv[1]) : 100000000;
    double *a = (double *)malloc((size_t)N * sizeof(double));
    double *b = (double *)malloc((size_t)N * sizeof(double));
    if (!a || !b) exit(1);

    #pragma omp parallel for
    for (long i = 0; i < N; i++) {
        a[i] = sin(0.000001 * (double)i);
        b[i] = cos(0.000001 * (double)i);
    }

    residual_atomic(a, b, N);
    residual_reduction(a, b, N);

    free(a);
    free(b);
    return 0;
}
