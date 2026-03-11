#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main() {

    int N = 1000000;

    int *A = malloc(sizeof(int) * N);
    int *B = malloc(sizeof(int) * N);

    #pragma omp parallel
    {

        #pragma omp for nowait
        for (int i = 0; i < N; i++)
            A[i] = i;

        #pragma omp for
        for (int i = 0; i < N; i++)
            B[i] = A[i] + 1;
    }

    int errors = 0;

    for (int i = 0; i < N; i++)
        if (B[i] != i + 1)
            errors++;

    printf("errors = %d\n", errors);

    free(A);
    free(B);

    return 0;
}
