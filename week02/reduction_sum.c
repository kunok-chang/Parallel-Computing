#include <stdio.h>
#include <omp.h>

int main() {
    long N = 100000000;
    double sum = 0.0;
    
    #pragma omp parallel for reduction(+:sum)
    for (long i = 0; i < N; i++) {
        sum +=1.0;
    }

    printf("Sum = %.0f (expected %.0f)\n", sum, (double)N);
    return 0;
}
