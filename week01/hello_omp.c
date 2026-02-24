#include <omp.h>
#include <stdio.h>

int main(void) {
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nt  = omp_get_num_threads();
        printf("Hello from thread %d / %d\n", tid, nt);
    }
    return 0;
}
