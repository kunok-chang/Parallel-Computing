#include <stdio.h>
#include <omp.h>

int main() {
    int x = 10;

    #pragma omp parallel private(x)
    {
        // x is uninitialized per thread
        printf("Thread %d: x = %d\n",
               omp_get_thread_num(), x);
    }
    return 0;
}
