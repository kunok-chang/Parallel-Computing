// ============================================================
// 2D Stencil (5-point Jacobi) - OpenMP
// 핵심:
//   - #pragma omp parallel  : 스레드 팀을 한 번만 생성 (반복문 밖)
//   - #pragma omp for       : 각 step 의 행(i) 분배
//   - #pragma omp single    : 포인터 스왑은 한 스레드만 수행
// ============================================================
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#define NX    2048
#define NY    2048
#define ITER  200

static void init(float *a) {
    for (int i = 0; i < NX; i++)
        for (int j = 0; j < NY; j++)
            a[i * NY + j] = (i == 0 || j == 0 || i == NX-1 || j == NY-1)
                             ? 1.0f : 0.0f;
}

int main(void) {
    float *old = malloc(NX * NY * sizeof(float));
    float *nw  = malloc(NX * NY * sizeof(float));

    init(old);
    memcpy(nw, old, NX * NY * sizeof(float));

    // ── 핵심 계산 ──────────────────────────────────────────
    double t0 = omp_get_wtime();

    #pragma omp parallel                    // ← 스레드 팀 생성 (한 번만)
    {
        for (int step = 0; step < ITER; step++) {

            #pragma omp for                 // ← 행(i)을 스레드들이 나눠서 처리
            for (int i = 1; i < NX - 1; i++)
                for (int j = 1; j < NY - 1; j++)
                    nw[i*NY+j] = 0.2f * (old[i*NY+j]
                                + old[(i-1)*NY+j] + old[(i+1)*NY+j]
                                + old[i*NY+(j-1)] + old[i*NY+(j+1)]);

            #pragma omp single              // ← 스왑은 한 스레드만 (나머지는 대기)
            { float *tmp = old; old = nw; nw = tmp; }

        }  // implicit barrier here: 모든 스레드가 이 step 끝날 때까지 대기
    }

    double t1 = omp_get_wtime();
    // ───────────────────────────────────────────────────────

    printf("[openmp]  %dx%d  iters=%d  threads=%d  time=%.4f s  center=%.6f\n",
           NX, NY, ITER, omp_get_max_threads(), t1 - t0, old[(NX/2)*NY + NY/2]);

    free(old); free(nw);
    return 0;
}
