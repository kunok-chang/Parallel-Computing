// ============================================================
// 2D Stencil (5-point Jacobi) - Serial
// 핵심: 격자(grid)의 각 점을 상하좌우 이웃의 평균으로 갱신
//       이 반복 계산이 많을수록 병렬화 효과가 커짐
// ============================================================
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define NX    2048   // 격자 행 수
#define NY    2048   // 격자 열 수
#define ITER  200    // 반복 횟수

static void init(float *a) {
    for (int i = 0; i < NX; i++)
        for (int j = 0; j < NY; j++)
            a[i * NY + j] = (i == 0 || j == 0 || i == NX-1 || j == NY-1)
                             ? 1.0f : 0.0f;   // 경계는 1, 내부는 0
}

int main(void) {
    float *old = malloc(NX * NY * sizeof(float));
    float *nw  = malloc(NX * NY * sizeof(float));

    init(old);
    memcpy(nw, old, NX * NY * sizeof(float));

    // ── 핵심 계산 ──────────────────────────────────────────
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    for (int step = 0; step < ITER; step++) {
        for (int i = 1; i < NX - 1; i++)
            for (int j = 1; j < NY - 1; j++)
                nw[i*NY+j] = 0.2f * (old[i*NY+j]
                            + old[(i-1)*NY+j] + old[(i+1)*NY+j]  // 상, 하
                            + old[i*NY+(j-1)] + old[i*NY+(j+1)]); // 좌, 우

        float *tmp = old; old = nw; nw = tmp;  // 포인터 스왑 (memcpy 없이)
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    // ───────────────────────────────────────────────────────

    double sec = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    printf("[serial]  %dx%d  iters=%d  time=%.4f s  center=%.6f\n",
           NX, NY, ITER, sec, old[(NX/2)*NY + NY/2]);

    free(old); free(nw);
    return 0;
}
