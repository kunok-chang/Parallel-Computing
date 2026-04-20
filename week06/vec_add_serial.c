// ============================================================
// 벡터 덧셈 - Serial (CPU 단일 스레드)
// 핵심: C[i] = A[i] + B[i]  를 for loop 으로 순차 실행
// ============================================================
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(void) {
    const int N = 1 << 24;  // 16,777,216 개 원소 (~64 MB x3)

    float *A = malloc(N * sizeof(float));
    float *B = malloc(N * sizeof(float));
    float *C = malloc(N * sizeof(float));

    // 초기화
    for (int i = 0; i < N; i++) {
        A[i] = 1.0f;
        B[i] = 2.0f;
    }

    // ── 핵심 계산 ──────────────────────────────────────────
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    for (int i = 0; i < N; i++)
        C[i] = A[i] + B[i];   // ← 이 한 줄이 전부

    clock_gettime(CLOCK_MONOTONIC, &t1);
    // ───────────────────────────────────────────────────────

    double sec = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    printf("[serial]  N=%d  time=%.4f s  C[0]=%.1f\n", N, sec, C[0]);

    free(A); free(B); free(C);
    return 0;
}
