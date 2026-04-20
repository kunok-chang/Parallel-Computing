// ============================================================
// 벡터 덧셈 - CUDA (GPU)
// 핵심 개념:
//   1. cudaMalloc   : GPU 메모리 할당
//   2. cudaMemcpy   : CPU → GPU, GPU → CPU 데이터 전송
//   3. kernel<<<grid, block>>> : GPU 함수(kernel) 실행
//   4. 각 thread 가 자신의 인덱스(idx)로 딱 하나의 원소를 처리
// ============================================================
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// GPU에서 실행되는 함수 = "kernel"
// __global__ : CPU가 호출하고, GPU에서 실행됨
__global__ void vec_add_kernel(const float *A, const float *B, float *C, int N) {
    // 이 thread 가 담당할 원소 인덱스
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N)          // 배열 범위 초과 방지
        C[idx] = A[idx] + B[idx];
}

int main(void) {
    const int N = 1 << 24;
    size_t bytes = N * sizeof(float);

    // ── CPU(host) 메모리 ────────────────────────────────────
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);

    for (int i = 0; i < N; i++) { h_A[i] = 1.0f; h_B[i] = 2.0f; }

    // ── GPU(device) 메모리 할당 ─────────────────────────────
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // ── CPU → GPU 복사 ──────────────────────────────────────
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // ── kernel 실행 ─────────────────────────────────────────
    int block = 256;                        // 블록당 thread 수
    int grid  = (N + block - 1) / block;   // 필요한 블록 수

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    vec_add_kernel<<<grid, block>>>(d_A, d_B, d_C, N);   // ← kernel 호출

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);            // GPU 작업 완료 대기

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    // ── GPU → CPU 복사 ──────────────────────────────────────
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    printf("[cuda]    N=%d  kernel=%.4f s  C[0]=%.1f\n", N, ms / 1000.0f, h_C[0]);

    // ── 메모리 해제 ─────────────────────────────────────────
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
    return 0;
}
