// ============================================================
// 2D Stencil (5-point Jacobi) - CUDA
// 핵심: 각 thread 가 격자의 (i, j) 하나씩 담당
//       shared memory 없이 global memory 만 사용 (구조 이해 우선)
//
// thread 인덱스 → 격자 좌표 변환:
//   j = blockIdx.x * blockDim.x + threadIdx.x  (열)
//   i = blockIdx.y * blockDim.y + threadIdx.y  (행)
// ============================================================
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#define NX    2048
#define NY    2048
#define ITER  200

// GPU kernel: thread (i,j) 하나가 격자 점 하나를 계산
__global__ void stencil_kernel(const float *old, float *nw, int nx, int ny) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;  // 열 인덱스
    int i = blockIdx.y * blockDim.y + threadIdx.y;  // 행 인덱스

    // 경계 제외, 유효 범위 내부만 계산
    if (i < 1 || i >= nx - 1 || j < 1 || j >= ny - 1) return;

    nw[i*ny+j] = 0.2f * (old[i*ny+j]
               + old[(i-1)*ny+j] + old[(i+1)*ny+j]   // 상, 하
               + old[i*ny+(j-1)] + old[i*ny+(j+1)]); // 좌, 우
}

static void init(float *a) {
    for (int i = 0; i < NX; i++)
        for (int j = 0; j < NY; j++)
            a[i * NY + j] = (i == 0 || j == 0 || i == NX-1 || j == NY-1)
                             ? 1.0f : 0.0f;
}

int main(void) {
    size_t bytes = NX * NY * sizeof(float);

    // CPU 메모리
    float *h_old = (float*)malloc(bytes);
    init(h_old);

    // GPU 메모리 할당 + 복사
    float *d_old, *d_nw;
    cudaMalloc(&d_old, bytes);
    cudaMalloc(&d_nw,  bytes);
    cudaMemcpy(d_old, h_old, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_nw,  h_old, bytes, cudaMemcpyHostToDevice);

    // 2D 블록/그리드 설정
    dim3 block(16, 16);                                    // 16x16 = 256 threads/block
    dim3 grid((NY + block.x - 1) / block.x,
              (NX + block.y - 1) / block.y);

    // ── 핵심 계산 ──────────────────────────────────────────
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (int step = 0; step < ITER; step++) {
        stencil_kernel<<<grid, block>>>(d_old, d_nw, NX, NY);   // kernel 호출
        float *tmp = d_old; d_old = d_nw; d_nw = tmp;           // 포인터 스왑
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);   // GPU 완료 대기
    // ───────────────────────────────────────────────────────

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    // 결과 확인용: 중앙값 가져오기
    cudaMemcpy(h_old, d_old, bytes, cudaMemcpyDeviceToHost);
    printf("[cuda]    %dx%d  iters=%d  time=%.4f s  center=%.6f\n",
           NX, NY, ITER, ms / 1000.0f, h_old[(NX/2)*NY + NY/2]);

    cudaFree(d_old); cudaFree(d_nw);
    free(h_old);
    return 0;
}
