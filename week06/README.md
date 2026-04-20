# Week 06 — Serial / OpenMP / CUDA 비교 실습

## 파일 구성

| 파일 | 설명 |
|---|---|
| `vec_add_serial.c`  | 벡터 덧셈 — CPU 단일 스레드 |
| `vec_add_openmp.c`  | 벡터 덧셈 — OpenMP 병렬 |
| `vec_add_cuda.cu`   | 벡터 덧셈 — CUDA GPU |
| `stencil_serial.c`  | 2D Stencil — CPU 단일 스레드 |
| `stencil_openmp.c`  | 2D Stencil — OpenMP 병렬 |
| `stencil_cuda.cu`   | 2D Stencil — CUDA GPU |

---

## 핵심 개념 요약

### 1. 벡터 덧셈 — 코드 구조 비교용

세 버전 모두 하는 일은 동일:

```
C[i] = A[i] + B[i]
```

| 버전 | 병렬화 방법 |
|---|---|
| serial  | for loop 순차 실행 |
| OpenMP  | `#pragma omp parallel for` 한 줄 추가 |
| CUDA    | GPU kernel 함수 + `<<<grid, block>>>` 호출 |

> vec_add 는 연산이 단순해서 GPU 가 빠르지 않을 수 있음.
> **목적은 속도 비교가 아니라 코드 구조 이해.**

---

### 2. 2D Stencil — 성능 비교용

격자 내 모든 점을 상하좌우 이웃의 평균으로 반복 갱신:

```
new[i][j] = 0.2 * (old[i][j] + old[i-1][j] + old[i+1][j]
                              + old[i][j-1] + old[i][j+1])
```

2048×2048 격자를 200번 반복 → 연산량이 충분해서 `serial < OpenMP < CUDA` 순서가 잘 나옴.

**CUDA 포인터 스왑 (메모리 복사 없이):**
```c
float *tmp = d_old; d_old = d_nw; d_nw = tmp;
```

---

## 빌드

```bash
make        # nvcc 있으면 CUDA 포함, 없으면 CPU/OpenMP 만 빌드
make clean
```

---

## 실행 & 성능 비교

### 벡터 덧셈
```bash
./vec_add_serial
OMP_NUM_THREADS=8 ./vec_add_openmp
./vec_add_cuda
```

### 2D Stencil (성능 비교)
```bash
./stencil_serial
OMP_NUM_THREADS=8 ./stencil_openmp
./stencil_cuda
```

### 예상 출력 예시
```
[serial]  2048x2048  iters=200  time=3.8000 s  center=0.xxxxxx
[openmp]  2048x2048  iters=200  threads=8  time=0.6000 s  center=0.xxxxxx
[cuda]    2048x2048  iters=200  time=0.0800 s  center=0.xxxxxx
```

---

## CUDA 핵심 흐름 (처음 보는 분들을 위해)

```
CPU                              GPU
───                              ───
malloc(h_A, h_B)
h_A, h_B 초기화

cudaMalloc(d_A, d_B, d_C)   →  GPU 메모리 할당
cudaMemcpy(H→D)              →  데이터 전송 (CPU→GPU)

kernel<<<grid,block>>>()     →  수만 개 thread 동시 실행
cudaDeviceSynchronize()      ←  GPU 완료 대기

cudaMemcpy(D→H)              ←  결과 전송 (GPU→CPU)
결과 사용
```

thread 인덱스 계산:
```c
int idx = blockIdx.x * blockDim.x + threadIdx.x;
```

2D kernel (stencil):
```c
int j = blockIdx.x * blockDim.x + threadIdx.x;  // 열
int i = blockIdx.y * blockDim.y + threadIdx.y;  // 행
```
