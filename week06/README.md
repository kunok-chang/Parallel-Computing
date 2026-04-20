# CUDA Introduction: simple vec_add + performance-friendly stencil

## Files

- `vec_add_cpu.c` : serial vector addition
- `vec_add_openmp.c` : OpenMP vector addition
- `vec_add_cuda.cu` : CUDA vector addition
- `stencil_serial.c` : serial 2D stencil
- `stencil_openmp.c` : OpenMP 2D stencil
- `stencil_cuda.cu` : CUDA 2D stencil (shared-memory tiled kernel)
- `Makefile`
- `run_examples.sh`

## Key idea

### Vector addition

세 버전 모두 사실상 아래 한 줄입니다.

```c
C[i] = A[i] + B[i];
```

즉, 여기서는 속도보다
- serial for loop
- OpenMP `#pragma omp parallel for`
- CUDA kernel launch / memcpy

이 세 가지 구조 차이를 보여주는 데 목적이 있습니다.

### Stencil

stencil 코드는 실제 속도 차이가 보이도록 아래처럼 구성했습니다.

- 충분히 큰 기본 문제 크기: `4096 x 4096`
- 여러 iteration 반복: 기본 `400`
- serial: 순수 단일 스레드
- OpenMP: row-wise 정적 분할 + 반복 내부 pointer swap
- CUDA: shared-memory tiled 5-point stencil
- stencil 시간은 **핵심 반복 계산 구간** 기준으로 측정

## Build

```bash
make
```

- `nvcc`가 있으면 CUDA 타깃까지 같이 빌드됩니다.
- `nvcc`가 없으면 CPU/OpenMP 타깃만 빌드됩니다.

개별 빌드:

```bash
gcc -O3 -march=native vec_add_cpu.c -o vec_add_cpu
gcc -O3 -march=native -fopenmp vec_add_openmp.c -o vec_add_openmp
gcc -O3 -march=native stencil_serial.c -o stencil_serial
gcc -O3 -march=native -fopenmp stencil_openmp.c -o stencil_openmp
```

## Run examples

### 1) Simple vector add

```bash
./vec_add_cpu 10000000
export OMP_NUM_THREADS=8
./vec_add_openmp 10000000
./vec_add_cuda 10000000
```

### 2) Performance-oriented stencil

권장:

```bash
./stencil_serial 4096 4096 400
export OMP_NUM_THREADS=8
./stencil_openmp 4096 4096 400
./stencil_cuda 4096 4096 400
```

더 분명하게 보려면:

```bash
./stencil_serial 4096 4096 800
export OMP_NUM_THREADS=8
./stencil_openmp 4096 4096 800
./stencil_cuda 4096 4096 800
```
