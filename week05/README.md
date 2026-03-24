# week05 — Finite Difference Method with OpenMP

병렬연산 수업용 예제 모음입니다. 목표는 **“어떤 최적화가 왜 빠른지 / 왜 느린지”**를 실험으로 바로 보여주는 것입니다.

## 포함된 예제

- `stencil_basic.c`  
  기본 2D stencil + OpenMP `collapse(2)` + `static`

- `stencil_bad_branch.c`  
  매 반복마다 boundary 분기를 넣는 **나쁜 예**

- `stencil_boundary_split.c`  
  boundary와 interior를 분리한 **좋은 예**

- `stencil_collapse_compare.c`  
  `collapse` 유무 비교

- `stencil_static_dynamic.c`  
  `schedule(static)` vs `schedule(dynamic,1)` 비교

- `stencil_false_sharing.c`  
  false sharing 나쁜 예 vs padding으로 개선한 예

- `stencil_atomic_vs_reduction.c`  
  `atomic` 남용 vs `reduction`

- `stencil_blocked.c`  
  plain stencil vs cache blocking

- `stencil_loop_order.c`  
  row-major 친화적 loop order vs 비효율적 loop order

- `stencil_neumann_bad.c`  
  Neumann 경계조건을 내부 루프에 섞는 나쁜 예 vs 분리한 예

- `diffusion_bad_region.c`  
  timestep마다 `parallel for`를 새로 여는 나쁜 예

- `diffusion_good_region.c`  
  하나의 `parallel` region을 재사용하는 좋은 예

## 빌드

```bash
make
```

## 추천 실행 예시

### 1) collapse 효과
```bash
export OMP_NUM_THREADS=8
./stencil_collapse_compare 16 4096 500
./stencil_collapse_compare 1024 1024 200
```

### 2) static vs dynamic
```bash
export OMP_NUM_THREADS=8
./stencil_static_dynamic 1024 1024 200
```

### 3) boundary branch 비용
```bash
export OMP_NUM_THREADS=8
./stencil_bad_branch 1024 1024 200
./stencil_boundary_split 1024 1024 200
```

### 4) blocking 효과
```bash
export OMP_NUM_THREADS=8
./stencil_blocked 2048 2048 100 32
./stencil_blocked 2048 2048 100 64
```

### 5) loop order 영향
```bash
export OMP_NUM_THREADS=8
./stencil_loop_order 2048 2048 100
```

### 6) atomic vs reduction
```bash
export OMP_NUM_THREADS=8
./stencil_atomic_vs_reduction 100000000
```

### 7) false sharing
```bash
export OMP_NUM_THREADS=8
./stencil_false_sharing 200000000
```

### 8) parallel region 재사용
```bash
export OMP_NUM_THREADS=8
./diffusion_bad_region 1024 1024 500
./diffusion_good_region 1024 1024 500
```

### 9) affinity 비교
```bash
export OMP_NUM_THREADS=8
unset OMP_PROC_BIND
unset OMP_PLACES
./stencil_basic 1024 1024 200

export OMP_PROC_BIND=true
export OMP_PLACES=cores
./stencil_basic 1024 1024 200
```

## 수업에서 강조할 메시지

1. 규칙적인 stencil에는 `static`이 보통 최선이다.
2. boundary 처리를 interior loop에 섞으면 성능과 코드 가독성이 함께 나빠진다.
3. `atomic`은 정답일 수는 있어도 빠른 답은 아닐 수 있다.
4. `collapse`는 병렬성이 부족할 때 특히 유용하다.
5. blocking은 memory-bound stencil에서 종종 가장 효과적인 최적화다.
6. C 배열은 row-major이므로 loop order가 중요하다.
7. thread affinity는 평균 성능보다 **분산 감소**에 특히 도움이 된다.

## 업로드 팁

```bash
cd Parallel-Computing
mkdir -p week05
cp /path/to/files/week05/* week05/
git add week05
git commit -m "Add Week 5 OpenMP FDM performance examples"
git push
```
