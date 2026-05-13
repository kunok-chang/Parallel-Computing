# NE CUDA Week 8 — Kernel Optimization

Shared memory tiling, memory coalescing, and bank-conflict avoidance.
Companion code for the Week 8 lecture slides.

---

## Directory Structure

```
source/
├── kernels/
│   ├── stencil_naive.cu / .h    # Baseline 2D 5-point stencil
│   ├── stencil_tiled.cu / .h    # Tiled stencil  ← complete the TODOs
│   ├── matmul_naive.cu  / .h    # Baseline GEMM
│   └── matmul_tiled.cu  / .h    # Tiled GEMM     ← complete the TODOs
├── benchmarks/
│   ├── timing_harness.cu / .h   # CUDA event timing (from Week 7)
│   └── compare_bench.cu         # Main benchmark driver
├── profiling/
│   ├── run_ncu.sh               # Nsight Compute wrapper
│   └── run_nvprof.sh            # nvprof wrapper (legacy)
├── plot/
│   └── plot_speedup.py          # Matplotlib speedup charts
└── Makefile
```

---

## Quick Start

```bash
# 1. Adjust the compute capability in Makefile (default: sm_75)
#    sm_80  → A100
#    sm_86  → RTX 30xx
#    sm_89  → RTX 40xx

# 2. Build
make

# 3. Run all benchmarks (N=1024)
make run

# 4. Run individually
./compare_bench stencil 1024
./compare_bench matmul  2048

# 5. Override tile size (e.g. T=32)
make TILE=32 && ./compare_bench matmul 1024

# 6. Profile with Nsight Compute
make profile

# 7. Plot results
make plot        # generates speedup_stencil.png, speedup_matmul.png
```

---

## Student Tasks

Search for `TODO` comments in the kernel files:

### `kernels/stencil_tiled.cu`
- TODO 1 — declare shared memory with halo and bank-conflict padding
- TODO 2 — cooperative load from global → shared (handle boundaries)
- TODO 3 — `__syncthreads()` after load
- TODO 4 — compute stencil for interior threads only

### `kernels/matmul_tiled.cu`
- TODO 1 — declare shared memory tiles `As` and `Bs` with +1 padding
- TODO 2 — tile loop: load, sync, accumulate, sync
- TODO 3 — write output with bounds check

---

## Expected Results (A100, single precision)

| Kernel         | N=1024 naive (ms) | N=1024 tiled (ms) | Speedup |
|----------------|:-----------------:|:-----------------:|:-------:|
| Stencil        | ~2.1              | ~0.4              | ~5×     |
| GEMM           | ~38               | ~4                | ~9×     |

Your numbers will vary by GPU. The key observation:
- Tiled stencil approaches the memory bandwidth roof.
- Tiled GEMM approaches the compute roof for large N.

---

## Profiling Tips

```bash
# Check L1 hit rate improvement
ncu --metrics l1tex__t_sector_hit_rate.pct ./compare_bench matmul 1024

# Check occupancy
ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active \
    ./compare_bench stencil 1024

# Check DRAM throughput
ncu --metrics dram__bytes_read.sum,dram__bytes_write.sum \
    ./compare_bench all 1024
```

---

## Dependencies

| Tool | Purpose |
|------|---------|
| `nvcc` (CUDA ≥ 11) | Compiler |
| `ncu` | Nsight Compute profiler |
| `nvprof` | Legacy profiler (CUDA ≤ 11) |
| Python ≥ 3.8 + `matplotlib`, `numpy` | Plotting |

Install Python dependencies:
```bash
pip install matplotlib numpy
```

---

## Report Checklist (Week 8 Deliverable)

- [ ] `stencil_tiled.cu` with all TODOs completed
- [ ] `matmul_tiled.cu` with all TODOs completed
- [ ] Performance table: naive vs tiled at N = 512 / 1024 / 2048
- [ ] `ncu` screenshot: L1 hit rate and memory throughput before/after
- [ ] 1-paragraph analysis: why measured speedup differs from theoretical T×
