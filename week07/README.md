# NE CUDA Week 7 — Performance Measurement and Metrics

Companion code for **Week 7: CUDA Performance Measurement and Metrics**  
(Dept. of Nuclear Engineering)

---

## Repository Structure

```
week7/
├── benchmarks/
│   ├── timing_harness.cuh      # CUDA event timer utilities, BenchResult struct
│   ├── bandwidth_bench.cu      # Vector-add bandwidth sweep (vary N)
│   └── stencil_bench.cu        # 2D stencil occupancy study (vary block size)
├── profiling/
│   ├── run_nvprof.sh           # nvprof wrapper (quick summary + metrics)
│   └── run_ncu.sh              # Nsight Compute wrapper (export for GUI)
├── plot/
│   └── plot_bandwidth.py       # matplotlib: bandwidth and kernel time vs N
└── Makefile
```

---

## Prerequisites

| Tool | Purpose |
|------|---------|
| NVIDIA GPU (Kepler or newer) | Required |
| CUDA Toolkit 11.x or 12.x | `nvcc`, `ncu`, `nvprof` |
| WSL2 + Ubuntu (Windows) | See Week 6 setup slides |
| Python 3 + matplotlib | Plotting only |

Verify your setup:
```bash
nvidia-smi
nvcc --version
```

---

## Build

```bash
cd week7

# Edit ARCH in Makefile to match your GPU:
#   sm_75  — Turing  (RTX 20xx, T4)
#   sm_86  — Ampere  (RTX 30xx, A10)
#   sm_89  — Ada     (RTX 40xx)

make
```

---

## Run

### Experiment 1: Bandwidth Sweep (vector addition)

```bash
./bandwidth_bench
```

Sample output:
```
N           H2D(ms)  Kernel(ms)  D2H(ms)  BW(GB/s)  Total(ms)
----------  -------  ----------  -------  -------   ---------
100000        0.041       0.012    0.019     99.8       0.072
1000000       0.364       0.044   0.175    272.1       0.583
10000000      3.621       0.312   1.742    481.6       5.675
100000000    36.218       2.897  17.421    620.3      56.536
```

Redirect to file for plotting:
```bash
./bandwidth_bench > results.txt
python plot/plot_bandwidth.py results.txt
```

### Experiment 2: Stencil Occupancy Study

```bash
./stencil_bench
```

---

## Profile

### nvprof (CUDA Toolkit < 12)

```bash
bash profiling/run_nvprof.sh ./bandwidth_bench
bash profiling/run_nvprof.sh ./stencil_bench
```

### Nsight Compute (recommended)

```bash
bash profiling/run_ncu.sh ./bandwidth_bench
# Opens: ncu_report.ncu-rep  →  ncu-ui ncu_report.ncu-rep
```

---

## Week 7 Deliverable Checklist

- [ ] Bandwidth table: H2D / kernel / D2H for at least 4 problem sizes
- [ ] Bandwidth vs N plot (use `plot_bandwidth.py`)
- [ ] Stencil block-size comparison table (at least 3 configurations)
- [ ] Screenshot from `nvprof` or `ncu` for one kernel
- [ ] Short paragraph: is your kernel memory-bound or compute-bound?

---

## Key Concepts

| Metric | Formula | What it tells you |
|--------|---------|------------------|
| Achieved bandwidth | bytes / (kernel_ms × 10⁶) | How close to peak memory BW |
| Arithmetic intensity | FLOPs / bytes | Memory-bound vs compute-bound |
| Occupancy | active warps / max warps | Latency hiding potential |

---

## Notes

- Edit `peak_bw` in `plot_bandwidth.py` to match your GPU's theoretical peak.
- The stencil kernel uses only global memory. Week 8 will add shared memory tiling.
- `cudaMallocHost` (pinned memory) is used in the benchmarks for accurate copy timing.
