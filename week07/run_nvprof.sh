#!/usr/bin/env bash
# profiling/run_nvprof.sh
# Usage: bash profiling/run_nvprof.sh ./bandwidth_bench
#        bash profiling/run_nvprof.sh ./stencil_bench
#
# Requires: nvprof (comes with CUDA Toolkit < 12)

BINARY=${1:-./bandwidth_bench}

echo "=== Summary (default) ==="
nvprof "$BINARY"

echo ""
echo "=== Memory throughput metrics ==="
nvprof --metrics dram_read_throughput,dram_write_throughput "$BINARY" 2>&1

echo ""
echo "=== GPU kernel trace ==="
nvprof --print-gpu-trace "$BINARY" 2>&1
