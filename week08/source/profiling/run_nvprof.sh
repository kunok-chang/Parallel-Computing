#!/usr/bin/env bash
# run_nvprof.sh
# Quick nvprof summary for older CUDA toolkits (pre-12).
#
# Usage:
#   bash profiling/run_nvprof.sh ./compare_bench [args...]

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <binary> [args...]"
    exit 1
fi

BINARY="$1"
shift
ARGS="${*:-}"

echo "=== nvprof: kernel summary ==="
nvprof "${BINARY}" ${ARGS}

echo ""
echo "=== nvprof: memory bandwidth metrics ==="
nvprof --metrics \
    dram_read_throughput,dram_write_throughput,\
l1_cache_global_hit_rate,l2_read_hit_rate \
    "${BINARY}" ${ARGS}

echo ""
echo "=== nvprof: GPU timeline ==="
nvprof --print-gpu-trace "${BINARY}" ${ARGS}
