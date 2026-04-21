#!/usr/bin/env bash
# profiling/run_ncu.sh
# Usage: bash profiling/run_ncu.sh ./bandwidth_bench
#        bash profiling/run_ncu.sh ./stencil_bench
#
# Requires: ncu (Nsight Compute, CUDA Toolkit 10.1+)

BINARY=${1:-./bandwidth_bench}
REPORT="ncu_report"

echo "=== Nsight Compute: quick summary ==="
ncu --set basic "$BINARY"

echo ""
echo "=== Nsight Compute: full metrics (export for GUI) ==="
ncu --set full -o "${REPORT}" "$BINARY"
echo "Saved: ${REPORT}.ncu-rep"
echo "Open with: ncu-ui ${REPORT}.ncu-rep"
