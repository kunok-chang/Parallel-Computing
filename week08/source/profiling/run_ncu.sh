#!/usr/bin/env bash
# run_ncu.sh
# Run Nsight Compute on the given binary and export a report.
#
# Usage:
#   bash profiling/run_ncu.sh ./compare_bench [args...]
#
# Requires: ncu (Nsight Compute CLI) in PATH

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <binary> [args...]"
    exit 1
fi

BINARY="$1"
shift
ARGS="${*:-}"
REPORT="ncu_report"

echo "=== Nsight Compute: full metrics ==="
ncu --set full \
    --target-processes all \
    -o "${REPORT}" \
    "${BINARY}" ${ARGS}

echo ""
echo "Report saved to ${REPORT}.ncu-rep"
echo "Open with:  ncu-ui ${REPORT}.ncu-rep"
echo ""
echo "=== Key metrics (stdout) ==="
ncu --metrics \
    l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum,\
l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
dram__bytes_read.sum,\
dram__bytes_write.sum \
    "${BINARY}" ${ARGS}
