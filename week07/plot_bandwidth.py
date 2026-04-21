#!/usr/bin/env python3
"""
plot/plot_bandwidth.py

Reads stdout from bandwidth_bench (redirected to a file) and
plots achieved bandwidth vs problem size.

Usage:
    ./bandwidth_bench > results.txt
    python plot/plot_bandwidth.py results.txt

If no file is given, reads from stdin.
"""

import sys
import re
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Parse the text table output from bandwidth_bench ─────────────────────
def parse_results(lines):
    Ns, bws, kernels = [], [], []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("N") or line.startswith("---"):
            continue
        parts = line.split()
        if len(parts) >= 5:
            try:
                Ns.append(int(parts[0]))
                kernels.append(float(parts[2]))
                bws.append(float(parts[4]))
            except ValueError:
                pass
    return Ns, kernels, bws

# ── Main ──────────────────────────────────────────────────────────────────
def main():
    fname = sys.argv[1] if len(sys.argv) > 1 else None
    if fname:
        with open(fname) as f:
            lines = f.readlines()
    else:
        lines = sys.stdin.readlines()

    Ns, kernels, bws = parse_results(lines)
    if not Ns:
        print("No data found. Check input format.", file=sys.stderr)
        sys.exit(1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Vector Addition: CUDA Performance vs Problem Size", fontsize=14)

    # ── Left: Bandwidth vs N ──────────────────────────────────────────────
    ax1.plot(Ns, bws, "o-", color="#0A7EA4", linewidth=2, markersize=6)
    ax1.set_xscale("log")
    ax1.set_xlabel("Problem size  N", fontsize=12)
    ax1.set_ylabel("Achieved bandwidth  [GB/s]", fontsize=12)
    ax1.set_title("Memory Bandwidth")
    ax1.grid(True, which="both", linestyle="--", alpha=0.5)
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"$10^{{{int(round(len(str(int(x)))-1))}}}$"
        if x >= 1 else str(x)))

    # annotate peak (edit to match your GPU)
    peak_bw = 900  # GB/s — change for your GPU
    ax1.axhline(peak_bw, color="red", linestyle="--", alpha=0.6,
                label=f"Theoretical peak ≈ {peak_bw} GB/s")
    ax1.legend()

    # ── Right: Kernel time vs N ───────────────────────────────────────────
    ax2.plot(Ns, kernels, "s-", color="#F59E0B", linewidth=2, markersize=6)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("Problem size  N", fontsize=12)
    ax2.set_ylabel("Kernel time  [ms]", fontsize=12)
    ax2.set_title("Kernel Execution Time")
    ax2.grid(True, which="both", linestyle="--", alpha=0.5)

    plt.tight_layout()
    out = "bandwidth_plot.png"
    plt.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.show()

if __name__ == "__main__":
    main()
