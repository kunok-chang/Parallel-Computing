#!/usr/bin/env python3
"""
plot_speedup.py
Reads results.csv produced by compare_bench and generates two plots:
  1. Kernel time (ms) – naive vs tiled, grouped by N
  2. Speedup (naive / tiled) vs N

Usage:
    python plot/plot_speedup.py results.csv

Output:
    speedup_stencil.png
    speedup_matmul.png
"""

import sys
import csv
import os
from collections import defaultdict

try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import numpy as np
except ImportError:
    print("matplotlib and numpy are required:  pip install matplotlib numpy")
    sys.exit(1)


# --------------------------------------------------------------------------- #
# Parse CSV                                                                    #
# --------------------------------------------------------------------------- #
def load_results(path):
    """Return dict: kernel -> N -> variant -> (time_ms, metric)"""
    data = defaultdict(lambda: defaultdict(dict))
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            kernel  = row["kernel"]
            N       = int(row["N"])
            variant = row["variant"]
            time_ms = float(row["time_ms"])
            metric  = float(row["metric"])
            data[kernel][N][variant] = (time_ms, metric)
    return data


# --------------------------------------------------------------------------- #
# Plot one kernel type                                                          #
# --------------------------------------------------------------------------- #
def plot_kernel(kernel, records, metric_label):
    Ns = sorted(records.keys())
    naive_times = [records[N].get("naive", (None, None))[0] for N in Ns]
    tiled_times = [records[N].get("tiled", (None, None))[0] for N in Ns]
    speedups    = [n/t if n and t else None
                   for n, t in zip(naive_times, tiled_times)]

    x = np.arange(len(Ns))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"{kernel.upper()} – Naive vs Tiled", fontsize=13)

    # --- Kernel time bar chart ---
    bars1 = ax1.bar(x - width/2, naive_times, width, label="Naive",
                    color="#5B9BD5", edgecolor="white")
    bars2 = ax1.bar(x + width/2, tiled_times, width, label="Tiled",
                    color="#ED7D31", edgecolor="white")
    ax1.set_xlabel("Problem size N")
    ax1.set_ylabel("Kernel time (ms)")
    ax1.set_title("Execution time")
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(n) for n in Ns])
    ax1.legend()
    ax1.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax1.grid(axis="y", linewidth=0.4, alpha=0.6)

    for bar in bars1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h * 1.01,
                 f"{h:.2f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h * 1.01,
                 f"{h:.2f}", ha="center", va="bottom", fontsize=8)

    # --- Speedup line chart ---
    ax2.plot(Ns, speedups, marker="o", color="#70AD47", linewidth=2,
             markersize=7, label="Speedup (naive/tiled)")
    ax2.axhline(1.0, color="gray", linewidth=0.8, linestyle="--")
    ax2.set_xlabel("Problem size N")
    ax2.set_ylabel("Speedup (×)")
    ax2.set_title("Speedup")
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(Ns)
    ax2.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax2.legend()
    ax2.grid(linewidth=0.4, alpha=0.6)

    for n, s in zip(Ns, speedups):
        if s:
            ax2.annotate(f"{s:.2f}×", xy=(n, s),
                         xytext=(0, 8), textcoords="offset points",
                         ha="center", fontsize=9)

    plt.tight_layout()
    out = f"speedup_{kernel}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


# --------------------------------------------------------------------------- #
# Main                                                                          #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "results.csv"
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        sys.exit(1)

    all_data = load_results(csv_path)

    metric_labels = {
        "stencil": "Bandwidth (GB/s)",
        "matmul":  "GFLOP/s",
    }

    for kernel, records in all_data.items():
        plot_kernel(kernel, records, metric_labels.get(kernel, "metric"))

    print("Done.")
