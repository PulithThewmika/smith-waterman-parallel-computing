#!/usr/bin/env python3
# ============================================================================
# plot.py  --  Turn data/results.csv into all graphs the assignment requires.
#
# Reads:  data/results.csv   (impl,config,m,n,score,time)
# Writes: report/graphs/*.png
#         - openmp_time.png, openmp_speedup.png
#         - mpi_time.png,    mpi_speedup.png
#         - cuda_time.png,   cuda_speedup.png
#         - compare_time.png, compare_speedup.png
#
# Speedup is measured against the serial baseline time (T_serial / T_parallel).
#
# Usage:  python3 plot.py
# Requires: pandas, matplotlib   (pip install pandas matplotlib)
# Author: Pulith Thewmika (IT23656338) - SE3082 Assignment 03
# ============================================================================
import os
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")            # headless-safe (works over SSH/WSL)
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "data", "results.csv")
GDIR = os.path.join(ROOT, "report", "graphs")
os.makedirs(GDIR, exist_ok=True)

if not os.path.exists(CSV):
    sys.exit(f"No results file at {CSV}. Run scripts/run_benchmarks.sh first.")

df = pd.read_csv(CSV)
# Average the repetitions for each (impl, config).
avg = (df.groupby(["impl", "config"], as_index=False)
         .agg(time=("time", "mean"), score=("score", "first")))

# Serial baseline time (fallback: OpenMP @ 1 thread if no serial row).
serial_rows = avg[avg.impl == "serial"]
if not serial_rows.empty:
    T_serial = float(serial_rows.time.iloc[0])
else:
    omp1 = avg[(avg.impl == "openmp") & (avg.config == 1)]
    T_serial = float(omp1.time.iloc[0]) if not omp1.empty else None
print(f"Serial baseline time: {T_serial}")

# Sanity: confirm every implementation agrees on the max score.
scores = df.groupby("impl").score.unique()
print("Scores per implementation (should all match):")
print(scores)


def subset(impl):
    s = avg[avg.impl == impl].sort_values("config")
    return s.config.tolist(), s.time.tolist()


def save_line(x, y, xlabel, ylabel, title, fname, marker="o"):
    plt.figure(figsize=(7, 4.5))
    plt.plot(x, y, marker=marker, linewidth=2)
    plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title)
    plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(GDIR, fname), dpi=140)
    plt.close()
    print("  wrote", fname)


def speedup(times):
    base = T_serial if T_serial else times[0]
    return [base / t for t in times]


# ---- OpenMP ----
x, t = subset("openmp")
if x:
    save_line(x, t, "Threads", "Execution time (s)",
              "OpenMP: Threads vs Execution Time", "openmp_time.png")
    save_line(x, speedup(t), "Threads", "Speedup (x)",
              "OpenMP: Threads vs Speedup", "openmp_speedup.png")

# ---- MPI ----
x, t = subset("mpi")
if x:
    save_line(x, t, "Processes", "Execution time (s)",
              "MPI: Processes vs Execution Time", "mpi_time.png")
    save_line(x, speedup(t), "Processes", "Speedup (x)",
              "MPI: Processes vs Speedup", "mpi_speedup.png")

# ---- CUDA ----
x, t = subset("cuda")
if x:
    save_line(x, t, "Threads per block", "Execution time (s)",
              "CUDA: Block Size vs Execution Time", "cuda_time.png")
    save_line(x, speedup(t), "Threads per block", "Speedup vs serial (x)",
              "CUDA: Block Size vs Speedup", "cuda_speedup.png")

# ---- Comparative (best config of each technology) ----
def best(impl):
    s = avg[avg.impl == impl]
    if s.empty:
        return None
    row = s.loc[s.time.idxmin()]
    return row.impl, float(row.time)

bars = [b for b in (best("openmp"), best("mpi"), best("cuda")) if b]
if bars:
    labels = [b[0].upper() for b in bars]
    times = [b[1] for b in bars]
    sp = speedup(times)

    plt.figure(figsize=(7, 4.5))
    plt.bar(labels, times)
    plt.ylabel("Execution time (s)")
    plt.title("Comparative: Best Execution Time per Technology")
    for i, v in enumerate(times):
        plt.text(i, v, f"{v:.3f}s", ha="center", va="bottom")
    plt.tight_layout(); plt.savefig(os.path.join(GDIR, "compare_time.png"), dpi=140)
    plt.close(); print("  wrote compare_time.png")

    plt.figure(figsize=(7, 4.5))
    plt.bar(labels, sp)
    plt.ylabel("Speedup vs serial (x)")
    plt.title("Comparative: Best Speedup per Technology")
    for i, v in enumerate(sp):
        plt.text(i, v, f"{v:.1f}x", ha="center", va="bottom")
    plt.tight_layout(); plt.savefig(os.path.join(GDIR, "compare_speedup.png"), dpi=140)
    plt.close(); print("  wrote compare_speedup.png")

print(f"\nAll graphs written to {GDIR}")
