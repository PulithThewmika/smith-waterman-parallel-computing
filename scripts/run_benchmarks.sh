#!/usr/bin/env bash
# ============================================================================
# run_benchmarks.sh  --  Build everything and sweep all configurations.
#
# Produces  data/results.csv  with one row per run:
#     impl,config,m,n,score,time
#   where 'config' is thread count (OpenMP), process count (MPI) or
#   block size (CUDA); for serial it is 1.
#
# Each configuration is run REPS times so plot.py can average out noise.
#
# Usage:  ./run_benchmarks.sh [M] [N] [SEED]
#         (defaults: 4000 4000 12345)
#
# Adjust the THREADS / PROCS / BLOCKS lists below to match your hardware.
# Author: Pulith Thewmika (IT23656338) - SE3082 Assignment 03
# ============================================================================
set -u

M=${1:-4000}
N=${2:-4000}
SEED=${3:-12345}
REPS=3

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/source"
OUT="$ROOT/data/results.csv"
mkdir -p "$ROOT/data"

# Configuration sweeps (edit to taste)
THREADS=(1 2 4 8 16)        # OpenMP: your i7-13650HX exposes 20 threads
PROCS=(1 2 4 8 16)          # MPI processes
BLOCKS=(32 64 128 256 512)  # CUDA threads per block

echo "impl,config,m,n,score,time" > "$OUT"
echo ">> Problem size: ${M} x ${N}, seed=${SEED}, reps=${REPS}"

grab_csv() { grep '^CSV,' | tail -1 | sed 's/^CSV,//'; }

# ---- Build all ----
echo ">> Building..."
make -C "$SRC/serial"  >/dev/null && echo "   serial OK"  || echo "   serial BUILD FAILED"
make -C "$SRC/openmp"  >/dev/null && echo "   openmp OK"  || echo "   openmp BUILD FAILED"
make -C "$SRC/mpi"     >/dev/null && echo "   mpi OK"     || echo "   mpi BUILD FAILED (need mpicc)"
make -C "$SRC/cuda"    >/dev/null && echo "   cuda OK"    || echo "   cuda BUILD FAILED (need nvcc)"

# ---- Serial baseline ----
if [ -x "$SRC/serial/serial_sw" ]; then
  echo ">> Serial baseline"
  for r in $(seq 1 $REPS); do
    line=$("$SRC/serial/serial_sw" "$M" "$N" "$SEED" | grab_csv)
    echo "$line" >> "$OUT"; echo "   rep$r: $line"
  done
fi

# ---- OpenMP ----
if [ -x "$SRC/openmp/omp_sw" ]; then
  echo ">> OpenMP sweep"
  for t in "${THREADS[@]}"; do
    for r in $(seq 1 $REPS); do
      line=$(OMP_NUM_THREADS=$t "$SRC/openmp/omp_sw" "$M" "$N" "$SEED" | grab_csv)
      echo "$line" >> "$OUT"; echo "   t=$t rep$r: $line"
    done
  done
fi

# ---- MPI ----
if [ -x "$SRC/mpi/mpi_sw" ]; then
  echo ">> MPI sweep"
  for p in "${PROCS[@]}"; do
    for r in $(seq 1 $REPS); do
      line=$(mpirun -np "$p" "$SRC/mpi/mpi_sw" "$M" "$N" "$SEED" 2>/dev/null | grab_csv)
      echo "$line" >> "$OUT"; echo "   p=$p rep$r: $line"
    done
  done
fi

# ---- CUDA ----
if [ -x "$SRC/cuda/cuda_sw" ]; then
  echo ">> CUDA sweep"
  for bs in "${BLOCKS[@]}"; do
    for r in $(seq 1 $REPS); do
      line=$("$SRC/cuda/cuda_sw" "$M" "$N" "$SEED" "$bs" | grab_csv)
      echo "$line" >> "$OUT"; echo "   bs=$bs rep$r: $line"
    done
  done
fi

echo ">> Done. Results -> $OUT"
echo ">> Now make graphs:  python3 $ROOT/scripts/plot.py"
