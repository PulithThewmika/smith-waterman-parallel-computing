# SE3082 Parallel Computing — Assignment 03
## Smith–Waterman Local Sequence Alignment: Serial, OpenMP, MPI, CUDA

**Author:** Pulith Thewmika (Pmax) — IT23656338
**Module:** SE3082 Parallel Computing, Semester 1, 2026
**Domain:** Bioinformatics and Computational Biology
**Algorithm:** Smith–Waterman local sequence alignment (matrix-fill phase)

---

### 1. What this project does

The Smith–Waterman algorithm finds the highest-scoring *local* alignment
between two sequences by filling a dynamic-programming matrix `H`:

```
H[i][j] = max( 0,
               H[i-1][j-1] + score(a[i-1], b[j-1]),   // diagonal (match/mismatch)
               H[i-1][j]   + GAP,                      // gap in sequence b
               H[i][j-1]   + GAP )                      // gap in sequence a
```

We parallelise the O(m·n) matrix-fill phase (the timed, compute-heavy part)
with three technologies and compare them. Scoring used:
`MATCH=+2, MISMATCH=-1, GAP=-2`.

All four programs generate **identical** random DNA sequences from the same
deterministic PRNG and seed, so they must all report the **same maximum
score** — this is our cross-implementation correctness check.

---

### 2. Folder layout

```
IT23656338_SE3082_A03/
├── source/
│   ├── serial/   serial_sw.c   + Makefile + sw_common.h   (baseline)
│   ├── openmp/   omp_sw.c      + Makefile + sw_common.h   (tiled wavefront)
│   ├── mpi/      mpi_sw.c      + Makefile + sw_common.h   (column-block pipeline)
│   └── cuda/     cuda_sw.cu    + Makefile + sw_common.h   (anti-diagonal kernels)
├── scripts/
│   ├── run_benchmarks.sh   sweeps all configs -> data/results.csv
│   └── plot.py             builds all graphs -> report/graphs/*.png
├── data/                   results.csv + any input/output files
├── report/                 your PDF report + graphs/
├── screenshots/            put execution screenshots here
└── Makefile                builds everything
```

---

### 3. Prerequisites (WSL2 / Ubuntu)

```bash
sudo apt update
sudo apt install -y build-essential            # gcc + make (serial, OpenMP)
sudo apt install -y mpich                       # or: openmpi-bin libopenmpi-dev (MPI)
# CUDA: install the NVIDIA CUDA Toolkit for WSL2 (gives nvcc).
#   Follow NVIDIA's "CUDA on WSL" guide; verify with:  nvcc --version  &&  nvidia-smi
pip install pandas matplotlib                    # for plot.py
```

> **GPU note (RTX 4060 = sm_89):** `source/cuda/Makefile` sets `-arch=sm_89`.
> If your CUDA toolkit is older and rejects it, change it to `sm_86` or remove
> the `-arch` line.

---

### 4. Build

```bash
make            # builds serial, openmp, mpi, cuda
# or build one at a time:
make -C source/serial
make -C source/openmp
make -C source/mpi
make -C source/cuda
make cpu         # only serial + openmp (if MPI/CUDA not installed yet)
```

---

### 5. Run (single executions)

```bash
# Serial baseline                       m    n    seed
./source/serial/serial_sw              8000 8000 12345

# OpenMP — pick thread count via env var (or 4th arg)
OMP_NUM_THREADS=8 ./source/openmp/omp_sw 8000 8000 12345
./source/openmp/omp_sw                 8000 8000 12345 8

# MPI — pick process count with -np
mpirun -np 4 ./source/mpi/mpi_sw       8000 8000 12345

# CUDA — 4th arg is threads-per-block (block size)
./source/cuda/cuda_sw                  8000 8000 12345 256
```

Every program prints a `Max score` line (must be identical across all four for
the same m/n/seed) and a `Fill time` line, plus a machine-readable `CSV,...`
line that the benchmark/plot scripts consume.

---

### 6. Benchmark + graphs (Part B)

```bash
# Sweeps threads {1,2,4,8,16}, processes {1,2,4,8,16}, block sizes
# {32,64,128,256,512}, 3 reps each, into data/results.csv
./scripts/run_benchmarks.sh 8000 8000 12345

# Build every required graph into report/graphs/
python3 scripts/plot.py
```

Graphs produced: OpenMP/MPI/CUDA each get a *time* and a *speedup* plot, plus
two comparative bar charts. Speedup = serial_time / parallel_time.

**Tip:** take your screenshots while `run_benchmarks.sh` prints each config, and
save them in `screenshots/`.

---

### 7. Notes on the parallel designs (summary; full detail in the report)

- **OpenMP — tiled anti-diagonal wavefront.** Matrix split into 64×64 tiles;
  tiles on the same tile-diagonal are independent and run in parallel inside a
  single reused thread team. Tiling makes the inner loop cache-friendly and
  slashes the number of barriers.
- **MPI — column-block pipeline.** Columns split across ranks; each rank streams
  its right-boundary column to the next rank in blocks of 128 rows, forming a
  systolic pipeline. `MPI_Reduce(MAX)` collects the global score.
- **CUDA — anti-diagonal kernels.** One kernel launch per anti-diagonal; one
  GPU thread per cell; `atomicMax` keeps the running maximum. Ordered default-
  stream launches enforce the wavefront dependency.

---

### 8. Academic integrity / AI usage

All code in this repository was written for this assignment. AI assistance
(Claude) was used during development; the prompts and how output was adapted
are documented in `AI_USAGE.md`, as required by the assignment brief. The
Smith–Waterman recurrence reference is cited in the report (IEEE format).
