# MPI Smith–Waterman (column-block pipeline)

Distributed-memory parallel version using a **column-block partition with a row-block
systolic pipeline**.

## Files

| File | Purpose |
| --- | --- |
| `mpi_sw.c` | Column-block pipeline with non-blocking sends |
| `sw_common.h` | Shared scoring scheme, PRNG, and helpers (see [`source/`](../README.md)) |
| `Makefile` | Build / run / clean targets |

## How it works

Anti-diagonal scheduling is awkward across processes (the diagonal changes shape every step
and needs ragged communication). This version uses a simpler, very effective decomposition:

- **Column-block partition.** The `n` columns are split into `P` contiguous blocks, one per
  rank. Rank `r` owns global columns `[gc0 .. gc1]` and keeps a local matrix of size
  `(m+1) × (local_w+1)`. Local column 0 is a **ghost column** holding the left neighbour's
  rightmost owned column. Each rank builds the full input sequences locally (cheap, `O(m+n)`
  memory) so indexing matches the serial code.
- **Row-block pipeline.** Because the recurrence reads `H[i][j-1]`, exactly one column
  crosses each rank boundary per row. Sending one value per row would be latency-bound, so
  rows are processed in **`BR = 128`-row blocks**: a rank `MPI_Recv`s `BR` ghost values from
  its left neighbour, computes `BR` rows over all its owned columns, then forwards the new
  right-boundary column to its right neighbour with a non-blocking `MPI_Isend`.
- **Overlap.** While rank `r` works on row-block `k`, rank `r−1` is already producing block
  `k+1` — the ranks form a systolic pipeline. `MPI_Waitall` drains the in-flight sends at the
  end.
- **Reduction.** The global score and the critical-path time are collected with
  `MPI_Reduce(MAX)` on rank 0.

`BR` is a compile-time constant (`#define BR 128`). Too small → latency-bound per-row
pinging; too large → later ranks idle while the first finishes its first block. `BR = 128`
sits in a wide flat optimum on the reference hardware.

## Build & run

```bash
make                                   # builds ./mpi_sw  (needs mpicc)
make run                               # mpirun -np 4, 8000 8000 12345
make clean

# process count via -np
mpirun -np 4 ./mpi_sw 8000 8000 12345

# to over-subscribe more ranks than physical cores:
mpirun --oversubscribe -np 16 ./mpi_sw 8000 8000 12345
```

## Output (rank 0 only)

```
=== MPI Smith-Waterman (column-block pipeline) ===
Sequences   : 8000 x 8000  (seed=12345)
Processes   : 16   (row-block BR=128)
Max score   : 3545
Fill time   : 0.047900 s
CSV,mpi,16,8000,8000,3545,0.047900
```

The CSV `config` field is the process count.

## Scaling on the reference hardware

**Best overall result: 5.20×** at 16 processes. Counter-intuitively MPI beats OpenMP on the
same laptop — separate processes avoid cross-thread false sharing, can each be pinned to a
physical core, and stride contiguously through memory (prefetcher-friendly). It scales
near-linearly to 4 processes (5.05×) then plateaus around 5.2× as communication latency and
core oversubscription cap the gains.

## Compiler flags

```
mpicc -O2 -Wall -Wextra -std=c11
```
