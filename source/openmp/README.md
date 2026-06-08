# OpenMP Smith–Waterman (tiled anti-diagonal wavefront)

Shared-memory parallel version using a **tiled anti-diagonal wavefront** inside a single,
long-lived parallel region.

## Files

| File | Purpose |
| --- | --- |
| `omp_sw.c` | Tiled wavefront fill with OpenMP |
| `sw_common.h` | Shared scoring scheme, PRNG, and helpers (see [`source/`](../README.md)) |
| `Makefile` | Build / run / clean targets |

## How it works

A naïve "one task per cell on each anti-diagonal" loop is correct but pathological: it needs
`(m+n)` barriers, strides through memory cache-hostilely, and pays fork–join overhead on
every diagonal. Instead the matrix is **tiled into `TS × TS = 64 × 64` blocks**:

- Tiles on the same tile-diagonal `D = ti + tj` are mutually independent and are distributed
  with `#pragma omp for schedule(dynamic)` — so fast threads do not idle at the short corner
  diagonals.
- The whole sweep runs inside **one** `#pragma omp parallel` region; the implicit barrier at
  the end of each tile-diagonal's `for` enforces that `D-1` finishes before `D` starts. This
  reuses the thread team instead of re-forking every diagonal.
- Each tile is filled serially in row-major order (cache- and SIMD-friendly).
- The running maximum is folded with `reduction(max:max_score)`, so no atomics or critical
  sections appear in the hot loop.

This cuts the barrier count from `~16 000` to `~251` (`TR + TC − 1`) for an 8000×8000 matrix.

`TS` is a compile-time constant (`#define TS 64`); change it and rebuild to experiment.

## Build & run

```bash
make                              # builds ./omp_sw
make run                          # OMP_NUM_THREADS=8, 8000 8000 12345
make clean

# thread count via env var ...
OMP_NUM_THREADS=8 ./omp_sw 8000 8000 12345
# ... or as the 4th argument
./omp_sw 8000 8000 12345 8
```

## Output

```
=== OpenMP Smith-Waterman (tiled anti-diagonal) ===
Sequences   : 8000 x 8000  (seed=12345)
Threads     : 8   (tile size TS=64)
Max score   : 3545
Fill time   : 0.088600 s
CSV,openmp,8,8000,8000,3545,0.088600
```

The CSV `config` field is the thread count.

## Scaling on the reference hardware

Best result **2.81×** at 8 threads; the curve plateaus and slightly regresses at 16. The
hybrid 6 P-core + 4 E-core CPU, the wavefront's implicit barriers, and limited SMT benefit
for a memory-bound integer kernel all flatten the curve beyond 8 threads.

> **Tip:** pin threads to the fast P-cores with `OMP_PROC_BIND=close` and
> `OMP_PLACES=cores` to avoid migration onto the slower E-cores.

## Compiler flags

```
gcc -O2 -Wall -Wextra -std=c11 -fopenmp
```
