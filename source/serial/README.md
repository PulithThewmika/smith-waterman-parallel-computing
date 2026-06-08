# Serial Smith–Waterman (baseline)

The single-threaded reference implementation. It defines the timing baseline that every
parallel version is measured against, and its maximum score is the ground truth for the
cross-implementation correctness check.

## Files

| File | Purpose |
| --- | --- |
| `serial_sw.c` | Row-major matrix fill + timing + reporting |
| `sw_common.h` | Shared scoring scheme, PRNG, and helpers (see [`source/`](../README.md)) |
| `Makefile` | Build / run / clean targets |

## How it works

The scoring matrix `H` is flattened to a single 1-D array of `(m+1)×(n+1)` ints, with cell
`(i, j)` at index `i*(n+1) + j`. Row 0 and column 0 stay zero (the local-alignment boundary
condition). The fill sweeps the matrix **row by row**; each cell is an `O(1)` 4-way max of
the diagonal, up, and left predecessors and zero:

```c
int diag = H[up_row  + (j-1)] + sw_pair_score(ai, b[j-1]);
int up   = H[up_row  +  j   ] + SW_GAP;
int left = H[cur_row + (j-1)] + SW_GAP;
int v    = sw_max4(0, diag, up, left);
```

The flat layout plus `-O2` lets the compiler auto-vectorise the inner loop. Timing uses
`clock_gettime(CLOCK_MONOTONIC)` around the fill only.

## Build & run

```bash
make                            # builds ./serial_sw
make run                        # runs 8000 8000 12345
make clean

#                m    n    seed
./serial_sw     8000 8000 12345
```

## Output

```
=== Serial Smith-Waterman ===
Sequences   : 8000 x 8000  (seed=12345)
Max score   : 3545
Fill time   : 0.249000 s
CSV,serial,1,8000,8000,3545,0.249000
```

The `config` field in the CSV is always `1` for the serial run. On the reference hardware the
average fill time was **0.249 s** (the `1.00×` baseline).

## Compiler flags

```
gcc -O2 -Wall -Wextra -std=c11
```
