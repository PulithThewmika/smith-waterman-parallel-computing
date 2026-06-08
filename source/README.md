# `source/` — Implementations

Source code for all four Smith–Waterman implementations. Each version lives in its own
subdirectory with a dedicated `Makefile`, a copy of the shared header `sw_common.h`, and its
own `README.md`.

| Directory | File | Paradigm | Key parameter | Docs |
| --- | --- | --- | --- | --- |
| [`serial/`](serial/) | `serial_sw.c` | Single-threaded baseline | — | [README](serial/README.md) |
| [`openmp/`](openmp/) | `omp_sw.c` | Shared memory (threads) | tile size `TS = 64` | [README](openmp/README.md) |
| [`mpi/`](mpi/) | `mpi_sw.c` | Distributed memory (processes) | row block `BR = 128` | [README](mpi/README.md) |
| [`cuda/`](cuda/) | `cuda_sw.cu` | GPU (CUDA) | threads/block (CLI) | [README](cuda/README.md) |

## Shared header — `sw_common.h`

Every version `#include`s an identical copy of `sw_common.h`, which guarantees the four
programs are comparable and verifiable:

- **Scoring scheme** — `SW_MATCH = +2`, `SW_MISMATCH = −1`, `SW_GAP = −2`.
- **`sw_make_sequence()`** — fills a buffer with random DNA bases `{A,C,G,T}` using a
  self-contained **xorshift-64 PRNG**. Because it does not use the C library `rand()` (which
  differs between `gcc` and `nvcc`), all toolchains produce **byte-for-byte identical**
  inputs for the same seed.
- **`sw_pair_score()` / `sw_max4()`** — substitution score and the 4-way max (clamped at 0
  for the local-alignment rule).

Both input strands come from the same seed offset by one: sequence `a` uses `seed`,
sequence `b` uses `seed + 1`.

## Common CLI and output

All programs take the same positional arguments and print the same three report lines plus a
machine-readable CSV line:

```
<prog> [m] [n] [seed]            # serial, mpi
<prog> [m] [n] [seed] [arg4]     # openmp: arg4 = thread count
                                 # cuda:   arg4 = threads-per-block
```

```
Max score   : 3545
Fill time   : 0.xxxxxx s
CSV,<impl>,<config>,<m>,<n>,<score>,<time>
```

The `Max score` must be **identical across all four** for the same `m`/`n`/`seed` — this is
the cross-implementation correctness check (3545 for 8000×8000, seed 12345). Only the
`O(m·n)` matrix-fill phase is timed; traceback is excluded.

## Build

```bash
# from the repository root
make                 # all four
make cpu             # serial + openmp only

# or individually, from this directory
make -C serial
make -C openmp
make -C mpi
make -C cuda
```

See the [root README](../README.md) for benchmarking, results, and analysis.
