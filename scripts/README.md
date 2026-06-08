# `scripts/` — Benchmarking & plotting

Automation for running the full configuration sweep and turning the results into the
performance figures.

| Script | Reads | Writes |
| --- | --- | --- |
| `run_benchmarks.sh` | — (builds and runs the programs) | `data/results.csv` |
| `plot.py` | `data/results.csv` | `report/graphs/*.png` |

## `run_benchmarks.sh`

Builds all four implementations, then sweeps every configuration, running each one `REPS`
times so the noise can be averaged out. One row is appended to `data/results.csv` per run.

```bash
#                 M    N    SEED   (defaults: 4000 4000 12345)
./run_benchmarks.sh 8000 8000 12345
```

Default sweeps (edit the arrays near the top of the script to match your hardware):

| Implementation | Variable | Values |
| --- | --- | --- |
| OpenMP threads | `THREADS` | `1 2 4 8 16` |
| MPI processes | `PROCS` | `1 2 4 8 16` |
| CUDA block sizes | `BLOCKS` | `32 64 128 256 512` |
| Repetitions | `REPS` | `3` |

If a toolchain is missing (`mpicc`, `nvcc`), that build is reported as failed and its sweep
is skipped — the rest still run.

## `plot.py`

Averages the repetitions per `(impl, config)`, computes speedup as `T_serial / T_parallel`
(falling back to OpenMP @ 1 thread if no serial row exists), and confirms every
implementation reports the same max score.

```bash
python3 plot.py            # requires: pip install pandas matplotlib
```

Figures written to [`report/graphs/`](../report/graphs/):

| File | Contents |
| --- | --- |
| `openmp_time.png` / `openmp_speedup.png` | OpenMP time & speedup vs threads |
| `mpi_time.png` / `mpi_speedup.png` | MPI time & speedup vs processes |
| `cuda_time.png` / `cuda_speedup.png` | CUDA kernel time & speedup vs block size |
| `compare_time.png` / `compare_speedup.png` | Best config of each technology |

`matplotlib` runs with the headless `Agg` backend, so this works over SSH / WSL with no
display.

## Typical workflow

```bash
./scripts/run_benchmarks.sh 8000 8000 12345   # -> data/results.csv
python3 scripts/plot.py                        # -> report/graphs/*.png
```
