# `screenshots/` — Execution captures

Terminal captures of every implementation running at each configuration on the reference
hardware (8000×8000 DNA sequences, seed 12345). Each implementation was run 3× per
configuration; **every run reports `Max score = 3545`**, which is the cross-implementation
correctness check.

## Environment

CPU / GPU / toolchain details (`nvidia-smi`, `nvcc --version`, `lscpu`, `gcc --version`):

![Environment: CPU, GPU and toolchains](Screenshot%202026-06-08%20011018.png)

## Serial baseline

Build and reference runs (fill time ≈ 0.249 s, the `1.00×` baseline):

![Serial build and first run](Screenshot%202026-06-08%20011146.png)
![Serial three timed runs](Screenshot%202026-06-08%20011248.png)

## OpenMP — threads 1 → 16

![OpenMP, 1 thread](Screenshot%202026-06-08%20011653.png)
![OpenMP, 2 threads](Screenshot%202026-06-08%20011710.png)
![OpenMP, 4 threads](Screenshot%202026-06-08%20011818.png)
![OpenMP, 8 threads (best, 2.81x)](Screenshot%202026-06-08%20011907.png)
![OpenMP, 16 threads](Screenshot%202026-06-08%20011951.png)

## MPI — processes 1 → 16

![MPI, 1 process](Screenshot%202026-06-08%20012106.png)
![MPI, 2 processes](Screenshot%202026-06-08%20012150.png)
![MPI, 4 processes](Screenshot%202026-06-08%20012235.png)
![MPI, 8 processes](Screenshot%202026-06-08%20012313.png)
![MPI, 16 processes (best overall, 5.20x)](Screenshot%202026-06-08%20012907.png)

## CUDA — block sizes 32 → 512

![CUDA run](Screenshot%202026-06-08%20012947.png)
![CUDA run](Screenshot%202026-06-08%20013042.png)
![CUDA run](Screenshot%202026-06-08%20013124.png)
![CUDA run](Screenshot%202026-06-08%20013207.png)
![CUDA run](Screenshot%202026-06-08%20013245.png)
![CUDA, block = 512](Screenshot%202026-06-08%20013447.png)
![CUDA, block = 512](Screenshot%202026-06-08%20013521.png)

---

The numerical results table, graphs, and analysis are in the [root README](../README.md).
