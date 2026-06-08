# CUDA Smith–Waterman (anti-diagonal kernels)

GPU version that maps **one thread to one cell** and issues **one kernel launch per
anti-diagonal**.

## Files

| File | Purpose |
| --- | --- |
| `cuda_sw.cu` | Anti-diagonal kernel, device memory management, event timing |
| `sw_common.h` | Host-side sequence generator + scoring macros (see [`source/`](../README.md)) |
| `Makefile` | Build / run / clean targets |

## How it works

- **One kernel per anti-diagonal.** For each diagonal `d = i + j`, `sw_diagonal_kernel`
  launches one thread per cell on that diagonal. Launches on the default stream execute in
  submission order, so diagonal `d−1` is globally visible before `d` begins — the wavefront
  ordering is enforced for free, with no `__syncthreads` or device-wide barrier.
- **Memory layout.** The full `(m+1)×(n+1)` integer matrix lives in GPU global memory for the
  entire sweep. The only host↔device transfers are the two input sequences (once in) and the
  single integer maximum (once out), so PCIe bandwidth does not dominate.
- **Running maximum.** The kernel calls `atomicMax(d_gmax, v)`, amortising the reduction
  across the `m·n` cell updates instead of running a separate pass.
- **Timing.** Only the kernel sweep is timed, via CUDA events (`cudaEventElapsedTime`), so
  the figure is comparable to the CPU fill times.
- **Tunable block size.** Threads-per-block is the 4th CLI argument, which is what the block
  size sweep `{32, 64, 128, 256, 512}` uses.

## Build & run

```bash
make                                   # builds ./cuda_sw  (needs nvcc)
make run                               # 256 threads/block, 8000 8000 12345
make clean

#               m    n    seed  threads-per-block
./cuda_sw      8000 8000 12345 256
```

> **GPU architecture.** The `Makefile` sets `-arch=sm_89` (Ada Lovelace / RTX 40-series). If
> your CUDA toolkit rejects it, change it to a value it supports (e.g. `sm_86`) or drop the
> `-arch` line and let `nvcc` choose a default. Verify your setup with
> `nvcc --version` and `nvidia-smi`.

## Output

```
=== CUDA Smith-Waterman (anti-diagonal) ===
Sequences   : 8000 x 8000  (seed=12345)
Block size  : 256 threads/block
Max score   : 3545
Fill time   : 0.114300 s  (kernel only)
CSV,cuda,256,8000,8000,3545,0.114300
```

The CSV `config` field is the block size.

## Performance note

At this problem size the CUDA version is **launch-overhead-bound**: an 8000×8000 matrix
issues `m + n − 1 = 15 999` kernel launches, whose cumulative driver overhead (~80–150 ms)
is comparable to the actual compute. The block-size sweep is therefore nearly flat
(0.099–0.114 s); block = 32 wins narrowly (**2.39×**), within run-to-run noise. At
genome-scale problem sizes each diagonal would carry enough work to dwarf the launch cost and
the GPU would dominate — a persistent-thread / tiled-shared-memory redesign would remove the
launch floor.

## Compiler flags

```
nvcc -O2 -arch=sm_89
```
