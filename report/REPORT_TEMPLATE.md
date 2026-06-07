# Smith–Waterman Local Sequence Alignment — Parallelisation Report
### SE3082 Parallel Computing, Assignment 03

**Author:** Pulith Thewmika (IT23656338)
**Algorithm:** Smith–Waterman local sequence alignment
**Domain:** Bioinformatics and Computational Biology

> This is a **template**. Replace every `[FILL]` with your own measured numbers,
> insert your graphs from `report/graphs/`, and tighten the prose to 3–4 pages
> single-spaced (excluding images/graphs). Keep the section structure — it maps
> 1:1 to the marking criteria.

---

## 1. Introduction (brief)

Smith–Waterman computes the optimal local alignment of two sequences by filling
a dynamic-programming matrix `H` of size (m+1)×(n+1) using the recurrence
`H[i][j] = max(0, H[i-1][j-1]+s, H[i-1][j]+g, H[i][j-1]+g)`. The matrix-fill
phase is O(m·n) and dominates runtime, so it is the target for parallelisation.
We implemented a serial baseline and three parallel versions (OpenMP, MPI,
CUDA), and verified all four report the same maximum score (correctness).

Test problem: random DNA sequences, m = n = `[FILL e.g. 8000]`, seed `12345`,
scoring `MATCH=+2, MISMATCH=-1, GAP=-2`. All versions reported a maximum score
of `[FILL e.g. 3545]`.

---

## 2. Parallelisation Strategies (4 marks)

### 2.1 OpenMP — tiled anti-diagonal wavefront
The dependency `H[i][j] <- H[i-1][j-1], H[i-1][j], H[i][j-1]` forbids a naive
row-parallel loop. Cells on one anti-diagonal `d = i+j` are independent, so we
sweep diagonals and parallelise within each. To avoid (a) the synchronisation
cost of (m+n) barriers and (b) cache-unfriendly strided access, we tile the
matrix into 64×64 blocks and run the wavefront at **tile granularity**: tiles on
the same tile-diagonal are independent and computed by separate threads, while
each tile is filled internally in cache-friendly row-major order. A **single**
parallel region is reused across all tile-diagonals to avoid repeated fork/join.

**Design justification:** tiling reduced the barrier count from `m+n` to
`(#tile-rows + #tile-cols − 1)` and improved single-thread time substantially
through better cache locality. Tile size is a trade-off — smaller tiles expose
more parallel headroom; larger tiles improve single-thread cache use but
saturate memory bandwidth sooner. `[FILL: state the TS you used and why]`

**Load balancing / data distribution:** `schedule(dynamic)` distributes tiles on
demand, which matters because tile-diagonals near the matrix corners are short.

### 2.2 MPI — column-block pipeline
Columns are partitioned into P contiguous blocks, one per rank. The only
cross-rank dependency is that a rank needs its left neighbour's last column at
each row. Sending one value per row would incur m latency-bound messages, so we
process rows in **blocks of 128**: a rank receives 128 boundary values, computes
128 rows for all its columns, then forwards 128 boundary values to the next
rank. This forms a **systolic pipeline** — once filled, all ranks compute
simultaneously on different row-blocks. Non-blocking `MPI_Isend` lets a rank
continue while its send is in flight; `MPI_Reduce(MAX)` collects the global
score and the critical-path time.

**Design justification:** the pipeline is the natural decomposition for SW under
distributed memory; the row-block size trades message count against pipeline
fill latency. `[FILL: discuss BR=128 choice]`

### 2.3 CUDA — anti-diagonal kernels
The whole `H` matrix lives in GPU global memory for the entire sweep (one H2D
transfer of sequences in, one D2H transfer of the score out). We launch **one
kernel per anti-diagonal**; thread *t* computes cell `(i_lo+t, d−(i_lo+t))`.
Ordered default-stream launches guarantee diagonal d−1 completes before d. The
running maximum is maintained with `atomicMax` on a single device integer,
avoiding a separate reduction pass.

**Design justification:** anti-diagonal mapping maximises the number of
independent threads per launch; keeping H resident on the GPU avoids per-step
transfers. Block size is a tunable launch parameter for the evaluation.

---

## 3. Runtime Configurations (3 marks)

**Hardware (ASUS ROG Strix G16 G614JV-AS73):**
- CPU: Intel Core i7-13650HX — 6 P-cores + 8 E-cores (14 cores / 20 threads)
- GPU: NVIDIA GeForce RTX 4060 Laptop, 8 GB GDDR6 (Ada, sm_89)
- RAM: 16 GB DDR5-4800
- OS: Windows 11 + WSL2 (Ubuntu `[FILL version]`)

**Software:**
- gcc `[FILL: gcc --version]`, OpenMP `[FILL]`
- MPI: `[FILL: mpich x.y / openmpi x.y — run mpirun --version]`
- CUDA Toolkit `[FILL: nvcc --version]`, driver `[FILL: nvidia-smi]`
- Python `[FILL]`, matplotlib/pandas for graphs

**Configuration parameters tested:**
- OpenMP threads: 1, 2, 4, 8, 16  (tile size TS = `[FILL]`)
- MPI processes: 1, 2, 4, 8, 16  (row-block BR = 128)
- CUDA block sizes: 32, 64, 128, 256, 512
- Problem size: m = n = `[FILL]`, seed = 12345, 3 repetitions averaged

---

## 4. Performance Analysis (4 marks)

> Insert graphs from `report/graphs/`. Reference each by name.

### 4.1 OpenMP
`[FILL]` — Insert `openmp_time.png` and `openmp_speedup.png`. Report speedup at
each thread count and **efficiency = speedup / threads**. Note where speedup
plateaus (likely once memory bandwidth saturates, since SW matrix-fill is
memory-bound: few arithmetic ops per memory access). Discuss P-core vs E-core
effects on the i7-13650HX hybrid architecture.

### 4.2 MPI
`[FILL]` — Insert `mpi_time.png` and `mpi_speedup.png`. Discuss pipeline fill
overhead (ranks idle until the wavefront reaches them) and per-block message
latency. On a single node, MPI processes share the same memory bus, so the
pipeline competes for bandwidth like OpenMP but with extra copy/communication
cost.

### 4.3 CUDA
`[FILL]` — Insert `cuda_time.png` and `cuda_speedup.png`. Discuss how block size
affects occupancy. Note the key SW-on-GPU bottleneck: short anti-diagonals near
the corners under-utilise the GPU, and one kernel launch per diagonal adds
launch overhead (≈ 2·n launches). Report best block size and speedup vs serial.

### 4.4 Comparative analysis (part of Part B, 7 marks)
`[FILL]` — Insert `compare_time.png` and `compare_speedup.png`. Compare all
three at the same problem size. Expected ordering for large sequences: **CUDA
fastest** (thousands of cells per diagonal in parallel), **OpenMP next** (good
shared-memory speedup until bandwidth limit), **MPI slowest on a single laptop**
(communication + pipeline overhead with no extra physical nodes). Justify which
you would choose given ample resources (CUDA for a single large alignment; MPI
becomes attractive only across a real multi-node cluster). Discuss each
approach's strengths/weaknesses for SW specifically.

---

## 5. Critical Reflection (4 marks)

- **Challenges:** `[FILL]` e.g. avoiding the wavefront race; getting OpenMP to
  actually scale (single parallel region + tiling); building a correct,
  deadlock-free MPI pipeline; CUDA launch-per-diagonal overhead.
- **Scalability limitations:** SW matrix-fill is memory-bandwidth bound; the
  wavefront has limited parallelism at the matrix corners; MPI communication and
  pipeline latency cap single-node speedup.
- **Future optimisations:** shared-memory tiling inside CUDA kernels; processing
  several anti-diagonals per launch to cut launch overhead; overlapping MPI
  communication with computation; an SIMD-vectorised inner loop; using `short`
  instead of `int` for `H` to halve memory traffic.
- **Lessons learned:** `[FILL]` e.g. correct ≠ fast; reducing synchronisation
  and improving locality often matter more than raw core count; the best
  paradigm depends on hardware and problem size.

---

## References (IEEE format)
[1] T. F. Smith and M. S. Waterman, "Identification of common molecular
    subsequences," *Journal of Molecular Biology*, vol. 147, no. 1,
    pp. 195–197, 1981.
[2] `[FILL: OpenMP / MPI / CUDA documentation or textbook you referenced]`
[3] `[FILL: AI usage — see AI_USAGE.md]`
