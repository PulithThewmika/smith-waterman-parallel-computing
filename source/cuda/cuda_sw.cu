/* ============================================================================
 * cuda_sw.cu  --  CUDA parallel Smith-Waterman (anti-diagonal wavefront)
 *
 * PARALLELIZATION STRATEGY: ANTI-DIAGONAL WAVEFRONT ON THE GPU
 * -----------------------------------------------------------
 * Identical idea to the OpenMP version, but mapped to thousands of GPU
 * threads. All cells on one anti-diagonal d = i + j are independent, so each
 * is handled by ONE GPU thread. We launch one kernel PER anti-diagonal:
 *
 *      for d = 2 .. m+n:                 // host loop, serial over wavefronts
 *          launch kernel: thread t -> cell (i = i_lo + t, j = d - i)
 *
 * Kernel launches on the default stream execute IN ORDER, so diagonal d-1 is
 * guaranteed complete before diagonal d starts -> the wavefront dependency is
 * satisfied with no explicit barrier between launches.
 *
 * The matrix H lives entirely in GPU global memory for the whole sweep, so we
 * pay the host<->device transfer cost only once (sequences in, max score out).
 *
 * The running maximum is kept on the device with atomicMax, avoiding a
 * separate reduction pass.
 *
 * BENCHMARK KNOB: the threads-per-block (block size) is the 4th CLI argument,
 * so you can sweep 32/64/128/256/512 as the assignment requires.
 *
 * Build : make            (nvcc; set the GPU arch in the Makefile)
 * Run   : ./cuda_sw 4000 4000 12345 256       (256 = threads/block)
 *
 * Author: Pulith Thewmika (IT23656338) - SE3082 Assignment 03
 * Target GPU: NVIDIA GeForce RTX 4060 (Ada, sm_89)
 * ==========================================================================*/
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "sw_common.h"          /* host-side sequence generator + macros */

#define DEFAULT_M 8000
#define DEFAULT_N 8000
#define DEFAULT_SEED 12345ULL
#define DEFAULT_BLOCK 256

/* Convenience macro: check a CUDA call and abort with a clear message. */
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t _e = (call);                                             \
        if (_e != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s at %s:%d -> %s\n", #call,         \
                    __FILE__, __LINE__, cudaGetErrorString(_e));             \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

/* ----------------------------------------------------------------------------
 * Kernel: compute every cell on ONE anti-diagonal.
 *   d        - the anti-diagonal index (i + j)
 *   i_lo     - smallest valid row index on this diagonal
 *   count    - number of cells on this diagonal
 *   n        - sequence length b (row stride is n+1)
 *   d_gmax   - device-side global maximum (updated via atomicMax)
 * --------------------------------------------------------------------------*/
__global__ void sw_diagonal_kernel(int *H, const char *a, const char *b,
                                    int n, int d, int i_lo, int count,
                                    int *d_gmax) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= count) return;

    int i = i_lo + t;          /* this thread's row     */
    int j = d - i;             /* its column on diag d  */

    size_t cols = (size_t)n + 1;
    size_t up_row  = (size_t)(i - 1) * cols;
    size_t cur_row = (size_t)i * cols;

    int s    = (a[i - 1] == b[j - 1]) ? SW_MATCH : SW_MISMATCH;
    int diag = H[up_row  + (j - 1)] + s;
    int up   = H[up_row  +  j     ] + SW_GAP;
    int left = H[cur_row + (j - 1)] + SW_GAP;

    int v = 0;
    if (diag > v) v = diag;
    if (up   > v) v = up;
    if (left > v) v = left;

    H[cur_row + j] = v;
    atomicMax(d_gmax, v);      /* fold into the running maximum */
}

int main(int argc, char **argv) {
    int m = DEFAULT_M, n = DEFAULT_N, block = DEFAULT_BLOCK;
    unsigned long long seed = DEFAULT_SEED;

    if (argc >= 3) { m = atoi(argv[1]); n = atoi(argv[2]); }
    if (argc >= 4) { seed = strtoull(argv[3], NULL, 10); }
    if (argc >= 5) { block = atoi(argv[4]); }
    if (m <= 0 || n <= 0 || block <= 0) {
        fprintf(stderr, "usage: %s [m] [n] [seed] [threads_per_block]\n", argv[0]);
        return 1;
    }

    /* ---- Host: build identical sequences ---- */
    char *h_a = (char *)malloc((size_t)m + 1);
    char *h_b = (char *)malloc((size_t)n + 1);
    sw_make_sequence(h_a, m, seed);
    sw_make_sequence(h_b, n, seed + 1);

    /* ---- Device allocations ---- */
    size_t cells = ((size_t)m + 1) * ((size_t)n + 1);
    int  *d_H = NULL, *d_gmax = NULL;
    char *d_a = NULL, *d_b = NULL;

    CUDA_CHECK(cudaMalloc(&d_H, cells * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_H, 0, cells * sizeof(int)));   /* boundary = 0 */
    CUDA_CHECK(cudaMalloc(&d_a, (size_t)m));
    CUDA_CHECK(cudaMalloc(&d_b, (size_t)n));
    CUDA_CHECK(cudaMalloc(&d_gmax, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_a, h_a, (size_t)m, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, (size_t)n, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_gmax, 0, sizeof(int)));

    /* ---- Time only the diagonal sweep (the parallel compute), using
     * CUDA events so the figure is comparable to the CPU fill times. ---- */
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    for (int d = 2; d <= m + n; ++d) {
        int i_lo = (d - n > 1) ? (d - n) : 1;
        int i_hi = (d - 1 < m) ? (d - 1) : m;
        int count = i_hi - i_lo + 1;
        if (count <= 0) continue;
        int grid = (count + block - 1) / block;
        sw_diagonal_kernel<<<grid, block>>>(d_H, d_a, d_b, n, d, i_lo, count, d_gmax);
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());        /* surface any kernel launch error */

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    double elapsed = ms / 1000.0;

    int h_max = 0;
    CUDA_CHECK(cudaMemcpy(&h_max, d_gmax, sizeof(int), cudaMemcpyDeviceToHost));

    printf("=== CUDA Smith-Waterman (anti-diagonal) ===\n");
    printf("Sequences   : %d x %d  (seed=%llu)\n", m, n, seed);
    printf("Block size  : %d threads/block\n", block);
    printf("Max score   : %d\n", h_max);
    printf("Fill time   : %.6f s  (kernel only)\n", elapsed);
    /* CSV: impl,block_size,m,n,score,time */
    printf("CSV,cuda,%d,%d,%d,%d,%.6f\n", block, m, n, h_max, elapsed);

    cudaFree(d_H); cudaFree(d_a); cudaFree(d_b); cudaFree(d_gmax);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    free(h_a); free(h_b);
    return 0;
}
