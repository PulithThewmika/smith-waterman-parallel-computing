#define _POSIX_C_SOURCE 199309L   /* expose CLOCK_MONOTONIC / clock_gettime */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "sw_common.h"

/* Default problem size if no CLI args are given. */
#define DEFAULT_M 8000
#define DEFAULT_N 8000
#define DEFAULT_SEED 12345ULL

int main(int argc, char **argv) {
    int m = DEFAULT_M, n = DEFAULT_N;
    unsigned long long seed = DEFAULT_SEED;

    /* ---- Parse command line: [m] [n] [seed] ---- */
    if (argc >= 3) { m = atoi(argv[1]); n = atoi(argv[2]); }
    if (argc >= 4) { seed = strtoull(argv[3], NULL, 10); }
    if (m <= 0 || n <= 0) { fprintf(stderr, "m and n must be positive\n"); return 1; }

    /* ---- Build the two input sequences (identical across all versions) ---- */
    char *a = (char *)malloc((size_t)m + 1);
    char *b = (char *)malloc((size_t)n + 1);
    if (!a || !b) { fprintf(stderr, "seq malloc failed\n"); return 1; }
    sw_make_sequence(a, m, seed);
    sw_make_sequence(b, n, seed + 1);   /* +1 so the two strands differ */

    /* ---- Allocate the scoring matrix H, flattened to 1-D, zero-initialised.
     * Row stride is (n+1). Cell (i, j) lives at index i*(n+1) + j.
     * Row 0 and column 0 stay 0 (local-alignment boundary conditions). ---- */
    size_t rows = (size_t)m + 1, cols = (size_t)n + 1;
    int *H = (int *)calloc(rows * cols, sizeof(int));
    if (!H) { fprintf(stderr, "matrix calloc failed (%zu cells)\n", rows * cols); return 1; }

    /* ---- Timed region: fill the matrix row by row ---- */
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    int max_score = 0;
    for (int i = 1; i <= m; ++i) {
        const char ai = a[i - 1];
        const int  up_row = (i - 1) * (int)cols;   /* base index of row i-1 */
        const int  cur_row = i * (int)cols;        /* base index of row i   */
        for (int j = 1; j <= n; ++j) {
            int diag = H[up_row  + (j - 1)] + sw_pair_score(ai, b[j - 1]);
            int up   = H[up_row  +  j     ] + SW_GAP;
            int left = H[cur_row + (j - 1)] + SW_GAP;
            int v = sw_max4(0, diag, up, left);
            H[cur_row + j] = v;
            if (v > max_score) max_score = v;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;

    /* ---- Report (machine-readable last line for the benchmark script) ---- */
    printf("=== Serial Smith-Waterman ===\n");
    printf("Sequences   : %d x %d  (seed=%llu)\n", m, n, seed);
    printf("Max score   : %d\n", max_score);
    printf("Fill time   : %.6f s\n", elapsed);
    /* CSV line consumed by scripts/plot.py : impl,threads,m,n,score,time */
    printf("CSV,serial,1,%d,%d,%d,%.6f\n", m, n, max_score, elapsed);

    free(a); free(b); free(H);
    return 0;
}
