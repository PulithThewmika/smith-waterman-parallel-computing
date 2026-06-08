#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "sw_common.h"

#define DEFAULT_M 8000
#define DEFAULT_N 8000
#define DEFAULT_SEED 12345ULL
#define TS 64                 /* tile size: 64x64 cells per block (tunable) */

int main(int argc, char **argv) {
    int m = DEFAULT_M, n = DEFAULT_N;
    unsigned long long seed = DEFAULT_SEED;

    if (argc >= 3) { m = atoi(argv[1]); n = atoi(argv[2]); }
    if (argc >= 4) { seed = strtoull(argv[3], NULL, 10); }
    if (argc >= 5) { omp_set_num_threads(atoi(argv[4])); }
    if (m <= 0 || n <= 0) { fprintf(stderr, "m and n must be positive\n"); return 1; }

    int nthreads = omp_get_max_threads();

    char *a = (char *)malloc((size_t)m + 1);
    char *b = (char *)malloc((size_t)n + 1);
    if (!a || !b) { fprintf(stderr, "seq malloc failed\n"); return 1; }
    sw_make_sequence(a, m, seed);
    sw_make_sequence(b, n, seed + 1);

    size_t cols = (size_t)n + 1;
    int *H = (int *)calloc(((size_t)m + 1) * cols, sizeof(int));
    if (!H) { fprintf(stderr, "matrix calloc failed\n"); return 1; }

    /* Number of tiles along each dimension. */
    int TR = (m + TS - 1) / TS;     /* tile rows    */
    int TC = (n + TS - 1) / TS;     /* tile columns */

    double t0 = omp_get_wtime();
    int max_score = 0;

   
    #pragma omp parallel
    for (int D = 0; D <= (TR - 1) + (TC - 1); ++D) {
        int ti_lo = (D - (TC - 1) > 0) ? (D - (TC - 1)) : 0;
        int ti_hi = (D < TR - 1) ? D : TR - 1;

        /* All tiles on this tile-diagonal are independent -> parallelise.
         * The implicit barrier at the end of this `for` enforces that
         * tile-diagonal D-1 finishes before D begins. */
        #pragma omp for schedule(dynamic) reduction(max:max_score)
        for (int ti = ti_lo; ti <= ti_hi; ++ti) {
            int tj = D - ti;

            int i0 = ti * TS + 1, i1 = (ti + 1) * TS; if (i1 > m) i1 = m;
            int j0 = tj * TS + 1, j1 = (tj + 1) * TS; if (j1 > n) j1 = n;

            int local_max = 0;
            for (int i = i0; i <= i1; ++i) {
                const char ai = a[i - 1];
                size_t up_row  = (size_t)(i - 1) * cols;
                size_t cur_row = (size_t)i * cols;
                for (int j = j0; j <= j1; ++j) {
                    int diag = H[up_row  + (j - 1)] + sw_pair_score(ai, b[j - 1]);
                    int up   = H[up_row  +  j     ] + SW_GAP;
                    int left = H[cur_row + (j - 1)] + SW_GAP;
                    int v = sw_max4(0, diag, up, left);
                    H[cur_row + j] = v;
                    if (v > local_max) local_max = v;
                }
            }
            if (local_max > max_score) max_score = local_max;
        }
    }

    double elapsed = omp_get_wtime() - t0;

    printf("=== OpenMP Smith-Waterman (tiled anti-diagonal) ===\n");
    printf("Sequences   : %d x %d  (seed=%llu)\n", m, n, seed);
    printf("Threads     : %d   (tile size TS=%d)\n", nthreads, TS);
    printf("Max score   : %d\n", max_score);
    printf("Fill time   : %.6f s\n", elapsed);
    printf("CSV,openmp,%d,%d,%d,%d,%.6f\n", nthreads, m, n, max_score, elapsed);

    free(a); free(b); free(H);
    return 0;
}
