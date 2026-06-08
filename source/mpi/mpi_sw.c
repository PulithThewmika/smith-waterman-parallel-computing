#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "sw_common.h"

#define DEFAULT_M 8000
#define DEFAULT_N 8000
#define DEFAULT_SEED 12345ULL
#define BR 128            /* row-block size: pipeline granularity (tunable) */

int main(int argc, char **argv) {
    int m = DEFAULT_M, n = DEFAULT_N;
    unsigned long long seed = DEFAULT_SEED;

    MPI_Init(&argc, &argv);
    int rank, P;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &P);

    if (argc >= 3) { m = atoi(argv[1]); n = atoi(argv[2]); }
    if (argc >= 4) { seed = strtoull(argv[3], NULL, 10); }
    if (m <= 0 || n <= 0) {
        if (rank == 0) fprintf(stderr, "m and n must be positive\n");
        MPI_Finalize(); return 1;
    }

    /* ---- Every rank builds the FULL sequences (cheap, O(m+n) memory).
       This avoids scattering and keeps indexing identical to the serial code. */
    char *a = (char *)malloc((size_t)m + 1);
    char *b = (char *)malloc((size_t)n + 1);
    sw_make_sequence(a, m, seed);
    sw_make_sequence(b, n, seed + 1);

    /* ---- Decide this rank's contiguous column range [gc0 .. gc1] (1-indexed).
      Columns 1..n are spread as evenly as possible across P ranks. ---- */
    int base = n / P, rem = n % P;
    int gc0 = rank * base + (rank < rem ? rank : rem) + 1;
    int local_w = base + (rank < rem ? 1 : 0);
    int gc1 = gc0 + local_w - 1;
    (void)gc1;

    /* ---- Local matrix: (m+1) x (local_w+1). Column 0 is the ghost column. */
    size_t lcols = (size_t)local_w + 1;
    int *H = (int *)calloc(((size_t)m + 1) * lcols, sizeof(int));
    if (!H) { fprintf(stderr, "rank %d: calloc failed\n", rank); MPI_Abort(MPI_COMM_WORLD, 1); }

    /* Send buffer: one big buffer; each row-block writes a disjoint slice, so
      several non-blocking sends can be in flight at once without clobbering. */
    int *sendbuf = (int *)malloc(((size_t)m + 1) * sizeof(int));
    int num_blocks = (m + BR - 1) / BR;
    MPI_Request *reqs = (MPI_Request *)malloc((size_t)num_blocks * sizeof(MPI_Request));
    int nreq = 0;

    int local_max = 0;

    MPI_Barrier(MPI_COMM_WORLD);          /* align the start for fair timing */
    double t0 = MPI_Wtime();

    /* ---- March down the matrix one row-block at a time ---- */
    for (int r0 = 1; r0 <= m; r0 += BR) {
        int r1 = (r0 + BR - 1 < m) ? (r0 + BR - 1) : m;   /* inclusive */
        int blk = r1 - r0 + 1;

        /* (1) Receive ghost-column values for these rows from the left rank. */
        if (rank > 0) {
            int tmp[BR];
            MPI_Recv(tmp, blk, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int t = 0; t < blk; ++t)
                H[(size_t)(r0 + t) * lcols + 0] = tmp[t];   /* fill ghost col */
        }
        /* rank 0's ghost column stays 0 (matrix left boundary). */

        /* (2) Compute every owned cell for the rows in this block. */
        for (int i = r0; i <= r1; ++i) {
            const char ai = a[i - 1];
            size_t up_row  = (size_t)(i - 1) * lcols;
            size_t cur_row = (size_t)i * lcols;
            for (int lc = 1; lc <= local_w; ++lc) {
                int gj = gc0 + lc - 1;                       /* global column */
                int diag = H[up_row  + (lc - 1)] + sw_pair_score(ai, b[gj - 1]);
                int up   = H[up_row  +  lc     ] + SW_GAP;
                int left = H[cur_row + (lc - 1)] + SW_GAP;
                int v = sw_max4(0, diag, up, left);
                H[cur_row + lc] = v;
                if (v > local_max) local_max = v;
            }
        }

        /* (3) Forward this block's right-boundary column to the next rank. */
        if (rank < P - 1) {
            for (int t = 0; t < blk; ++t)
                sendbuf[r0 + t] = H[(size_t)(r0 + t) * lcols + local_w];
            MPI_Isend(&sendbuf[r0], blk, MPI_INT, rank + 1, 0,
                      MPI_COMM_WORLD, &reqs[nreq++]);
        }
    }

    /* Ensure all our non-blocking sends have completed before freeing buffers. */
    if (nreq > 0) MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE);

    double elapsed = MPI_Wtime() - t0;

    /* ---- Combine results: global max score + critical-path (max) time ---- */
    int global_max = 0;
    double max_elapsed = 0.0;
    MPI_Reduce(&local_max, &global_max, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("=== MPI Smith-Waterman (column-block pipeline) ===\n");
        printf("Sequences   : %d x %d  (seed=%llu)\n", m, n, seed);
        printf("Processes   : %d   (row-block BR=%d)\n", P, BR);
        printf("Max score   : %d\n", global_max);
        printf("Fill time   : %.6f s\n", max_elapsed);
        /* CSV: impl,processes,m,n,score,time */
        printf("CSV,mpi,%d,%d,%d,%d,%.6f\n", P, m, n, global_max, max_elapsed);
    }

    free(a); free(b); free(H); free(sendbuf); free(reqs);
    MPI_Finalize();
    return 0;
}
