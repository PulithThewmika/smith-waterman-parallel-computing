/* ============================================================================
 * sw_common.h  --  Shared helpers for Smith-Waterman implementations
 *
 * Purpose:
 *   A small, deterministic DNA-sequence generator used by ALL four
 *   implementations (serial, OpenMP, MPI, CUDA). Because every version seeds
 *   the SAME xorshift PRNG with the SAME seed, they all build IDENTICAL input
 *   sequences and therefore MUST report the SAME maximum alignment score.
 *   This gives us a trivial correctness check across implementations.
 *
 *   We deliberately avoid the C library rand(), whose output can differ
 *   between toolchains/links (e.g. gcc vs nvcc). A self-contained xorshift64
 *   PRNG removes that risk entirely.
 *
 * Author: Pulith Thewmika (IT23656338)
 * Module: SE3082 Parallel Computing - Assignment 03
 * Note  : Written for this assignment. Algorithm reference cited in report.
 * ==========================================================================*/
#ifndef SW_COMMON_H
#define SW_COMMON_H

#include <stdlib.h>

/* ---- Scoring scheme (standard simple DNA scoring) ---- */
#define SW_MATCH     2     /* reward for a matching pair          */
#define SW_MISMATCH (-1)    /* penalty for a mismatching pair      */
#define SW_GAP      (-2)    /* linear gap penalty (insert/delete)  */

/* ---- xorshift64 PRNG: deterministic and toolchain-independent ---- */
static unsigned long long sw_rng_state = 88172645463325252ULL;

static inline void sw_rng_seed(unsigned long long s) {
    /* Avoid the all-zero state which xorshift cannot escape. */
    sw_rng_state = (s == 0ULL) ? 88172645463325252ULL : s;
}

static inline unsigned long long sw_rng_next(void) {
    unsigned long long x = sw_rng_state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    sw_rng_state = x;
    return x;
}

/* Fill buf[0..len-1] with random DNA bases {A,C,G,T} and NUL-terminate. */
static inline void sw_make_sequence(char *buf, int len, unsigned long long seed) {
    static const char bases[4] = { 'A', 'C', 'G', 'T' };
    sw_rng_seed(seed);
    for (int i = 0; i < len; ++i)
        buf[i] = bases[sw_rng_next() & 3ULL];   /* &3 == %4, fast */
    buf[len] = '\0';
}

/* Substitution score for a single pair of bases. */
static inline int sw_pair_score(char x, char y) {
    return (x == y) ? SW_MATCH : SW_MISMATCH;
}

/* Branch-free-ish max of four ints, clamped at 0 (local alignment rule). */
static inline int sw_max4(int a, int b, int c, int d) {
    int m = a;
    if (b > m) m = b;
    if (c > m) m = c;
    if (d > m) m = d;
    return m;
}

#endif /* SW_COMMON_H */
