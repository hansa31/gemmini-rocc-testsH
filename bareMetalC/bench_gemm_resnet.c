// See LICENSE for license details.
// GEMM-ResNet Benchmark: 256x1000x2048 — ResNet-50 final FC (batch=256, 2048->1000 classes)

#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#include <time.h>
#endif
#include "include/gemmini_testutils.h"

#define BENCH_NAME "GEMM-ResNet"
#define MAT_DIM_I 256
#define MAT_DIM_J 1000
#define MAT_DIM_K 2048
#define NUM_RUNS 5

// Wall-clock timing
#ifdef BAREMETAL
static inline uint64_t read_wall_time(void) {
    uint64_t t;
    asm volatile ("rdtime %0" : "=r"(t));
    return t;
}
#define WALL_UNIT "ticks"
#else
#ifndef CLOCK_MONOTONIC
#define CLOCK_MONOTONIC CLOCK_REALTIME
#endif
static inline uint64_t read_wall_time(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}
#define WALL_UNIT "ns"
#endif

int main() {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
      perror("mlockall failed");
      exit(1);
    }
#endif

    printf("=== %s Benchmark ===\n", BENCH_NAME);
    printf("Dimensions: M=%d, N=%d, K=%d\n", MAT_DIM_I, MAT_DIM_J, MAT_DIM_K);
    printf("Total FLOPs: %lu\n", (unsigned long)(2UL * MAT_DIM_I * MAT_DIM_J * MAT_DIM_K));
    printf("Number of runs: %d\n", NUM_RUNS);
    printf("Wall time unit: %s\n", WALL_UNIT);
#ifndef BAREMETAL
    fflush(stdout);
#endif

    gemmini_flush(0);

    static elem_t A[MAT_DIM_I][MAT_DIM_K] row_align(1);
    static elem_t B[MAT_DIM_K][MAT_DIM_J] row_align(1);
    static elem_t C[MAT_DIM_I][MAT_DIM_J] row_align(1);

    // Initialize matrices
    for (size_t i = 0; i < MAT_DIM_I; ++i)
        for (size_t j = 0; j < MAT_DIM_K; ++j)
            A[i][j] = (elem_t)(rand() % 2);

    for (size_t i = 0; i < MAT_DIM_K; ++i)
        for (size_t j = 0; j < MAT_DIM_J; ++j)
            B[i][j] = (elem_t)(rand() % 2);

    printf("Matrices initialized. Starting benchmark...\n");
#ifndef BAREMETAL
    fflush(stdout);
#endif

    uint64_t cycles_arr[NUM_RUNS];
    uint64_t wall_arr[NUM_RUNS];

    for (int run = 0; run < NUM_RUNS; run++) {
        uint64_t start_cycles = read_cycles();
        uint64_t start_wall = read_wall_time();

        tiled_matmul_auto(MAT_DIM_I, MAT_DIM_J, MAT_DIM_K,
                (elem_t*)A, (elem_t*)B, NULL, (elem_t*)C,
                MAT_DIM_K, MAT_DIM_J, MAT_DIM_J, MAT_DIM_J,
                MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
                NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, false,
                false, false,
                false, false,
                0,
                WS);

        uint64_t end_cycles = read_cycles();
        uint64_t end_wall = read_wall_time();

        cycles_arr[run] = end_cycles - start_cycles;
        wall_arr[run] = end_wall - start_wall;

        printf("--- Run %d/%d ---\n", run + 1, NUM_RUNS);
        printf("  Cycles:    %lu\n", (unsigned long)cycles_arr[run]);
#ifdef BAREMETAL
        printf("  Wall time: %lu ticks\n", (unsigned long)wall_arr[run]);
#else
        printf("  Wall time: %lu ns (%.6f s)\n", (unsigned long)wall_arr[run],
               (double)wall_arr[run] / 1000000000.0);
#endif
#ifndef BAREMETAL
        fflush(stdout);
#endif
    }

    // Compute statistics
    uint64_t min_cyc = cycles_arr[0], max_cyc = cycles_arr[0], sum_cyc = 0;
    uint64_t min_wall = wall_arr[0], max_wall = wall_arr[0], sum_wall = 0;

    for (int i = 0; i < NUM_RUNS; i++) {
        sum_cyc += cycles_arr[i];
        sum_wall += wall_arr[i];
        if (cycles_arr[i] < min_cyc) min_cyc = cycles_arr[i];
        if (cycles_arr[i] > max_cyc) max_cyc = cycles_arr[i];
        if (wall_arr[i] < min_wall) min_wall = wall_arr[i];
        if (wall_arr[i] > max_wall) max_wall = wall_arr[i];
    }

    uint64_t avg_cyc = sum_cyc / NUM_RUNS;
    uint64_t avg_wall = sum_wall / NUM_RUNS;

    printf("\n=== %s Summary ===\n", BENCH_NAME);
    printf("Cycles  — min: %lu, avg: %lu, max: %lu\n",
           (unsigned long)min_cyc, (unsigned long)avg_cyc, (unsigned long)max_cyc);
    printf("Wall(%s) — min: %lu, avg: %lu, max: %lu\n",
           WALL_UNIT, (unsigned long)min_wall, (unsigned long)avg_wall, (unsigned long)max_wall);
    printf("\nCSV,%s,%d,%d,%d,%lu,%lu,%lu,%lu,%lu,%lu,%lu\n",
           BENCH_NAME, MAT_DIM_I, MAT_DIM_J, MAT_DIM_K,
           (unsigned long)(2UL * MAT_DIM_I * MAT_DIM_J * MAT_DIM_K),
           (unsigned long)min_cyc, (unsigned long)avg_cyc, (unsigned long)max_cyc,
           (unsigned long)min_wall, (unsigned long)avg_wall, (unsigned long)max_wall);

    exit(0);
}
