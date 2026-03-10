// Test 3: Small 4x4 matmul (single tile, matches DIM=4)
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini_testutils.h"

#define N 4

int main() {
    printf("TEST3: Starting small matmul\n");
    fflush(stdout);

    static elem_t A[N][N] __attribute__((aligned(16)));
    static elem_t B[N][N] __attribute__((aligned(16)));
    static elem_t C[N][N] __attribute__((aligned(16)));

    // Init to simple values
    for (int i = 0; i < N; i++)
      for (int j = 0; j < N; j++) {
        A[i][j] = (i + j) % 3;
        B[i][j] = (i * j + 1) % 4;
      }

    printf("TEST3: Matrices initialized\n");
    fflush(stdout);

    printf("TEST3: Calling gemmini_flush\n");
    fflush(stdout);
    gemmini_flush(0);
    printf("TEST3: gemmini_flush OK\n");
    fflush(stdout);

    printf("TEST3: Calling tiled_matmul_auto\n");
    fflush(stdout);

    tiled_matmul_auto(N, N, N,
            (elem_t*)A, (elem_t*)B, NULL, (elem_t*)C,
            N, N, N, N,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, false,
            false, false,
            false, false,
            0,
            WS);

    printf("TEST3: tiled_matmul_auto done\n");
    fflush(stdout);

    printf("Result C:\n");
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++)
        printf("%d ", C[i][j]);
      printf("\n");
    }

    printf("TEST3: DONE\n");
    return 0;
}
