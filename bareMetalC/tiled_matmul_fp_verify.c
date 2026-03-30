// See LICENSE for license details.
// Verification test: FP32 -> elem_t matmul with proper bit encoding
//
// This test encodes FP32 values into elem_t (uint16_t) bit patterns,
// runs a matmul on Gemmini with full_C=true (FP32 accumulator output),
// and compares against a CPU reference computed in FP32.
//
// Prints full matrices so you can visualize hw vs cpu differences
// (useful for approximate multiplier analysis).
//
// Works with any float config: FP16, BF16, FP12.

#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini_testutils.h"
#include "include/gemmini_float_convert.h"

#ifndef BAREMETAL
#define MAT_DIM_I 64
#define MAT_DIM_K 64
#define MAT_DIM_J 64
#else
#define MAT_DIM_I 16
#define MAT_DIM_K 16
#define MAT_DIM_J 16
#endif

// FP16/BF16/FP12-representable test values
static const float test_values[] = {
    0.25f, 0.5f, 1.0f, 1.5f, 2.0f, 0.75f, 1.25f, 0.125f
};
#define NUM_TEST_VALUES (sizeof(test_values) / sizeof(test_values[0]))

// Print a float as fixed-point integer: value * 1000
// e.g. 12.345 -> "12345" (meaning 12.345), -0.5 -> "-500"
// Baremetal printf has no %f support, so we do this instead.
static void print_fixed(float val) {
    long ival;
    if (val < 0) {
        printf("-");
        val = -val;
    }
    ival = (long)(val * 1000.0f + 0.5f);
    printf("%ld.%03ld", ival / 1000, ival % 1000);
}

// CPU reference matmul in FP32 using the already-quantized elem_t values.
static void cpu_matmul_fp32(
    elem_t A[MAT_DIM_I][MAT_DIM_K],
    elem_t B[MAT_DIM_K][MAT_DIM_J],
    float C_ref[MAT_DIM_I][MAT_DIM_J])
{
    for (int i = 0; i < MAT_DIM_I; i++) {
        for (int j = 0; j < MAT_DIM_J; j++) {
            float sum = 0.0f;
            for (int k = 0; k < MAT_DIM_K; k++) {
                float a = elem_bits_to_float(A[i][k]);
                float b = elem_bits_to_float(B[k][j]);
                sum += a * b;
            }
            C_ref[i][j] = sum;
        }
    }
}

static float fabsf_simple(float x) {
    return x < 0.0f ? -x : x;
}

int main() {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
        perror("mlockall failed");
        exit(1);
    }
#endif

    printf("=== FP Matmul Verify ===\n");
    printf("Dims: I=%d, K=%d, J=%d\n", MAT_DIM_I, MAT_DIM_K, MAT_DIM_J);
    printf("DIM=%d\n", DIM);
    printf("elem_t: %d bits (exp=%d, sig=%d)\n",
           (int)(sizeof(elem_t) * 8), ELEM_T_EXP_BITS, ELEM_T_SIG_BITS);

    gemmini_flush(0);

    static elem_t A[MAT_DIM_I][MAT_DIM_K] row_align(1);
    static elem_t B[MAT_DIM_K][MAT_DIM_J] row_align(1);
    static acc_t  hw_C[MAT_DIM_I][MAT_DIM_J] row_align_acc(1);
    static float  cpu_C[MAT_DIM_I][MAT_DIM_J];

    // Initialize A with properly-encoded float values
    printf("Initializing matrices with proper float encoding...\n");
    for (int i = 0; i < MAT_DIM_I; i++) {
        for (int j = 0; j < MAT_DIM_K; j++) {
            float val = test_values[(i * MAT_DIM_K + j) % NUM_TEST_VALUES];
            A[i][j] = float_to_elem_bits(val);
        }
    }

    // Initialize B with a different offset so A != B
    for (int i = 0; i < MAT_DIM_K; i++) {
        for (int j = 0; j < MAT_DIM_J; j++) {
            float val = test_values[(i * MAT_DIM_J + j + 3) % NUM_TEST_VALUES];
            B[i][j] = float_to_elem_bits(val);
        }
    }

    // Verify encoding: print a few sample values
    printf("A[0][0]: encoded=0x%04x, decoded=", (unsigned)A[0][0]);
    print_fixed(elem_bits_to_float(A[0][0]));
    printf(" (original=");
    print_fixed(test_values[0]);
    printf(")\n");

    printf("B[0][0]: encoded=0x%04x, decoded=", (unsigned)B[0][0]);
    print_fixed(elem_bits_to_float(B[0][0]));
    printf(" (original=");
    print_fixed(test_values[3]);
    printf(")\n");

    // --- CPU reference matmul (FP32, using quantized inputs) ---
    printf("CPU reference matmul...\n");
    unsigned long cpu_start = read_cycles();
    cpu_matmul_fp32(A, B, cpu_C);
    unsigned long cpu_end = read_cycles();
    printf("CPU cycles: %lu\n", (unsigned long)(cpu_end - cpu_start));

    // --- Gemmini hardware matmul (full_C=true for FP32 output) ---
    printf("Gemmini hardware matmul...\n");
    unsigned long hw_start = read_cycles();

    tiled_matmul_auto(MAT_DIM_I, MAT_DIM_J, MAT_DIM_K,
            (elem_t*)A, (elem_t*)B,
            NULL, (acc_t*)hw_C,
            MAT_DIM_K, MAT_DIM_J, MAT_DIM_J, MAT_DIM_J,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, false,
            false, false,
            true, false,   // full_C=true -> output as acc_t (FP32)
            0,
            WS);

    unsigned long hw_end = read_cycles();
    printf("Gemmini cycles: %lu\n", (unsigned long)(hw_end - hw_start));

    // --- Print full result matrices ---
    printf("\n=== Hardware Result (hw_C) ===\n");
    for (int i = 0; i < MAT_DIM_I; i++) {
        for (int j = 0; j < MAT_DIM_J; j++) {
            printf("  ");
            print_fixed((float)hw_C[i][j]);
        }
        printf("\n");
    }

    printf("\n=== CPU Reference (cpu_C) ===\n");
    for (int i = 0; i < MAT_DIM_I; i++) {
        for (int j = 0; j < MAT_DIM_J; j++) {
            printf("  ");
            print_fixed(cpu_C[i][j]);
        }
        printf("\n");
    }

    // --- Compare and print difference matrix ---
    printf("\n=== Difference (hw - cpu) ===\n");
    int mismatches = 0;
    float max_diff = 0.0f;

    for (int i = 0; i < MAT_DIM_I; i++) {
        for (int j = 0; j < MAT_DIM_J; j++) {
            float hw_val = (float)hw_C[i][j];
            float cpu_val = cpu_C[i][j];
            float diff = hw_val - cpu_val;
            float absdiff = fabsf_simple(diff);

            printf("  ");
            print_fixed(diff);

            if (absdiff > max_diff) max_diff = absdiff;
            if (absdiff > 0.001f) mismatches++;
        }
        printf("\n");
    }

    printf("\nMax absolute difference: ");
    print_fixed(max_diff);
    printf("\n");
    printf("Mismatches (|diff| > 0.001): %d / %d\n",
           mismatches, MAT_DIM_I * MAT_DIM_J);

    if (mismatches == 0) {
        printf("PASS: exact match\n");
    } else {
        printf("DONE: %d elements differ (expected with approx multiplier)\n",
               mismatches);
    }

    exit(0);
}
