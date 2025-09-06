// See LICENSE for license details.

#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini_testutils.h"

#define CHECK_RESULT 1

#define NO_BIAS 1
#define FULL_BIAS_WIDTH 1

#if FULL_BIAS_WIDTH
typedef acc_t ACC_T;
#else
typedef elem_t ACC_T;
#endif

#ifndef BAREMETAL
//#define MAT_DIM_I 512
//#define MAT_DIM_K 512
//#define MAT_DIM_J 512
#define MAT_DIM_I 10
#define MAT_DIM_K 10
#define MAT_DIM_J 10
#else
//#define MAT_DIM_I 64
//#define MAT_DIM_K 64
//#define MAT_DIM_J 64FP
#define MAT_DIM_I 6
#define MAT_DIM_K 6
#define MAT_DIM_J 6
#endif

// Use CLOCK_REALTIME if CLOCK_MONOTONIC is not available
#ifndef CLOCK_MONOTONIC
    #define CLOCK_MONOTONIC CLOCK_REALTIME
#endif

// Start time measurement
struct timespec start_timec, end_timec, start_timeG, end_timeG;

/*
// Read cycle and instruction counters
static inline uint64_t rdcycle() {
    uint64_t cycle;
    asm volatile ("rdcycle %0" : "=r"(cycle));
    return cycle;
}

static inline uint64_t rdinstret() {
    uint64_t instret;
    asm volatile ("rdinstret %0" : "=r"(instret));
    return instret;
}
*/
static inline uint64_t rdtime() {
    uint64_t time;
    asm volatile ("rdtime %0" : "=r"(time));
    return time;
}


void print_tile(elem_t* in, int tile_dim) {
  for (size_t r = 0; r < tile_dim; r++) {
    printf("row starts at: %p\n", in +r*MAT_DIM_J);
    for (size_t c = 0; c < tile_dim; c++) {
      printf("%d ", *(in +r*MAT_DIM_J + c));
    }
    printf("\n");
  }
}

void full_matmul(elem_t A[MAT_DIM_I][MAT_DIM_K], elem_t B[MAT_DIM_K][MAT_DIM_J], ACC_T D[MAT_DIM_I][MAT_DIM_J], full_t C_full[MAT_DIM_I][MAT_DIM_J]) {
  for (size_t r = 0; r < MAT_DIM_I; r++)
    for (size_t c = 0; c < MAT_DIM_J; c++) {
      C_full[r][c] = D[r][c];
      for (size_t k = 0; k < MAT_DIM_K; k++)
        C_full[r][c] += A[r][k]*B[k][c];
    }
}

void full_printMatrix(elem_t m[MAT_DIM_I][MAT_DIM_J]) {
  for (size_t i = 0; i < MAT_DIM_I; ++i) {
    for (size_t j = 0; j < MAT_DIM_J; ++j)
      //printf("%f ", (float)m[i][j]);
      //printf("%x ", (float)m[i][j]);
      //printf("%d ", m[i][j]); //added by me
      printf("%x ", elem_t_to_elem_t_bits(m[i][j]));
    printf("\n");
  }
}

int full_is_equal(elem_t x[MAT_DIM_I][MAT_DIM_J], elem_t y[MAT_DIM_I][MAT_DIM_J]) {
  for (size_t i = 0; i < MAT_DIM_I; ++i)
    for (size_t j = 0; j < MAT_DIM_J; ++j)
      if (x[i][j] != y[i][j])
        return 0;
  return 1;
}

void full_matscale(full_t full[MAT_DIM_I][MAT_DIM_J], elem_t out[MAT_DIM_I][MAT_DIM_J], acc_scale_t scale) {
  for (size_t r = 0; r < MAT_DIM_I; r++)                             
    for (size_t c = 0; c < MAT_DIM_J; c++) {
      // Scale element
      full_t scaled = ACC_SCALE(full[r][c], scale);

      // Saturate and cast element
#ifndef ELEM_T_IS_FLOAT
      full_t elem = scaled > elem_t_max ? elem_t_max : (scaled < elem_t_min ? elem_t_min : scaled);
      out[r][c] = elem;
#else
      out[r][c] = scaled; // TODO should we also saturate when using floats?
#endif
    }
} 

//Just Use for FP16
// Convert FP32 to IEEE754 FP16 (round to nearest, ties to even)
uint16_t float_to_fp16(float f) {
    uint32_t x = *((uint32_t*)&f);  // bitcast float -> uint32
    uint32_t sign = (x >> 16) & 0x8000;  // sign bit
    uint32_t mantissa = x & 0x7fffff;
    int exp = ((x >> 23) & 0xff) - 127 + 15; // re-bias exponent

    if (exp <= 0) {
        // Subnormal or zero
        if (exp < -10) return (uint16_t)sign; // underflow
        mantissa |= 0x800000;
        uint16_t h_mant = mantissa >> (14 - exp);
        return (uint16_t)(sign | h_mant);
    } else if (exp >= 31) {
        // Overflow → Inf
        return (uint16_t)(sign | 0x7c00);
    }

    uint16_t h_exp = (uint16_t)(exp << 10);
    uint16_t h_mant = (uint16_t)(mantissa >> 13);

    return (uint16_t)(sign | h_exp | h_mant);
}

// Convert FP16 to FP32
float fp16_to_float(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1f;
    uint32_t mantissa = h & 0x3ff;

    uint32_t f;
    if (exp == 0) {
        if (mantissa == 0) {
            f = sign; // zero
        } else {
            // subnormal
            exp = 127 - 15 + 1;
            while ((mantissa & 0x400) == 0) {
                mantissa <<= 1;
                exp--;
            }
            mantissa &= 0x3ff;
            f = sign | (exp << 23) | (mantissa << 13);
        }
    } else if (exp == 31) {
        // Inf/NaN
        f = sign | 0x7f800000 | (mantissa << 13);
    } else {
        // Normalized
        exp = exp - 15 + 127;
        f = sign | (exp << 23) | (mantissa << 13);
    }
    return *((float*)&f);
}




int main() {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
      perror("mlockall failed");
      exit(1);
    }
#endif

    /*
    uint64_t start_cycles = rdcycle();
    uint64_t start_instret = rdinstret();
    */
   


    printf("MAT_DIM_I: %d\n", MAT_DIM_I);// Start time measurement
    struct timespec start_time, end_time;
    printf("MAT_DIM_J: %d\n", MAT_DIM_J);
    printf("MAT_DIM_K: %d\n", MAT_DIM_K);

    gemmini_flush(0);

    static elem_t full_A[MAT_DIM_I][MAT_DIM_K] row_align(1);
    static elem_t full_B[MAT_DIM_K][MAT_DIM_J] row_align(1);
    static elem_t full_C[MAT_DIM_I][MAT_DIM_J] row_align(1);
    static ACC_T full_D[MAT_DIM_I][MAT_DIM_J] row_align_acc(1);

    static full_t gold_full[MAT_DIM_I][MAT_DIM_J];
    static elem_t gold[MAT_DIM_I][MAT_DIM_J];

#if CHECK_RESULT == 1
#ifdef FAST
//#define RAND 1
#define RAND rand_double()
#else
#define RAND rand_double()
#endif
    // printf("Init A\n");
    float counterA =0.0;
    for (size_t i = 0; i < MAT_DIM_I; ++i) {
      for (size_t j = 0; j < MAT_DIM_K; ++j) {
        //full_A[i][j] = (elem_t)counterA;
        full_A[i][j] = float_to_fp16((float)(counterA));
        //full_A[i][j] = (elem_t)RAND;
        counterA += 1.256;
        //full_A[i][j] = 60.2f;
        //printf("Size of full_A[%zu][%zu]: %zu bytes\n", i, j, sizeof(full_A[i][j]));

      }
    }
    printf("A:\n");
    // Write matrix to CSV file
    /*
    FILE *fileA = fopen("A.csv", "w");
    if (fileA == NULL) {
        printf("Error opening file!\n");
        return 1;
    }
    for (size_t i = 0; i < MAT_DIM_I; ++i) {
        for (size_t j = 0; j < MAT_DIM_K; ++j) {
            fprintf(fileA, "%.9e", full_A[i][j]);  // High precision
            if (j < MAT_DIM_K - 1) fprintf(fileA, ",");
        }
        fprintf(fileA, "\n");
    }

    fclose(fileA);
    printf("Matrix saved to A.csv\n");
    */
    //printMatrix(full_A);
    full_printMatrix(full_A);
    //exit(0);

    // printf("Init B\n");
    float counterB =0.0;
    for (size_t i = 0; i < MAT_DIM_K; ++i) {
      for (size_t j = 0; j < MAT_DIM_J; ++j) {
        //printf("MAT_DIM_J: %d\n", MAT_DIM_J);
        full_B[i][j] = float_to_fp16((float)(counterB));
        //full_B[i][j] = (elem_t)counterB;
        counterB += 1.25;

        //full_B[i][j] = 6.0f;
        //printf("Size of full_A[%zu][%zu]: %zu bytes\n", i, j, sizeof(full_A[i][j]));

      }
    }
    
    printf("B:\n");
    /*
    // Write matrix to CSV file
    FILE *fileB = fopen("B.csv", "w");
    if (fileB == NULL) {
        printf("Error opening file!\n");
        return 1;
    }
    for (size_t i = 0; i < MAT_DIM_I; ++i) {
        for (size_t j = 0; j < MAT_DIM_K; ++j) {
            fprintf(fileB, "%.9e", full_B[i][j]);  // High precision
            if (j < MAT_DIM_K - 1) fprintf(fileB, ",");
        }
        fprintf(fileB, "\n");
    }

    fclose(fileB);
    printf("Matrix saved to B.csv\n");
    */
    //printMatrix(full_B);
    full_printMatrix(full_B);

    //printf("elem_t_max = %x\n", elem_t_max); // Scientific notation
    //printf("elem_t_max = %x\n", elem_t_max); // Fixed-point notation

    // printf("Init D\n");
    for (size_t i = 0; i < MAT_DIM_I; ++i) {
      for (size_t j = 0; j < MAT_DIM_J; ++j) {
        full_D[i][j] = NO_BIAS ? 0 : RAND;
      }
    }
    
    printf("D:\n");
    // Write matrix to CSV file
    /*
    FILE *fileD = fopen("D.csv", "w");
    if (fileD == NULL) {
        printf("Error opening file!\n");
        return 1;
    }
    for (size_t i = 0; i < MAT_DIM_I; ++i) {
        for (size_t j = 0; j < MAT_DIM_K; ++j) {
            fprintf(fileD, "%.9e", full_D[i][j]);  // High precision
            if (j < MAT_DIM_K - 1) fprintf(fileD, ",");
        }
        fprintf(fileD, "\n");
    }

    fclose(fileD);
    printf("Matrix saved to D.csv\n");
    */
    //printMatrix(full_D);
    full_printMatrix(full_D);

    printf("Starting gemmini matmul\n");
    //unsigned long start = read_cycles();
    //(CLOCK_MONOTONIC, &start_timeG);
    // Calculate elapsed time
    //long seconds = end_timeG.tv_sec - start_timeG.tv_sec;
    //long nanoseconds = end_timeG.tv_nsec - start_timeG.tv_nsec;
    //double elapsed_time = seconds + nanoseconds * 1e-9;  // Convert to seconds

    //printf("Time taken: %f seconds\n", elapsed_time);

    //uint64_t start = rdtime();
    //printf("rdtime");

    tiled_matmul_auto(MAT_DIM_I, MAT_DIM_J, MAT_DIM_K,
            (elem_t*)full_A, (elem_t*)full_B, NO_BIAS ? NULL : &full_D[0][0], (elem_t*)full_C,
            MAT_DIM_K, MAT_DIM_J, MAT_DIM_J, MAT_DIM_J,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, false,
            false, false,
            false, !FULL_BIAS_WIDTH,
            0,
            WS);

    //unsigned long end = read_cycles();
    //printf("Cycles taken: %u\n", end-start);
    //clock_gettime(CLOCK_MONOTONIC, &end_timeG);

    //long seconds = end_timeG.tv_sec - start_timeG.tv_sec;
    //long nanoseconds = end_timeG.tv_nsec - start_timeG.tv_nsec;
    //double elapsed_time = seconds + nanoseconds * 1e-9;  // Convert to seconds

    //uint64_t end = rdtime();
    //printf("Time units: %llu\n", end - start);


    printf("Starting slow CPU matmul1\n");
    //unsigned long cpu_start = read_cycles();
    //uint64_t start2 = rdtime();
#ifdef FAST
    for (size_t i = 0; i < MAT_DIM_I; ++i) {
      for (size_t j = 0; j < MAT_DIM_J; ++j) {
        gold_full[i][j] = MAT_DIM_K + (NO_BIAS ? 0 : (RAND % 2));
      }
    }

#else
    full_matmul(full_A, full_B, full_D, gold_full);
#endif
    //unsigned long cpu_end = read_cycles();
    //printf("Cycles taken: %u\n", cpu_end-cpu_start);
    //uint64_t end2 = rdtime();
    //printf("Time units CPU: %llu\n", end2 - start2);
    full_matscale(gold_full, gold, ACC_SCALE_IDENTITY);
#endif

#if CHECK_RESULT == 1
    //if (!full_is_equal(full_C, gold)) {
      printf("C:\n");
      // Write matrix to CSV file
      /*
      FILE *fileC = fopen("C.csv", "w");
      if (fileC == NULL) {
          printf("Error opening file!\n");
          return 1;
      }
      for (size_t i = 0; i < MAT_DIM_I; ++i) {
        for (size_t j = 0; j < MAT_DIM_K; ++j) {
            fprintf(fileC, "%.9e", full_C[i][j]);  // High precision
            if (j < MAT_DIM_K - 1) fprintf(fileC, ",");
        }
        fprintf(fileC, "\n");
      }
      printf("Matrix saved to C.csv\n");
      */
      //printMatrix(full_C);
      full_printMatrix(full_C);
      printf("Gold:\n");
      /*
      // Write matrix to CSV file
      FILE *fileG = fopen("GOLD.csv", "w");
      if (fileG == NULL) {
        printf("Error opening file!\n");
        return 1;
      }
      for (size_t i = 0; i < MAT_DIM_I; ++i) {
        for (size_t j = 0; j < MAT_DIM_K; ++j) {
            fprintf(fileG, "%.9e", gold[i][j]);  // High precision
            if (j < MAT_DIM_K - 1) fprintf(fileG, ",");
        }
        fprintf(fileG, "\n");
      }
      printf("Matrix saved to GOLD.csv\n");
      */
      //printMatrix(gold);
      full_printMatrix(gold);
      printf("\n");

      //exit(1);
    //}
#endif

  exit(0);
}

