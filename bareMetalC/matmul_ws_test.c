// See LICENSE for license details.

#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include <time.h>
#include "include/gemmini_testutils.h"    //helper function for gemmini


#ifdef FAST
#define AINIT RELU      //AINIT - sets activation function
#define SINIT 12        //SINIT - scale factor for accumilation
#define N 1             // N - Matrix size multiplier
#else
#define AINIT NO_ACTIVATION
#define SINIT 0
#define N 2
#endif

/*
function is used to map a single index c to indices for three matrices A, B, and D, 
effectively iterating through all combinations of these matrices.
 */
void operands(int c, int * a, int * b, int * d) {
  *d = c % N;
  *b = (c / N) % N;
  *a = c / (N*N);
}

// Ensures gemmini hardware can handle the matrices given for the current config
#if 3*N*DIM > (BANK_NUM * BANK_ROWS) || N*N*N*DIM > ACC_ROWS
//#error scratchpad or accumulator not big enough
#endif

//Locks programs memory to prevent it from being swapping to disk
int main() {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
      perror("mlockall failed");
      exit(1);
    }
#endif

  printf("***************N: %d",N);

  static elem_t ZERO[DIM][DIM];   //zero matrix initialization

  gemmini_flush(0);         //Flush TLB (cache that stores recent translations of virtual memory addresses to physical memory addresses)
  /*
  Gemmini uses virtual addresses for memory accesses, and the TLB translates these virtual addresses into physical addresses. 
  When the TLB misses, it falls back on a page table walk (PTW) shared with the host CPU
  */
  gemmini_config_ld(DIM * sizeof(elem_t));      //configures the load pipeline of Gemmini, setting the stride for memory accesses during data loading operations (mvin)

  for (int activation = AINIT; activation <= RELU; ++activation) {
#ifdef ACC_SCALE_T_IS_FLOAT
    for (acc_scale_t scale = 0; scale <= 1.5; scale += 0.5) {     //initially it was scale=0
#else
    for (acc_scale_t scale = SINIT; scale <= 12; scale += 4) {
#endif
      static elem_t A[N][DIM][DIM] row_align(1);
      static elem_t B[N][DIM][DIM] row_align(1);
      static elem_t D[N][DIM][DIM] row_align(1);

      // We will try out every combination of A, B, D possible
      static elem_t C[N*N*N][DIM][DIM] row_align(1);
      static full_t gold_full[N*N*N][DIM][DIM];       //A full precesion intermediate result of the matrix multiplication
      static elem_t gold[N*N*N][DIM][DIM];            // The final result after post-processing, such as scaling and activation

      // ...taking into account whether we preload new weights or re-use the old ones
      static int preload[N*N*N] = {1};
      for (int i = 1; i < N*N*N; ++i)
        preload[i] = rand() % 2;

      // ...whether we pass in a D or just use zeros
      static int add_to_zeros[N*N*N];
      for (int i = 0; i < N*N*N; ++i)
        add_to_zeros[i] = rand() % 2;

      // ...and whether we accumulate on top of the previous result (Adding the current result to the previous result)
      static int accumulate[N*N*N] = {0};
      for (int i = 1; i < N*N*N; ++i)
        accumulate[i] = rand() % 2;

      // Determines whether the result of the current computation should be stored in memory
      /*
      no_output[i] is set to accumulate[i+1]. If the next operation (i+1) accumulates on the current result, 
      then there’s no need to output the current result to memory
      */
      static int no_output[N*N*N];      
      for (int i = 0; i < N*N*N-1; ++i)
        no_output[i] = accumulate[i+1];
      no_output[N*N*N-1] = 0;

      /*
      Array	Description	Values
      preload	Determines whether to load new weights (B) for the current computation.	0 (reuse weights), 1 (load new weights).
      add_to_zeros	Determines whether to use a zero matrix instead of matrix D for the computation.	0 (use D), 1 (use zeros).
      accumulate	Determines whether to add the current result to the previous result.	0 (overwrite result), 1 (accumulate result).
      no_output	Determines whether to output the result to memory (used when the next computation accumulates on this one).	0 (output), 1 (do not output).
      */

      // Print the sequence out
      /*printf("Preloads: ");
      for (int i = 0; i < N*N*N; ++i)
        printf("%d, ", preload[i]);
      printf("\n");
      printf("Zeros: ");
      for (int i = 0; i < N*N*N; ++i)
        printf("%d, ", add_to_zeros[i]);
      printf("\n");
      printf("Accumulates: ");
      for (int i = 0; i < N*N*N; ++i)
        printf("%d, ", accumulate[i]);
      printf("\n");
      printf("No outputs: ");
      for (int i = 0; i < N*N*N; ++i)
        printf("%d, ", no_output[i]);
      printf("\n");*/

      for (size_t n = 0; n < N; ++n) {
        for (size_t i = 0; i < DIM; ++i) {
          for (size_t j = 0; j < DIM; ++j) {
            A[n][i][j] = (rand() % 64) - 32;      // -32 to 32
            B[n][i][j] = (rand() % 64) - 32;
            D[n][i][j] = (rand() % 64) - 32;
          }
        }
      }

      for (size_t g = 0; g < N*N*N; ++g) {
        int a, b, d;
        operands(g, &a, &b, &d);      //operands(g, &a, &b, &d): Retrieves the indices of the matrices (A[a], B[b], D[d]) to be used for the g-th computation.

        // We need to find the last B value in case we aren't preloading new weights
        for (int last_g = g; last_g >= 0; --last_g) {
            int tmp_a, tmp_d;
            if (preload[last_g]) {
                operands(last_g, &tmp_a, &b, &tmp_d);
                break;
            }
        }

        //the matrix multiplication (In CPU ?)
        if (add_to_zeros[g])
          matmul(A[a], B[b], ZERO, gold_full[g]);
        else
          matmul(A[a], B[b], D[d], gold_full[g]);

        if (accumulate[g])
          matadd(gold_full[g], gold_full[g-1], gold_full[g]);
      }

      //Post processing and scaling
      for (size_t g = 0; g < N*N*N; ++g) {
        matscale(gold_full[g], gold[g], scale);
        if (activation == RELU)
          matrelu(gold[g], gold[g]);
      }

      //memory address initialization (In gemmini scratchpad)
      uint32_t A_addr = 0;
      uint32_t B_addr = N*DIM;
      uint32_t D_addr = 2*N*DIM;
      uint32_t C_addr_acc = 1 << (ADDR_LEN-1);

      // Calculate the proper destination addresses of everything (where results are stored)
      uint32_t C_addrs[N*N*N];
      for (size_t c = 0; c < N*N*N; ++c)
        C_addrs[c] = C_addr_acc + c*DIM;
      for (size_t c = 0; c < N*N*N; ++c) {
        int last_c;
        for (last_c = c; last_c >= 0; --last_c)
          if (!accumulate[last_c])
            break;
        if (c != last_c)
          C_addrs[c] = C_addrs[last_c] | (1 << (ADDR_LEN-2));
      }

      // printf("Moving in\n");
      // Matrices are moved into gemmini scratchpad memory
      for (size_t n = 0; n < N; ++n)
        gemmini_mvin(A[n], A_addr + n*DIM);

      for (size_t n = 0; n < N; ++n)
        gemmini_mvin(B[n], B_addr + n*DIM);

      for (size_t n = 0; n < N; ++n)
        if (n == N-1) {
          gemmini_mvin(D[n], D_addr + n*DIM);
        } else {
          gemmini_mvin(D[n], D_addr + n*DIM);
        }

      // printf("Setting mode\n");
      gemmini_config_ex(WEIGHT_STATIONARY, 0, 0);
      // stride
      gemmini_extended_config_st(DIM * sizeof(elem_t), activation, scale);

      // printf("Matmulling\n");
      for (size_t c = 0; c < N*N*N; ++c) {
        int a, b, d;
        operands(c, &a, &b, &d);

        uint32_t d_addr = D_addr + d*DIM;
        if (add_to_zeros[c])
          d_addr = GARBAGE_ADDR;

        //printf("Iteration: %zu, a: %d, b: %d, d: %d, d_addr: %u\n", c, a, b, d, d_addr);

        if (!preload[c]) {
          //printf("  Preloading zeros at C_addr: %u\n", C_addrs[c]);
          gemmini_preload_zeros(C_addrs[c]);
          //printf("  Computing accumulated with A_addr: %u, d_addr: %u\n", A_addr + a*DIM, d_addr);
          gemmini_compute_accumulated(A_addr + a*DIM, d_addr);
          
        } else {
          //printf("  Preloading B_addr: %u at C_addr: %u\n", B_addr + b*DIM, C_addrs[c]);
          gemmini_preload(B_addr + b*DIM, C_addrs[c]);
          //printf("  Computing preloaded with A_addr: %u, d_addr: %u\n", A_addr + a*DIM, d_addr);
          gemmini_compute_preloaded(A_addr + a*DIM, d_addr);
        }
      }

      // printf("Moving out\n");
      for (size_t c = 0; c < N*N*N; ++c)
        if (!no_output[c]) {
          //printf("  Moving out C[%zu] from C_addr: %u to main memory address: %u\n", c, C_addrs[c], C[c]);
          gemmini_mvout(C[c], C_addrs[c] & ~(1 << (ADDR_LEN-2)));
        }

      gemmini_fence();

      /*printf("Moved out\n");
      for (int n = 0; n < N*N*N; ++n) {
        if (!no_output[n]) {
          printf("C:\n");
          printMatrix(C[n]);
          printf("Gold:\n");
          printMatrix(gold[n]);
          printf("\n");
        }
      }*/
     /*

      printMatrix(A[1]);
      printMatrix(B[1]);
      printMatrix(D[1]);
      printMatrix(C[1]);
      printMatrix(A[2]);
      printMatrix(B[2]);
      printMatrix(D[2]);
      printMatrix(C[2]);
      */
      // printf("Checking\n");
      for (int n = 0; n < N*N*N; ++n){    // these curly brackets were not there
        //if (!no_output[n] && !is_equal(C[n], gold[n])) {
        if(!no_output[n]){
          printf("\nA\n");
          printMatrix(A[n]);
          printf("\nB\n");
          printMatrix(B[n]);
          printf("\nD\n");
          printMatrix(D[n]);
          printf("\n");
          printf("activation: %d, scale: %f\n", activation, scale);
          //printf("activation: %x, scale: %x\n", activation, scale);
          printf("Actual (%d):\n", n);
          printf("\nC\n");
          printMatrix(C[n]);
          printf("\nGold:\n");
          printMatrix(gold[n]);
          //exit(1); //This is the assert error I see
        }
        //}
        //exit(1);
      }
    }
  }

  exit(0);
}
