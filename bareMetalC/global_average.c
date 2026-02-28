#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini_testutils.h"

#ifndef BAREMETAL

#define BATCHES 4
#define INPUT_DIM 7
#define CHANNELS 2048

#else

#define BATCHES 2
#define INPUT_DIM 3
#define CHANNELS 47

#endif

void init_random(elem_t * buf, int len) {
  for (int i = 0; i < len; i++) {
    buf[i] = rand() % 10;
  }
}

bool is_same(elem_t * x, elem_t * y, int len) {
  for (int i = 0; i < len; i++)
    if (x[i] != y[i])
      return false;
  return true;
}

int main() {
  static elem_t input[BATCHES][INPUT_DIM][INPUT_DIM][CHANNELS];
  static elem_t output[BATCHES][CHANNELS];
  static elem_t gold[BATCHES][CHANNELS];

  init_random((elem_t*)input, BATCHES * INPUT_DIM * INPUT_DIM * CHANNELS);

  printf("CPU average pooling...\n");
  tiled_global_average_auto((elem_t*)input, (elem_t*)gold,
    BATCHES, CHANNELS, INPUT_DIM, CPU);

  printf("Gemmini average pooling...\n");
  tiled_global_average_auto((elem_t*)input, (elem_t*)output,
    BATCHES, CHANNELS, INPUT_DIM, WS);

  if (!is_same((elem_t*)gold, (elem_t*)output, BATCHES * CHANNELS)) {
    printf("Fail\n");

    printf("Input:\n");
    for (int b = 0; b < BATCHES; b++) {
      for (int row = 0; row < INPUT_DIM; row++) {
        printf("{");
        for (int col = 0; col < INPUT_DIM; col++) {
          printf("{");
          for (int ch = 0; ch < CHANNELS; ch++) {
            printf("%d ", input[b][row][col][ch]);
          }
          printf("}");
        }
        printf("}");
      }
      printf("\n");
    }

    printf("Output:\n");
    for (int b = 0; b < BATCHES; b++) {
      for (int ch = 0; ch < CHANNELS; ch++) {
        printf("%d ", output[b][ch]);
      }
      printf("\n");
    }

    printf("Gold:\n");
    for (int b = 0; b < BATCHES; b++) {
      for (int ch = 0; ch < CHANNELS; ch++) {
        printf("%d ", gold[b][ch]);
      }
      printf("\n");
    }

    exit(1);
  }

  exit(0);
}

#ifndef GEMMINI_PARAMS_H
#define GEMMINI_PARAMS_H

#include <stdint.h>
#include <limits.h>

#define XCUSTOM_ACC 3
#define DIM 16
#define ADDR_LEN 32
#define BANK_NUM 4
#define BANK_ROWS 4096
#define ACC_ROWS 1024
#define MAX_BYTES 64
#define MAX_BLOCK_LEN (MAX_BYTES/(DIM*1))
#define MAX_BLOCK_LEN_ACC (MAX_BYTES/(DIM*4))

typedef int8_t elem_t;
static const elem_t elem_t_max = 127;
static const elem_t elem_t_min = -128;
typedef int32_t acc_t;
typedef int64_t full_t;

#define HAS_MVIN_SCALE
typedef float scale_t;
typedef uint32_t scale_t_bits;

typedef int32_t scale_acc_t;
typedef uint32_t scale_acc_t_bits;

typedef float acc_scale_t;
typedef uint32_t acc_scale_t_bits;

#define row_align(blocks) __attribute__((aligned(blocks*DIM*sizeof(elem_t))))
#define row_align_acc(blocks) __attribute__((aligned(blocks*DIM*sizeof(acc_t))))

#define MVIN_SCALE_IDENTITY 1.0

#define ACC_SCALE_IDENTITY 1.0

// Rounding right shift equation: https://riscv.github.io/documents/riscv-v-spec/#_vector_fixed_point_rounding_mode_register_vxrm
#define ROUNDING_RIGHT_SHIFT(x, shift) \
    ((shift) > 0 ? (((x) >> (shift)) + \
        (((shift) == 0 ? 0 : (((x) >> ((shift)-1)) & 1)) & \
             ((((shift) <= 1 ? 0 : ((x) & ((1 << ((shift)-1)) - 1))) != 0) | (((x) >> (shift)) & 1)))) : ((x) << (-(shift))))

#ifdef __cplusplus
#define SAME_TYPE(x) decltype(x)
#else
#define SAME_TYPE(x) typeof(x)
#endif

#define ROUND_NEAR_EVEN(x) \
    ({ const SAME_TYPE(x) x_ = (x); \
         const long long i = x_; \
         const long long next = x_ < 0 ? x_ - 1 : x_ + 1; \
         SAME_TYPE(x) rem = x_ - i; \
         rem = rem < 0 ? -rem : rem; \
         SAME_TYPE(x) result = rem < 0.5 ? i : (rem > 0.5 ? next : ( \
                     i % 2 == 0 ? i : next)); \
         result; })

// Rounding right shift equation: https://riscv.github.io/documents/riscv-v-spec/#_vector_fixed_point_rounding_mode_register_vxrm
#define ROUNDING_RIGHT_SHIFT_BITS(x, shift) \
((shift) > 0 ? (((x) >> (shift)) + \gemmini_testutils
    (((shift) == 0 ? 0 : (((x) >> ((shift)-1)) & 1)) & \
         ((((shift) <= 1 ? 0 : ((x) & ((1 << ((shift)-1)) - 1))) != 0) | (((x) >> (shift)) & 1)))) : ((x) << (-(shift))))

#define ACC_SCALE(x, scale) \
    ({float y = ROUND_NEAR_EVEN((x) * (scale)); y > INT8_MAX ? INT8_MAX : (y < INT8_MIN ? INT8_MIN : (acc_t)y);})

#define MVIN_SCALE(x, scale) \
    ({float y = ROUND_NEAR_EVEN((x) * (scale)); y > INT8_MAX ? INT8_MAX : (y < INT8_MIN ? INT8_MIN : (elem_t)y);})

#define MVIN_SCALE_ACC(x, scale) (x)

#define ACC_SCALE_T_IS_FLOAT
#define ACC_SCALE_EXP_BITS 8
#define ACC_SCALE_SIG_BITS 24

#define ACC_READ_SMALL_WIDTH
#define ACC_READ_FULL_WIDTH
gemmini_testutils
#define HAS_FIRST_LAYER_OPTIMIZATIONS

#endif // GEMMINI_PARAMS_H
