
#include <stdio.h>
#include "include/gemmini.h"

#define LEN(arr) ((int) (sizeof (arr) / sizeof (arr[0])))

// GPT-2 Small FFN (Radford et al., 2019)
// batch size: 64
// before zeropad: 1024x4096x1024
// after zeropad: 1024x4096x1024
static elem_t input_mat[64][1024] row_align(1)= {0};
static elem_t weights0[1024][4096] row_align(1)= {0};
static elem_t inter_results0[64][4096] row_align(1)= {0};
static elem_t weights1[4096][1024] row_align(1)= {0};
static elem_t inter_results1[64][1024] row_align(1)= {0};
