
#include <stdio.h>
#include "include/gemmini.h"

#define LEN(arr) ((int) (sizeof (arr) / sizeof (arr[0])))

// LeNet-300-100 (LeCun et al., 1998)
// batch size: 64
// before zeropad: 784x300x100x10
// after zeropad: 784x304x112x16
static elem_t input_mat[64][784] row_align(1)= {0};
static elem_t weights0[784][304] row_align(1)= {0};
static elem_t inter_results0[64][304] row_align(1)= {0};
static elem_t weights1[304][112] row_align(1)= {0};
static elem_t inter_results1[64][112] row_align(1)= {0};
static elem_t weights2[112][16] row_align(1)= {0};
static elem_t inter_results2[64][16] row_align(1)= {0};
