
#include <stdio.h>
#include "include/gemmini.h"

#define LEN(arr) ((int) (sizeof (arr) / sizeof (arr[0])))

// BERT-base FFN (Devlin et al., 2019)
// batch size: 64
// before zeropad: 768x3072x768
// after zeropad: 768x3072x768
static elem_t input_mat[64][768] row_align(1)= {0};
static elem_t weights0[768][3072] row_align(1)= {0};
static elem_t inter_results0[64][3072] row_align(1)= {0};
static elem_t weights1[3072][768] row_align(1)= {0};
static elem_t inter_results1[64][768] row_align(1)= {0};
