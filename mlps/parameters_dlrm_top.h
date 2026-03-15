
#include <stdio.h>
#include "include/gemmini.h"

#define LEN(arr) ((int) (sizeof (arr) / sizeof (arr[0])))

// DLRM Top MLP (Naumov et al., 2019 / MLPerf)
// batch size: 64
// before zeropad: 1024x1024x512x256x1
// after zeropad: 1024x1024x512x256x16
static elem_t input_mat[64][1024] row_align(1)= {0};
static elem_t weights0[1024][1024] row_align(1)= {0};
static elem_t inter_results0[64][1024] row_align(1)= {0};
static elem_t weights1[1024][512] row_align(1)= {0};
static elem_t inter_results1[64][512] row_align(1)= {0};
static elem_t weights2[512][256] row_align(1)= {0};
static elem_t inter_results2[64][256] row_align(1)= {0};
static elem_t weights3[256][16] row_align(1)= {0};
static elem_t inter_results3[64][16] row_align(1)= {0};
