
#include <stdio.h>
#include "include/gemmini.h"

#define LEN(arr) ((int) (sizeof (arr) / sizeof (arr[0])))

// DLRM Bottom MLP (Naumov et al., 2019 / MLPerf)
// batch size: 64
// before zeropad: 512x256x128x64
// after zeropad: 512x256x128x64
static elem_t input_mat[64][512] row_align(1)= {0};
static elem_t weights0[512][256] row_align(1)= {0};
static elem_t inter_results0[64][256] row_align(1)= {0};
static elem_t weights1[256][128] row_align(1)= {0};
static elem_t inter_results1[64][128] row_align(1)= {0};
static elem_t weights2[128][64] row_align(1)= {0};
static elem_t inter_results2[64][64] row_align(1)= {0};
