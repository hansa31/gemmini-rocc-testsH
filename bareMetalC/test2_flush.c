// Test 2: Include gemmini headers + call gemmini_flush
// If this fails -> RoCC custom instruction crashes the process
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini_testutils.h"

int main() {
    printf("TEST2: Before gemmini_flush\n");
    fflush(stdout);
    gemmini_flush(0);
    printf("TEST2: After gemmini_flush OK\n");
    fflush(stdout);
    printf("TEST2: DONE\n");
    return 0;
}
