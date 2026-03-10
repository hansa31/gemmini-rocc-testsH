// Test 1: Include gemmini headers but NO gemmini instructions
// If this fails -> headers/compilation cause the issue
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini_testutils.h"

int main() {
    printf("TEST1: Headers included OK\n");
    fflush(stdout);
    printf("TEST1: DONE\n");
    return 0;
}
