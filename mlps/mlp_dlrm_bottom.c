#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#include <time.h>
#endif

#include "include/gemmini.h"
#include "include/gemmini_nn.h"

#include "parameters_dlrm_bottom.h"

int main (int argc, char * argv[]) {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
      perror("mlockall failed");
      exit(1);
    }
#endif

    gemmini_flush(0);

    enum tiled_matmul_type_t tiled_matmul_type;
    if (argc < 2) {
        tiled_matmul_type = WS;
    } else if (strcmp(argv[1], "cpu") == 0) {
        tiled_matmul_type = CPU;
    } else if (strcmp(argv[1], "os") == 0) {
        tiled_matmul_type = OS;
    } else if (strcmp(argv[1], "ws") == 0) {
        tiled_matmul_type = WS;
    } else if (strcmp(argv[1], "-h") == 0) {
        printf("usage: %s [-h] matmul_option [check]\n  matmul_option may be 'os', 'ws', or cpu'\n", argv[0]);
        exit(0);
    } else {
        printf("Unknown command-line argument\n");
        printf("usage: %s [-h] matmul_option [check]\n  matmul_option may be 'os', 'ws', or cpu'\n", argv[0]);
        exit(1);
    }

    bool check;
    if (argc < 3) {
        check = false;
    } else if (strcmp(argv[2], "check") == 0) {
        check = true;
    } else {
        printf("Unknown command-line argument\n");
        printf("usage: %s [-h] matmul_option [check]\n  matmul_option may be 'os', 'ws', or cpu'\n", argv[0]);
        exit(1);
    }

    uint64_t cycles[3] = {0};
    uint64_t start, end;

#ifndef BAREMETAL
    struct timespec _wall_start, _wall_end;
    clock_gettime(CLOCK_MONOTONIC, &_wall_start);
#endif

    /* matmul number: 0 */
    start = read_cycles();

    tiled_matmul_nn_auto(64, 256, 512,
        input_mat, weights0, NULL, inter_results0,
        RELU, 0, false,
        tiled_matmul_type, check, "layer_0");

    end = read_cycles();
    cycles[0] = end-start;

    /* matmul number: 1 */
    start = read_cycles();

    tiled_matmul_nn_auto(64, 128, 256,
        inter_results0, weights1, NULL, inter_results1,
        RELU, 0, false,
        tiled_matmul_type, check, "layer_1");

    end = read_cycles();
    cycles[1] = end-start;

    /* matmul number: 2 */
    start = read_cycles();

    tiled_matmul_nn_auto(64, 64, 128,
        inter_results1, weights2, NULL, inter_results2,
        RELU, 0, false,
        tiled_matmul_type, check, "layer_2");

    end = read_cycles();
    cycles[2] = end-start;

#ifndef BAREMETAL
    clock_gettime(CLOCK_MONOTONIC, &_wall_end);
#endif

    uint64_t overall_cycles = 0;
    for(int cyc = 0; cyc < 3 ; cyc++){
        overall_cycles += cycles[cyc];
        printf("Cycles taken in layer %d: %llu\n", cyc, cycles[cyc]);
    }
    printf("Overall cycles taken: %llu\n", overall_cycles);
#ifndef BAREMETAL
    {
        uint64_t _wall_ns = (uint64_t)(_wall_end.tv_sec - _wall_start.tv_sec) * 1000000000ULL
                          + (uint64_t)(_wall_end.tv_nsec - _wall_start.tv_nsec);
        printf("Wall time: %llu ns\n", (unsigned long long)_wall_ns);
    }
#endif

    return 0;
}
