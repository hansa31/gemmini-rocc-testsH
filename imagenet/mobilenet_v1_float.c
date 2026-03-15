#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini.h"
#include "include/gemmini_nn.h"

#include "mobilenet_params_float.h"
#include "images.h"   // Reference test images for A/B comparison

#include <time.h>
#include <stdint.h>

#ifndef CLOCK_MONOTONIC
    #define CLOCK_MONOTONIC CLOCK_REALTIME
#endif

#define TOP_K 10

// ---- Configuration: change these for your dataset ----
#define NUM_IMAGES 50000
#define BATCH_SIZE 4
#define IMAGE_SIZE (224 * 224 * 3)

#define IMAGES_BIN_FILE "imagenet_val_50000.bin"
#define LABELS_TXT_FILE "imagenet_val_50000_labels.txt"
// ------------------------------------------------------

static inline uint64_t get_time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

// Only one batch of images in memory at a time (~600 KB)
static elem_t batch_images[BATCH_SIZE * IMAGE_SIZE];

int main (int argc, char * argv[]) {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
      perror("mlockall failed");
      exit(1);
    }
#endif

    gemmini_flush(0);

    enum tiled_matmul_type_t tiled_matmul_type = WS;

    if (argc < 2) {
        tiled_matmul_type = WS;
    } else if (strcmp(argv[1], "cpu") == 0) {
        tiled_matmul_type = CPU;
    } else if (strcmp(argv[1], "os") == 0) {
        tiled_matmul_type = OS;
    } else if (strcmp(argv[1], "ws") == 0) {
        tiled_matmul_type = WS;
    } else if (strcmp(argv[1], "-h") == 0) {
        printf("usage: %s [-h] matmul_option [conv|matmul] [check]\n  matmul_option may be 'os', 'ws', or 'cpu'\n", argv[0]);
        exit(0);
    } else {
        printf("Unknown command-line argument\n");
        printf("usage: %s [-h] matmul_option [conv|matmul] [check]\n", argv[0]);
        exit(1);
    }

    bool conv = true;

    if (argc < 3) {
        conv = true;
    } else if (strcmp(argv[2], "conv") == 0) {
        conv = true;
    } else if (strcmp(argv[2], "matmul") == 0) {
        conv = false;
    } else {
        printf("Unknown command-line argument\n");
        exit(1);
    }

    bool check = false;

    if (argc < 4) {
        check = false;
    } else if (strcmp(argv[3], "check") == 0) {
        check = true;
    } else {
        printf("Unknown command-line argument\n");
        exit(1);
    }

    printf("\n--- MobileNetV1 Float Streaming Inference ---\n");
    printf("NUM_IMAGES: %d, BATCH_SIZE: %d\n", NUM_IMAGES, BATCH_SIZE);
    printf("tiled_matmul_type: %s\n",
        tiled_matmul_type == CPU ? "CPU" : tiled_matmul_type == OS ? "OS" : "WS");
    printf("conv: %s, check: %s\n", conv ? "true" : "false", check ? "true" : "false");

    // Load all labels (ints are tiny, ~200 KB for 50K)
    int labels[NUM_IMAGES];
    FILE *fp_labels = fopen(LABELS_TXT_FILE, "r");
    if (!fp_labels) {
        printf("Error: cannot open %s\n", LABELS_TXT_FILE);
        exit(1);
    }
    for (int i = 0; i < NUM_IMAGES; i++) {
        if (fscanf(fp_labels, "%d", &labels[i]) != 1) {
            printf("Error: could not read label %d from %s\n", i, LABELS_TXT_FILE);
            fclose(fp_labels);
            exit(1);
        }
    }
    fclose(fp_labels);

    // Open image binary for streaming
    FILE *fp_images = fopen(IMAGES_BIN_FILE, "rb");
    if (!fp_images) {
        printf("Error: cannot open %s\n", IMAGES_BIN_FILE);
        exit(1);
    }

    int num_batches = NUM_IMAGES / BATCH_SIZE;
    int top1_correct = 0;
    int top5_correct = 0;
    int top10_correct = 0;

    // Per-100-image window counters
    int window_top1 = 0;
    int window_top5 = 0;
    int window_top10 = 0;

    setvbuf(stdout, NULL, _IONBF, 0);

    // ===== A/B TEST: Run images.h first to establish baseline =====
    printf("\n===== A/B TEST: Running images.h reference data =====\n");
    {
        elem_t *current_images = (elem_t*)images;  // from images.h
        uint64_t start = get_time_ns();

        // --- Run the full network with images.h data ---
        if (!conv) {
            im2col(conv_1_params.batch_size, conv_1_params.in_channels,
                conv_1_params.in_row_dim, conv_1_params.in_col_dim,
                conv_1_params.I, conv_1_params.K,
                current_images, conv_1_in, &conv_1_params);
            tiled_matmul_nn_auto(conv_1_params.I, conv_1_params.J, conv_1_params.K,
                conv_1_in, conv_1_w, conv_1_b, conv_1_out,
                RELU, conv_1_params.output_scale, true,
                tiled_matmul_type, check, "conv_1");
        } else {
            tiled_conv_auto(
                conv_1_params.batch_size, conv_1_params.in_row_dim, conv_1_params.in_col_dim,
                conv_1_params.in_channels,
                conv_1_params.out_channels, conv_1_params.out_row_dim, conv_1_params.out_col_dim,
                conv_1_params.stride, 1, 1, conv_1_params.padding, conv_1_params.kernel_size,
                false, false, false, false, false,
                (elem_t*)current_images, (elem_t*)conv_1_w, (acc_t*)conv_1_b, (elem_t*)conv_1_out,
                RELU, conv_1_params.output_scale,
                conv_1_params.pool_size, 0, conv_1_params.pool_padding,
                tiled_matmul_type);
        }

        // Print images.h conv_1 stats
        int c1_min = 127, c1_max = -128, c1_nz = 0;
        for (int r = 0; r < conv_1_params.I / 4; r++)
            for (int c = 0; c < conv_1_params.J; c++) {
                int v = conv_1_out[r][c];
                if (v < c1_min) c1_min = v;
                if (v > c1_max) c1_max = v;
                if (v != 0) c1_nz++;
            }
        printf("  [REF] conv_1_out: min=%d, max=%d, nonzero=%d/%d\n",
               c1_min, c1_max, c1_nz, (conv_1_params.I/4)*conv_1_params.J);

        // Run remaining layers (same as main loop)
        tiled_conv_dw_auto(conv_dw_2_params.batch_size, conv_dw_2_params.in_row_dim, conv_dw_2_params.in_col_dim, conv_dw_2_params.in_channels, conv_dw_2_params.out_row_dim, conv_dw_2_params.out_col_dim, conv_dw_2_params.stride, conv_dw_2_params.padding, conv_dw_2_params.kernel_size, (elem_t*)conv_1_out, (elem_t*)conv_dw_2_w, (acc_t*)conv_dw_2_b, (elem_t*)conv_dw_2_out, RELU, conv_dw_2_params.output_scale, conv_dw_2_params.pool_size, 0, conv_dw_2_params.pool_padding, tiled_matmul_type);
        tiled_matmul_nn_auto(conv_3_params.I, conv_3_params.J, conv_3_params.K, conv_dw_2_out, conv_3_w, conv_3_b, conv_3_out, NO_ACTIVATION, conv_3_params.output_scale, true, tiled_matmul_type, check, "conv_3");
        tiled_matmul_nn_auto(conv_4_params.I, conv_4_params.J, conv_4_params.K, conv_3_out, conv_4_w, conv_4_b, conv_4_out, RELU, conv_4_params.output_scale, true, tiled_matmul_type, check, "conv_4");
        tiled_conv_dw_auto(conv_dw_5_params.batch_size, conv_dw_5_params.in_row_dim, conv_dw_5_params.in_col_dim, conv_dw_5_params.in_channels, conv_dw_5_params.out_row_dim, conv_dw_5_params.out_col_dim, conv_dw_5_params.stride, conv_dw_5_params.padding, conv_dw_5_params.kernel_size, (elem_t*)conv_4_out, (elem_t*)conv_dw_5_w, (acc_t*)conv_dw_5_b, (elem_t*)conv_dw_5_out, RELU, conv_dw_5_params.output_scale, conv_dw_5_params.pool_size, 0, conv_dw_5_params.pool_padding, tiled_matmul_type);
        tiled_matmul_nn_auto(conv_6_params.I, conv_6_params.J, conv_6_params.K, conv_dw_5_out, conv_6_w, conv_6_b, conv_6_out, NO_ACTIVATION, conv_6_params.output_scale, true, tiled_matmul_type, check, "conv_6");
        tiled_matmul_nn_auto(conv_7_params.I, conv_7_params.J, conv_7_params.K, conv_6_out, conv_7_w, conv_7_b, conv_7_out, RELU, conv_7_params.output_scale, true, tiled_matmul_type, check, "conv_7");
        tiled_conv_dw_auto(conv_dw_8_params.batch_size, conv_dw_8_params.in_row_dim, conv_dw_8_params.in_col_dim, conv_dw_8_params.in_channels, conv_dw_8_params.out_row_dim, conv_dw_8_params.out_col_dim, conv_dw_8_params.stride, conv_dw_8_params.padding, conv_dw_8_params.kernel_size, (elem_t*)conv_7_out, (elem_t*)conv_dw_8_w, (acc_t*)conv_dw_8_b, (elem_t*)conv_dw_8_out, RELU, conv_dw_8_params.output_scale, conv_dw_8_params.pool_size, 0, conv_dw_8_params.pool_padding, tiled_matmul_type);
        tiled_matmul_nn_auto(conv_9_params.I, conv_9_params.J, conv_9_params.K, conv_dw_8_out, conv_9_w, conv_9_b, conv_9_out, NO_ACTIVATION, conv_9_params.output_scale, true, tiled_matmul_type, check, "conv_9");
        tiled_resadd_auto(conv_9_params.I, conv_9_params.J, conv_9_params.res_scale, MVIN_SCALE_IDENTITY, ACC_SCALE_IDENTITY, conv_6_out, conv_9_out, conv_9_out, false, tiled_matmul_type == CPU ? CPU : WS);
        tiled_matmul_nn_auto(conv_10_params.I, conv_10_params.J, conv_10_params.K, conv_9_out, conv_10_w, conv_10_b, conv_10_out, RELU, conv_10_params.output_scale, true, tiled_matmul_type, check, "conv_10");
        tiled_conv_dw_auto(conv_dw_11_params.batch_size, conv_dw_11_params.in_row_dim, conv_dw_11_params.in_col_dim, conv_dw_11_params.in_channels, conv_dw_11_params.out_row_dim, conv_dw_11_params.out_col_dim, conv_dw_11_params.stride, conv_dw_11_params.padding, conv_dw_11_params.kernel_size, (elem_t*)conv_10_out, (elem_t*)conv_dw_11_w, (acc_t*)conv_dw_11_b, (elem_t*)conv_dw_11_out, RELU, conv_dw_11_params.output_scale, conv_dw_11_params.pool_size, 0, conv_dw_11_params.pool_padding, tiled_matmul_type);
        tiled_matmul_nn_auto(conv_12_params.I, conv_12_params.J, conv_12_params.K, conv_dw_11_out, conv_12_w, conv_12_b, conv_12_out, NO_ACTIVATION, conv_12_params.output_scale, true, tiled_matmul_type, check, "conv_12");
        tiled_matmul_nn_auto(conv_13_params.I, conv_13_params.J, conv_13_params.K, conv_12_out, conv_13_w, conv_13_b, conv_13_out, RELU, conv_13_params.output_scale, true, tiled_matmul_type, check, "conv_13");
        tiled_conv_dw_auto(conv_dw_14_params.batch_size, conv_dw_14_params.in_row_dim, conv_dw_14_params.in_col_dim, conv_dw_14_params.in_channels, conv_dw_14_params.out_row_dim, conv_dw_14_params.out_col_dim, conv_dw_14_params.stride, conv_dw_14_params.padding, conv_dw_14_params.kernel_size, (elem_t*)conv_13_out, (elem_t*)conv_dw_14_w, (acc_t*)conv_dw_14_b, (elem_t*)conv_dw_14_out, RELU, conv_dw_14_params.output_scale, conv_dw_14_params.pool_size, 0, conv_dw_14_params.pool_padding, tiled_matmul_type);
        tiled_matmul_nn_auto(conv_15_params.I, conv_15_params.J, conv_15_params.K, conv_dw_14_out, conv_15_w, conv_15_b, conv_15_out, NO_ACTIVATION, conv_15_params.output_scale, true, tiled_matmul_type, check, "conv_15");
        tiled_resadd_auto(conv_15_params.I, conv_15_params.J, conv_15_params.res_scale, MVIN_SCALE_IDENTITY, ACC_SCALE_IDENTITY, conv_12_out, conv_15_out, conv_15_out, false, tiled_matmul_type == CPU ? CPU : WS);
        tiled_matmul_nn_auto(conv_16_params.I, conv_16_params.J, conv_16_params.K, conv_15_out, conv_16_w, conv_16_b, conv_16_out, RELU, conv_16_params.output_scale, true, tiled_matmul_type, check, "conv_16");
        tiled_conv_dw_auto(conv_dw_17_params.batch_size, conv_dw_17_params.in_row_dim, conv_dw_17_params.in_col_dim, conv_dw_17_params.in_channels, conv_dw_17_params.out_row_dim, conv_dw_17_params.out_col_dim, conv_dw_17_params.stride, conv_dw_17_params.padding, conv_dw_17_params.kernel_size, (elem_t*)conv_16_out, (elem_t*)conv_dw_17_w, (acc_t*)conv_dw_17_b, (elem_t*)conv_dw_17_out, RELU, conv_dw_17_params.output_scale, conv_dw_17_params.pool_size, 0, conv_dw_17_params.pool_padding, tiled_matmul_type);
        tiled_matmul_nn_auto(conv_18_params.I, conv_18_params.J, conv_18_params.K, conv_dw_17_out, conv_18_w, conv_18_b, conv_18_out, NO_ACTIVATION, conv_18_params.output_scale, true, tiled_matmul_type, check, "conv_18");
        tiled_resadd_auto(conv_18_params.I, conv_18_params.J, conv_18_params.res_scale, MVIN_SCALE_IDENTITY, ACC_SCALE_IDENTITY, conv_15_out, conv_18_out, conv_18_out, false, tiled_matmul_type == CPU ? CPU : WS);
        tiled_matmul_nn_auto(conv_19_params.I, conv_19_params.J, conv_19_params.K, conv_18_out, conv_19_w, conv_19_b, conv_19_out, RELU, conv_19_params.output_scale, true, tiled_matmul_type, check, "conv_19");
        tiled_conv_dw_auto(conv_dw_20_params.batch_size, conv_dw_20_params.in_row_dim, conv_dw_20_params.in_col_dim, conv_dw_20_params.in_channels, conv_dw_20_params.out_row_dim, conv_dw_20_params.out_col_dim, conv_dw_20_params.stride, conv_dw_20_params.padding, conv_dw_20_params.kernel_size, (elem_t*)conv_19_out, (elem_t*)conv_dw_20_w, (acc_t*)conv_dw_20_b, (elem_t*)conv_dw_20_out, RELU, conv_dw_20_params.output_scale, conv_dw_20_params.pool_size, 0, conv_dw_20_params.pool_padding, tiled_matmul_type);
        tiled_matmul_nn_auto(conv_21_params.I, conv_21_params.J, conv_21_params.K, conv_dw_20_out, conv_21_w, conv_21_b, conv_21_out, NO_ACTIVATION, conv_21_params.output_scale, true, tiled_matmul_type, check, "conv_21");
        tiled_matmul_nn_auto(conv_22_params.I, conv_22_params.J, conv_22_params.K, conv_21_out, conv_22_w, conv_22_b, conv_22_out, RELU, conv_22_params.output_scale, true, tiled_matmul_type, check, "conv_22");
        tiled_conv_dw_auto(conv_dw_23_params.batch_size, conv_dw_23_params.in_row_dim, conv_dw_23_params.in_col_dim, conv_dw_23_params.in_channels, conv_dw_23_params.out_row_dim, conv_dw_23_params.out_col_dim, conv_dw_23_params.stride, conv_dw_23_params.padding, conv_dw_23_params.kernel_size, (elem_t*)conv_22_out, (elem_t*)conv_dw_23_w, (acc_t*)conv_dw_23_b, (elem_t*)conv_dw_23_out, RELU, conv_dw_23_params.output_scale, conv_dw_23_params.pool_size, 0, conv_dw_23_params.pool_padding, tiled_matmul_type);
        tiled_matmul_nn_auto(conv_24_params.I, conv_24_params.J, conv_24_params.K, conv_dw_23_out, conv_24_w, conv_24_b, conv_24_out, NO_ACTIVATION, conv_24_params.output_scale, true, tiled_matmul_type, check, "conv_24");
        tiled_resadd_auto(conv_24_params.I, conv_24_params.J, conv_24_params.res_scale, MVIN_SCALE_IDENTITY, ACC_SCALE_IDENTITY, conv_21_out, conv_24_out, conv_24_out, false, tiled_matmul_type == CPU ? CPU : WS);
        tiled_matmul_nn_auto(conv_25_params.I, conv_25_params.J, conv_25_params.K, conv_24_out, conv_25_w, conv_25_b, conv_25_out, RELU, conv_25_params.output_scale, true, tiled_matmul_type, check, "conv_25");
        tiled_conv_dw_auto(conv_dw_26_params.batch_size, conv_dw_26_params.in_row_dim, conv_dw_26_params.in_col_dim, conv_dw_26_params.in_channels, conv_dw_26_params.out_row_dim, conv_dw_26_params.out_col_dim, conv_dw_26_params.stride, conv_dw_26_params.padding, conv_dw_26_params.kernel_size, (elem_t*)conv_25_out, (elem_t*)conv_dw_26_w, (acc_t*)conv_dw_26_b, (elem_t*)conv_dw_26_out, RELU, conv_dw_26_params.output_scale, conv_dw_26_params.pool_size, 0, conv_dw_26_params.pool_padding, tiled_matmul_type);
        tiled_matmul_nn_auto(conv_27_params.I, conv_27_params.J, conv_27_params.K, conv_dw_26_out, conv_27_w, conv_27_b, conv_27_out, NO_ACTIVATION, conv_27_params.output_scale, true, tiled_matmul_type, check, "conv_27");
        tiled_resadd_auto(conv_27_params.I, conv_27_params.J, conv_27_params.res_scale, MVIN_SCALE_IDENTITY, ACC_SCALE_IDENTITY, conv_24_out, conv_27_out, conv_27_out, false, tiled_matmul_type == CPU ? CPU : WS);
        tiled_matmul_nn_auto(conv_28_params.I, conv_28_params.J, conv_28_params.K, conv_27_out, conv_28_w, conv_28_b, conv_28_out, RELU, conv_28_params.output_scale, true, tiled_matmul_type, check, "conv_28");
        tiled_conv_dw_auto(conv_dw_29_params.batch_size, conv_dw_29_params.in_row_dim, conv_dw_29_params.in_col_dim, conv_dw_29_params.in_channels, conv_dw_29_params.out_row_dim, conv_dw_29_params.out_col_dim, conv_dw_29_params.stride, conv_dw_29_params.padding, conv_dw_29_params.kernel_size, (elem_t*)conv_28_out, (elem_t*)conv_dw_29_w, (acc_t*)conv_dw_29_b, (elem_t*)conv_dw_29_out, RELU, conv_dw_29_params.output_scale, conv_dw_29_params.pool_size, 0, conv_dw_29_params.pool_padding, tiled_matmul_type);
        tiled_matmul_nn_auto(conv_30_params.I, conv_30_params.J, conv_30_params.K, conv_dw_29_out, conv_30_w, conv_30_b, conv_30_out, NO_ACTIVATION, conv_30_params.output_scale, true, tiled_matmul_type, check, "conv_30");
        tiled_resadd_auto(conv_30_params.I, conv_30_params.J, conv_30_params.res_scale, MVIN_SCALE_IDENTITY, ACC_SCALE_IDENTITY, conv_27_out, conv_30_out, conv_30_out, false, tiled_matmul_type == CPU ? CPU : WS);
        tiled_matmul_nn_auto(conv_31_params.I, conv_31_params.J, conv_31_params.K, conv_30_out, conv_31_w, conv_31_b, conv_31_out, RELU, conv_31_params.output_scale, true, tiled_matmul_type, check, "conv_31");
        tiled_conv_dw_auto(conv_dw_32_params.batch_size, conv_dw_32_params.in_row_dim, conv_dw_32_params.in_col_dim, conv_dw_32_params.in_channels, conv_dw_32_params.out_row_dim, conv_dw_32_params.out_col_dim, conv_dw_32_params.stride, conv_dw_32_params.padding, conv_dw_32_params.kernel_size, (elem_t*)conv_31_out, (elem_t*)conv_dw_32_w, (acc_t*)conv_dw_32_b, (elem_t*)conv_dw_32_out, RELU, conv_dw_32_params.output_scale, conv_dw_32_params.pool_size, 0, conv_dw_32_params.pool_padding, tiled_matmul_type);
        tiled_matmul_nn_auto(conv_33_params.I, conv_33_params.J, conv_33_params.K, conv_dw_32_out, conv_33_w, conv_33_b, conv_33_out, NO_ACTIVATION, conv_33_params.output_scale, true, tiled_matmul_type, check, "conv_33");
        tiled_matmul_nn_auto(conv_34_params.I, conv_34_params.J, conv_34_params.K, conv_33_out, conv_34_w, conv_34_b, conv_34_out, RELU, conv_34_params.output_scale, true, tiled_matmul_type, check, "conv_34");
        tiled_conv_dw_auto(conv_dw_35_params.batch_size, conv_dw_35_params.in_row_dim, conv_dw_35_params.in_col_dim, conv_dw_35_params.in_channels, conv_dw_35_params.out_row_dim, conv_dw_35_params.out_col_dim, conv_dw_35_params.stride, conv_dw_35_params.padding, conv_dw_35_params.kernel_size, (elem_t*)conv_34_out, (elem_t*)conv_dw_35_w, (acc_t*)conv_dw_35_b, (elem_t*)conv_dw_35_out, RELU, conv_dw_35_params.output_scale, conv_dw_35_params.pool_size, 0, conv_dw_35_params.pool_padding, tiled_matmul_type);
        tiled_matmul_nn_auto(conv_36_params.I, conv_36_params.J, conv_36_params.K, conv_dw_35_out, conv_36_w, conv_36_b, conv_36_out, NO_ACTIVATION, conv_36_params.output_scale, true, tiled_matmul_type, check, "conv_36");
        tiled_resadd_auto(conv_36_params.I, conv_36_params.J, conv_36_params.res_scale, MVIN_SCALE_IDENTITY, ACC_SCALE_IDENTITY, conv_33_out, conv_36_out, conv_36_out, false, tiled_matmul_type == CPU ? CPU : WS);
        tiled_matmul_nn_auto(conv_37_params.I, conv_37_params.J, conv_37_params.K, conv_36_out, conv_37_w, conv_37_b, conv_37_out, RELU, conv_37_params.output_scale, true, tiled_matmul_type, check, "conv_37");
        tiled_conv_dw_auto(conv_dw_38_params.batch_size, conv_dw_38_params.in_row_dim, conv_dw_38_params.in_col_dim, conv_dw_38_params.in_channels, conv_dw_38_params.out_row_dim, conv_dw_38_params.out_col_dim, conv_dw_38_params.stride, conv_dw_38_params.padding, conv_dw_38_params.kernel_size, (elem_t*)conv_37_out, (elem_t*)conv_dw_38_w, (acc_t*)conv_dw_38_b, (elem_t*)conv_dw_38_out, RELU, conv_dw_38_params.output_scale, conv_dw_38_params.pool_size, 0, conv_dw_38_params.pool_padding, tiled_matmul_type);
        tiled_matmul_nn_auto(conv_39_params.I, conv_39_params.J, conv_39_params.K, conv_dw_38_out, conv_39_w, conv_39_b, conv_39_out, NO_ACTIVATION, conv_39_params.output_scale, true, tiled_matmul_type, check, "conv_39");
        tiled_resadd_auto(conv_39_params.I, conv_39_params.J, conv_39_params.res_scale, MVIN_SCALE_IDENTITY, ACC_SCALE_IDENTITY, conv_36_out, conv_39_out, conv_39_out, false, tiled_matmul_type == CPU ? CPU : WS);
        tiled_matmul_nn_auto(conv_40_params.I, conv_40_params.J, conv_40_params.K, conv_39_out, conv_40_w, conv_40_b, conv_40_out, RELU, conv_40_params.output_scale, true, tiled_matmul_type, check, "conv_40");
        tiled_conv_dw_auto(conv_dw_41_params.batch_size, conv_dw_41_params.in_row_dim, conv_dw_41_params.in_col_dim, conv_dw_41_params.in_channels, conv_dw_41_params.out_row_dim, conv_dw_41_params.out_col_dim, conv_dw_41_params.stride, conv_dw_41_params.padding, conv_dw_41_params.kernel_size, (elem_t*)conv_40_out, (elem_t*)conv_dw_41_w, (acc_t*)conv_dw_41_b, (elem_t*)conv_dw_41_out, RELU, conv_dw_41_params.output_scale, conv_dw_41_params.pool_size, 0, conv_dw_41_params.pool_padding, tiled_matmul_type);
        tiled_matmul_nn_auto(conv_42_params.I, conv_42_params.J, conv_42_params.K, conv_dw_41_out, conv_42_w, conv_42_b, conv_42_out, NO_ACTIVATION, conv_42_params.output_scale, true, tiled_matmul_type, check, "conv_42");
        tiled_matmul_nn_auto(conv_43_params.I, conv_43_params.J, conv_43_params.K, conv_42_out, conv_43_w, conv_43_b, conv_43_out, RELU, conv_43_params.output_scale, true, tiled_matmul_type, check, "conv_43");
        tiled_conv_dw_auto(conv_dw_44_params.batch_size, conv_dw_44_params.in_row_dim, conv_dw_44_params.in_col_dim, conv_dw_44_params.in_channels, conv_dw_44_params.out_row_dim, conv_dw_44_params.out_col_dim, conv_dw_44_params.stride, conv_dw_44_params.padding, conv_dw_44_params.kernel_size, (elem_t*)conv_43_out, (elem_t*)conv_dw_44_w, (acc_t*)conv_dw_44_b, (elem_t*)conv_dw_44_out, RELU, conv_dw_44_params.output_scale, conv_dw_44_params.pool_size, 0, conv_dw_44_params.pool_padding, tiled_matmul_type);
        tiled_matmul_nn_auto(conv_45_params.I, conv_45_params.J, conv_45_params.K, conv_dw_44_out, conv_45_w, conv_45_b, conv_45_out, NO_ACTIVATION, conv_45_params.output_scale, true, tiled_matmul_type, check, "conv_45");
        tiled_resadd_auto(conv_45_params.I, conv_45_params.J, conv_45_params.res_scale, MVIN_SCALE_IDENTITY, ACC_SCALE_IDENTITY, conv_42_out, conv_45_out, conv_45_out, false, tiled_matmul_type == CPU ? CPU : WS);
        tiled_matmul_nn_auto(conv_46_params.I, conv_46_params.J, conv_46_params.K, conv_45_out, conv_46_w, conv_46_b, conv_46_out, RELU, conv_46_params.output_scale, true, tiled_matmul_type, check, "conv_46");
        tiled_conv_dw_auto(conv_dw_47_params.batch_size, conv_dw_47_params.in_row_dim, conv_dw_47_params.in_col_dim, conv_dw_47_params.in_channels, conv_dw_47_params.out_row_dim, conv_dw_47_params.out_col_dim, conv_dw_47_params.stride, conv_dw_47_params.padding, conv_dw_47_params.kernel_size, (elem_t*)conv_46_out, (elem_t*)conv_dw_47_w, (acc_t*)conv_dw_47_b, (elem_t*)conv_dw_47_out, RELU, conv_dw_47_params.output_scale, conv_dw_47_params.pool_size, 0, conv_dw_47_params.pool_padding, tiled_matmul_type);
        tiled_matmul_nn_auto(conv_48_params.I, conv_48_params.J, conv_48_params.K, conv_dw_47_out, conv_48_w, conv_48_b, conv_48_out, NO_ACTIVATION, conv_48_params.output_scale, true, tiled_matmul_type, check, "conv_48");
        tiled_resadd_auto(conv_48_params.I, conv_48_params.J, conv_48_params.res_scale, MVIN_SCALE_IDENTITY, ACC_SCALE_IDENTITY, conv_45_out, conv_48_out, conv_48_out, false, tiled_matmul_type == CPU ? CPU : WS);
        tiled_matmul_nn_auto(conv_49_params.I, conv_49_params.J, conv_49_params.K, conv_48_out, conv_49_w, conv_49_b, conv_49_out, RELU, conv_49_params.output_scale, true, tiled_matmul_type, check, "conv_49");
        tiled_conv_dw_auto(conv_dw_50_params.batch_size, conv_dw_50_params.in_row_dim, conv_dw_50_params.in_col_dim, conv_dw_50_params.in_channels, conv_dw_50_params.out_row_dim, conv_dw_50_params.out_col_dim, conv_dw_50_params.stride, conv_dw_50_params.padding, conv_dw_50_params.kernel_size, (elem_t*)conv_49_out, (elem_t*)conv_dw_50_w, (acc_t*)conv_dw_50_b, (elem_t*)conv_dw_50_out, RELU, conv_dw_50_params.output_scale, conv_dw_50_params.pool_size, 0, conv_dw_50_params.pool_padding, tiled_matmul_type);
        tiled_matmul_nn_auto(conv_51_params.I, conv_51_params.J, conv_51_params.K, conv_dw_50_out, conv_51_w, conv_51_b, conv_51_out, NO_ACTIVATION, conv_51_params.output_scale, true, tiled_matmul_type, check, "conv_51");
        tiled_matmul_nn_auto(conv_52_params.I, conv_52_params.J, conv_52_params.K, conv_51_out, conv_52_w, conv_52_b, conv_52_out, RELU, conv_52_params.output_scale, true, tiled_matmul_type, check, "conv_52");

        // Print images.h conv_52 and fc_53 stats
        static elem_t ref_average[1280][4] row_align(1);
        for (int batch = 0; batch < conv_52_params.batch_size; batch++)
            for (int channel = 0; channel < conv_52_params.out_channels; channel++) {
                int sum = 0;
                for (int row = 0; row < conv_52_params.out_row_dim; row++)
                    for (int col = 0; col < conv_52_params.out_col_dim; col++) {
                        size_t r = batch * conv_52_params.out_row_dim * conv_52_params.out_col_dim + row * conv_52_params.out_col_dim + col;
                        sum += conv_52_out[r][channel];
                    }
                const int count = conv_52_params.out_row_dim * conv_52_params.out_col_dim;
                ref_average[channel][batch] = (sum + count/2) / count;
            }

        int c52_min = 127, c52_max = -128, c52_nz = 0;
        for (int r = 0; r < conv_52_params.I / 4; r++)
            for (int c = 0; c < conv_52_params.J; c++) {
                int v = conv_52_out[r][c];
                if (v < c52_min) c52_min = v;
                if (v > c52_max) c52_max = v;
                if (v != 0) c52_nz++;
            }
        printf("  [REF] conv_52_out: min=%d, max=%d, nonzero=%d/%d\n",
               c52_min, c52_max, c52_nz, (conv_52_params.I/4)*conv_52_params.J);

        int avg_min = 127, avg_max = -128, avg_nz = 0;
        for (int c = 0; c < 1280; c++) {
            int v = ref_average[c][0];
            if (v < avg_min) avg_min = v;
            if (v > avg_max) avg_max = v;
            if (v != 0) avg_nz++;
        }
        printf("  [REF] average: min=%d, max=%d, nonzero=%d/1280\n", avg_min, avg_max, avg_nz);

        tiled_matmul_nn_auto(fc_53_params.I, fc_53_params.J, fc_53_params.K,
            fc_53_w, ref_average, fc_53_b, fc_53_out,
            NO_ACTIVATION, fc_53_params.output_scale, false,
            tiled_matmul_type, check, "fc_53");

        int fc_min = 127, fc_max = -128;
        for (int i = 0; i < 1000; i++) {
            int v = fc_53_out[i][0];
            if (v < fc_min) fc_min = v;
            if (v > fc_max) fc_max = v;
        }
        printf("  [REF] fc_53_out: min=%d, max=%d\n", fc_min, fc_max);

        // Print predictions for images.h
        for (int batch = 0; batch < 4; batch++) {
            int max_idx = 0;
            elem_t max_val = fc_53_out[0][batch];
            for (int i = 1; i < 1000; i++) {
                if (fc_53_out[i][batch] > max_val) {
                    max_val = fc_53_out[i][batch];
                    max_idx = i;
                }
            }
            printf("  [REF] Image %d: pred=%d (score=%d)\n", batch, max_idx, max_val);
        }
        int ref_correct[] = {75, 900, 125, 897};
        printf("  [REF] Expected: {75, 900, 125, 897}\n");
        uint64_t ref_end = get_time_ns();
        printf("  [REF] Time: %llu ns\n", (unsigned long long)(ref_end - start));
    }
    printf("===== END A/B TEST =====\n\n");

    for (int batch_idx = 0; batch_idx < num_batches; batch_idx++) {

        // Stream one batch from disk
        size_t items_read = fread(batch_images, sizeof(elem_t), BATCH_SIZE * IMAGE_SIZE, fp_images);
        if (items_read != (size_t)(BATCH_SIZE * IMAGE_SIZE)) {
            printf("Warning: short read at batch %d (got %zu of %d bytes), stopping\n",
                   batch_idx, items_read, BATCH_SIZE * IMAGE_SIZE);
            break;
        }

        elem_t *current_images = batch_images;

        uint64_t start = get_time_ns();

        // ====================== conv_1 ======================
        if (!conv) {
            im2col(conv_1_params.batch_size, conv_1_params.in_channels,
                conv_1_params.in_row_dim, conv_1_params.in_col_dim,
                conv_1_params.I, conv_1_params.K,
                current_images, conv_1_in, &conv_1_params);

            tiled_matmul_nn_auto(conv_1_params.I, conv_1_params.J, conv_1_params.K,
                conv_1_in, conv_1_w, conv_1_b, conv_1_out,
                RELU, conv_1_params.output_scale, true,
                tiled_matmul_type, check, "conv_1");
        } else {
            tiled_conv_auto(
                conv_1_params.batch_size, conv_1_params.in_row_dim, conv_1_params.in_col_dim,
                conv_1_params.in_channels,
                conv_1_params.out_channels, conv_1_params.out_row_dim, conv_1_params.out_col_dim,
                conv_1_params.stride, 1, 1, conv_1_params.padding, conv_1_params.kernel_size,
                false, false, false, false, false,

                (elem_t*)current_images, (elem_t*)conv_1_w, (acc_t*)conv_1_b, (elem_t*)conv_1_out,

                RELU, conv_1_params.output_scale,
                conv_1_params.pool_size, 0, conv_1_params.pool_padding,

                tiled_matmul_type);
        }

        // ====================== conv_dw_2 ======================
        if (!conv) {
            conv_dw_with_col2im(conv_1_params.I, conv_1_params.J, conv_dw_2_params.I, conv_dw_2_params.J,
                conv_dw_2_params.batch_size, conv_dw_2_params.in_channels,
                conv_dw_2_params.out_row_dim, conv_dw_2_params.out_col_dim,
                conv_dw_2_params.kernel_size,
                conv_1_out, conv_dw_2_w, conv_dw_2_b, conv_dw_2_out, &conv_dw_2_params);
        } else {
            tiled_conv_dw_auto(
                conv_dw_2_params.batch_size, conv_dw_2_params.in_row_dim, conv_dw_2_params.in_col_dim,
                conv_dw_2_params.in_channels,
                conv_dw_2_params.out_row_dim, conv_dw_2_params.out_col_dim,
                conv_dw_2_params.stride, conv_dw_2_params.padding, conv_dw_2_params.kernel_size,

                (elem_t*)conv_1_out, (elem_t*)conv_dw_2_w, (acc_t*)conv_dw_2_b, (elem_t*)conv_dw_2_out,

                RELU, conv_dw_2_params.output_scale,
                conv_dw_2_params.pool_size, 0, conv_dw_2_params.pool_padding,

                tiled_matmul_type);
        }

        // ====================== conv_3 ======================
        tiled_matmul_nn_auto(conv_3_params.I, conv_3_params.J, conv_3_params.K,
            conv_dw_2_out, conv_3_w, conv_3_b, conv_3_out,
            NO_ACTIVATION, conv_3_params.output_scale, true,
            tiled_matmul_type, check, "conv_3");

        // ====================== conv_4 ======================
        tiled_matmul_nn_auto(conv_4_params.I, conv_4_params.J, conv_4_params.K,
            conv_3_out, conv_4_w, conv_4_b, conv_4_out,
            RELU, conv_4_params.output_scale, true,
            tiled_matmul_type, check, "conv_4");

        // ====================== conv_dw_5 ======================
        if (!conv) {
            conv_dw_with_col2im(conv_4_params.I, conv_4_params.J, conv_dw_5_params.I, conv_dw_5_params.J,
                conv_dw_5_params.batch_size, conv_dw_5_params.in_channels,
                conv_dw_5_params.out_row_dim, conv_dw_5_params.out_col_dim,
                conv_dw_5_params.kernel_size,
                conv_4_out, conv_dw_5_w, conv_dw_5_b, conv_dw_5_out, &conv_dw_5_params);
        } else {
            tiled_conv_dw_auto(
                conv_dw_5_params.batch_size, conv_dw_5_params.in_row_dim, conv_dw_5_params.in_col_dim,
                conv_dw_5_params.in_channels,
                conv_dw_5_params.out_row_dim, conv_dw_5_params.out_col_dim,
                conv_dw_5_params.stride, conv_dw_5_params.padding, conv_dw_5_params.kernel_size,

                (elem_t*)conv_4_out, (elem_t*)conv_dw_5_w, (acc_t*)conv_dw_5_b, (elem_t*)conv_dw_5_out,

                RELU, conv_dw_5_params.output_scale,
                conv_dw_5_params.pool_size, 0, conv_dw_5_params.pool_padding,

                tiled_matmul_type);
        }

        // ====================== conv_6 ======================
        tiled_matmul_nn_auto(conv_6_params.I, conv_6_params.J, conv_6_params.K,
            conv_dw_5_out, conv_6_w, conv_6_b, conv_6_out,
            NO_ACTIVATION, conv_6_params.output_scale, true,
            tiled_matmul_type, check, "conv_6");

        // ====================== conv_7 ======================
        tiled_matmul_nn_auto(conv_7_params.I, conv_7_params.J, conv_7_params.K,
            conv_6_out, conv_7_w, conv_7_b, conv_7_out,
            RELU, conv_7_params.output_scale, true,
            tiled_matmul_type, check, "conv_7");

        // ====================== conv_dw_8 ======================
        if (!conv) {
            conv_dw_with_col2im(conv_7_params.I, conv_7_params.J, conv_dw_8_params.I, conv_dw_8_params.J,
                conv_dw_8_params.batch_size, conv_dw_8_params.in_channels,
                conv_dw_8_params.out_row_dim, conv_dw_8_params.out_col_dim,
                conv_dw_8_params.kernel_size,
                conv_7_out, conv_dw_8_w, conv_dw_8_b, conv_dw_8_out, &conv_dw_8_params);
        } else {
            tiled_conv_dw_auto(
                conv_dw_8_params.batch_size, conv_dw_8_params.in_row_dim, conv_dw_8_params.in_col_dim,
                conv_dw_8_params.in_channels,
                conv_dw_8_params.out_row_dim, conv_dw_8_params.out_col_dim,
                conv_dw_8_params.stride, conv_dw_8_params.padding, conv_dw_8_params.kernel_size,

                (elem_t*)conv_7_out, (elem_t*)conv_dw_8_w, (acc_t*)conv_dw_8_b, (elem_t*)conv_dw_8_out,

                RELU, conv_dw_8_params.output_scale,
                conv_dw_8_params.pool_size, 0, conv_dw_8_params.pool_padding,

                tiled_matmul_type);
        }

        // ====================== conv_9 ======================
        tiled_matmul_nn_auto(conv_9_params.I, conv_9_params.J, conv_9_params.K,
            conv_dw_8_out, conv_9_w, conv_9_b, conv_9_out,
            NO_ACTIVATION, conv_9_params.output_scale, true,
            tiled_matmul_type, check, "conv_9");

        // ====================== res_add (conv_6 + conv_9) ======================
        tiled_resadd_auto(conv_9_params.I, conv_9_params.J,
            conv_9_params.res_scale,
            MVIN_SCALE_IDENTITY,
            ACC_SCALE_IDENTITY,
            conv_6_out,
            conv_9_out,
            conv_9_out,
            false,
            tiled_matmul_type == CPU ? CPU : WS);

        // ====================== conv_10 ======================
        tiled_matmul_nn_auto(conv_10_params.I, conv_10_params.J, conv_10_params.K,
            conv_9_out, conv_10_w, conv_10_b, conv_10_out,
            RELU, conv_10_params.output_scale, true,
            tiled_matmul_type, check, "conv_10");

        // ====================== conv_dw_11 ======================
        if (!conv) {
            conv_dw_with_col2im(conv_10_params.I, conv_10_params.J, conv_dw_11_params.I, conv_dw_11_params.J,
                conv_dw_11_params.batch_size, conv_dw_11_params.in_channels,
                conv_dw_11_params.out_row_dim, conv_dw_11_params.out_col_dim,
                conv_dw_11_params.kernel_size,
                conv_10_out, conv_dw_11_w, conv_dw_11_b, conv_dw_11_out, &conv_dw_11_params);
        } else {
            tiled_conv_dw_auto(
                conv_dw_11_params.batch_size, conv_dw_11_params.in_row_dim, conv_dw_11_params.in_col_dim,
                conv_dw_11_params.in_channels,
                conv_dw_11_params.out_row_dim, conv_dw_11_params.out_col_dim,
                conv_dw_11_params.stride, conv_dw_11_params.padding, conv_dw_11_params.kernel_size,

                (elem_t*)conv_10_out, (elem_t*)conv_dw_11_w, (acc_t*)conv_dw_11_b, (elem_t*)conv_dw_11_out,

                RELU, conv_dw_11_params.output_scale,
                conv_dw_11_params.pool_size, 0, conv_dw_11_params.pool_padding,

                tiled_matmul_type);
        }

        // ====================== conv_12 ======================
        tiled_matmul_nn_auto(conv_12_params.I, conv_12_params.J, conv_12_params.K,
            conv_dw_11_out, conv_12_w, conv_12_b, conv_12_out,
            NO_ACTIVATION, conv_12_params.output_scale, true,
            tiled_matmul_type, check, "conv_12");

        // ====================== conv_13 ======================
        tiled_matmul_nn_auto(conv_13_params.I, conv_13_params.J, conv_13_params.K,
            conv_12_out, conv_13_w, conv_13_b, conv_13_out,
            RELU, conv_13_params.output_scale, true,
            tiled_matmul_type, check, "conv_13");

        // ====================== conv_dw_14 ======================
        if (!conv) {
            conv_dw_with_col2im(conv_13_params.I, conv_13_params.J, conv_dw_14_params.I, conv_dw_14_params.J,
                conv_dw_14_params.batch_size, conv_dw_14_params.in_channels,
                conv_dw_14_params.out_row_dim, conv_dw_14_params.out_col_dim,
                conv_dw_14_params.kernel_size,
                conv_13_out, conv_dw_14_w, conv_dw_14_b, conv_dw_14_out, &conv_dw_14_params);
        } else {
            tiled_conv_dw_auto(
                conv_dw_14_params.batch_size, conv_dw_14_params.in_row_dim, conv_dw_14_params.in_col_dim,
                conv_dw_14_params.in_channels,
                conv_dw_14_params.out_row_dim, conv_dw_14_params.out_col_dim,
                conv_dw_14_params.stride, conv_dw_14_params.padding, conv_dw_14_params.kernel_size,

                (elem_t*)conv_13_out, (elem_t*)conv_dw_14_w, (acc_t*)conv_dw_14_b, (elem_t*)conv_dw_14_out,

                RELU, conv_dw_14_params.output_scale,
                conv_dw_14_params.pool_size, 0, conv_dw_14_params.pool_padding,

                tiled_matmul_type);
        }

        // ====================== conv_15 ======================
        tiled_matmul_nn_auto(conv_15_params.I, conv_15_params.J, conv_15_params.K,
            conv_dw_14_out, conv_15_w, conv_15_b, conv_15_out,
            NO_ACTIVATION, conv_15_params.output_scale, true,
            tiled_matmul_type, check, "conv_15");

        // ====================== res_add (conv_12 + conv_15) ======================
        tiled_resadd_auto(conv_15_params.I, conv_15_params.J,
            conv_15_params.res_scale,
            MVIN_SCALE_IDENTITY,
            ACC_SCALE_IDENTITY,
            conv_12_out,
            conv_15_out,
            conv_15_out,
            false,
            tiled_matmul_type == CPU ? CPU : WS);

        // ====================== conv_16 ======================
        tiled_matmul_nn_auto(conv_16_params.I, conv_16_params.J, conv_16_params.K,
            conv_15_out, conv_16_w, conv_16_b, conv_16_out,
            RELU, conv_16_params.output_scale, true,
            tiled_matmul_type, check, "conv_16");

        // ====================== conv_dw_17 ======================
        if (!conv) {
            conv_dw_with_col2im(conv_16_params.I, conv_16_params.J, conv_dw_17_params.I, conv_dw_17_params.J,
                conv_dw_17_params.batch_size, conv_dw_17_params.in_channels,
                conv_dw_17_params.out_row_dim, conv_dw_17_params.out_col_dim,
                conv_dw_17_params.kernel_size,
                conv_16_out, conv_dw_17_w, conv_dw_17_b, conv_dw_17_out, &conv_dw_17_params);
        } else {
            tiled_conv_dw_auto(
                conv_dw_17_params.batch_size, conv_dw_17_params.in_row_dim, conv_dw_17_params.in_col_dim,
                conv_dw_17_params.in_channels,
                conv_dw_17_params.out_row_dim, conv_dw_17_params.out_col_dim,
                conv_dw_17_params.stride, conv_dw_17_params.padding, conv_dw_17_params.kernel_size,

                (elem_t*)conv_16_out, (elem_t*)conv_dw_17_w, (acc_t*)conv_dw_17_b, (elem_t*)conv_dw_17_out,

                RELU, conv_dw_17_params.output_scale,
                conv_dw_17_params.pool_size, 0, conv_dw_17_params.pool_padding,

                tiled_matmul_type);
        }

        // ====================== conv_18 ======================
        tiled_matmul_nn_auto(conv_18_params.I, conv_18_params.J, conv_18_params.K,
            conv_dw_17_out, conv_18_w, conv_18_b, conv_18_out,
            NO_ACTIVATION, conv_18_params.output_scale, true,
            tiled_matmul_type, check, "conv_18");

        // ====================== res_add (conv_15 + conv_18) ======================
        tiled_resadd_auto(conv_18_params.I, conv_18_params.J,
            conv_18_params.res_scale,
            MVIN_SCALE_IDENTITY,
            ACC_SCALE_IDENTITY,
            conv_15_out,
            conv_18_out,
            conv_18_out,
            false,
            tiled_matmul_type == CPU ? CPU : WS);

        // ====================== conv_19 ======================
        tiled_matmul_nn_auto(conv_19_params.I, conv_19_params.J, conv_19_params.K,
            conv_18_out, conv_19_w, conv_19_b, conv_19_out,
            RELU, conv_19_params.output_scale, true,
            tiled_matmul_type, check, "conv_19");

        // ====================== conv_dw_20 ======================
        if (!conv) {
            conv_dw_with_col2im(conv_19_params.I, conv_19_params.J, conv_dw_20_params.I, conv_dw_20_params.J,
                conv_dw_20_params.batch_size, conv_dw_20_params.in_channels,
                conv_dw_20_params.out_row_dim, conv_dw_20_params.out_col_dim,
                conv_dw_20_params.kernel_size,
                conv_19_out, conv_dw_20_w, conv_dw_20_b, conv_dw_20_out, &conv_dw_20_params);
        } else {
            tiled_conv_dw_auto(
                conv_dw_20_params.batch_size, conv_dw_20_params.in_row_dim, conv_dw_20_params.in_col_dim,
                conv_dw_20_params.in_channels,
                conv_dw_20_params.out_row_dim, conv_dw_20_params.out_col_dim,
                conv_dw_20_params.stride, conv_dw_20_params.padding, conv_dw_20_params.kernel_size,

                (elem_t*)conv_19_out, (elem_t*)conv_dw_20_w, (acc_t*)conv_dw_20_b, (elem_t*)conv_dw_20_out,

                RELU, conv_dw_20_params.output_scale,
                conv_dw_20_params.pool_size, 0, conv_dw_20_params.pool_padding,

                tiled_matmul_type);
        }

        // ====================== conv_21 ======================
        tiled_matmul_nn_auto(conv_21_params.I, conv_21_params.J, conv_21_params.K,
            conv_dw_20_out, conv_21_w, conv_21_b, conv_21_out,
            NO_ACTIVATION, conv_21_params.output_scale, true,
            tiled_matmul_type, check, "conv_21");

        // ====================== conv_22 ======================
        tiled_matmul_nn_auto(conv_22_params.I, conv_22_params.J, conv_22_params.K,
            conv_21_out, conv_22_w, conv_22_b, conv_22_out,
            RELU, conv_22_params.output_scale, true,
            tiled_matmul_type, check, "conv_22");

        // ====================== conv_dw_23 ======================
        if (!conv) {
            conv_dw_with_col2im(conv_22_params.I, conv_22_params.J, conv_dw_23_params.I, conv_dw_23_params.J,
                conv_dw_23_params.batch_size, conv_dw_23_params.in_channels,
                conv_dw_23_params.out_row_dim, conv_dw_23_params.out_col_dim,
                conv_dw_23_params.kernel_size,
                conv_22_out, conv_dw_23_w, conv_dw_23_b, conv_dw_23_out, &conv_dw_23_params);
        } else {
            tiled_conv_dw_auto(
                conv_dw_23_params.batch_size, conv_dw_23_params.in_row_dim, conv_dw_23_params.in_col_dim,
                conv_dw_23_params.in_channels,
                conv_dw_23_params.out_row_dim, conv_dw_23_params.out_col_dim,
                conv_dw_23_params.stride, conv_dw_23_params.padding, conv_dw_23_params.kernel_size,

                (elem_t*)conv_22_out, (elem_t*)conv_dw_23_w, (acc_t*)conv_dw_23_b, (elem_t*)conv_dw_23_out,

                RELU, conv_dw_23_params.output_scale,
                conv_dw_23_params.pool_size, 0, conv_dw_23_params.pool_padding,

                tiled_matmul_type);
        }

        // ====================== conv_24 ======================
        tiled_matmul_nn_auto(conv_24_params.I, conv_24_params.J, conv_24_params.K,
            conv_dw_23_out, conv_24_w, conv_24_b, conv_24_out,
            NO_ACTIVATION, conv_24_params.output_scale, true,
            tiled_matmul_type, check, "conv_24");

        // ====================== res_add (conv_21 + conv_24) ======================
        tiled_resadd_auto(conv_24_params.I, conv_24_params.J,
            conv_24_params.res_scale,
            MVIN_SCALE_IDENTITY,
            ACC_SCALE_IDENTITY,
            conv_21_out,
            conv_24_out,
            conv_24_out,
            false,
            tiled_matmul_type == CPU ? CPU : WS);

        // ====================== conv_25 ======================
        tiled_matmul_nn_auto(conv_25_params.I, conv_25_params.J, conv_25_params.K,
            conv_24_out, conv_25_w, conv_25_b, conv_25_out,
            RELU, conv_25_params.output_scale, true,
            tiled_matmul_type, check, "conv_25");

        // ====================== conv_dw_26 ======================
        if (!conv) {
            conv_dw_with_col2im(conv_25_params.I, conv_25_params.J, conv_dw_26_params.I, conv_dw_26_params.J,
                conv_dw_26_params.batch_size, conv_dw_26_params.in_channels,
                conv_dw_26_params.out_row_dim, conv_dw_26_params.out_col_dim,
                conv_dw_26_params.kernel_size,
                conv_25_out, conv_dw_26_w, conv_dw_26_b, conv_dw_26_out, &conv_dw_26_params);
        } else {
            tiled_conv_dw_auto(
                conv_dw_26_params.batch_size, conv_dw_26_params.in_row_dim, conv_dw_26_params.in_col_dim,
                conv_dw_26_params.in_channels,
                conv_dw_26_params.out_row_dim, conv_dw_26_params.out_col_dim,
                conv_dw_26_params.stride, conv_dw_26_params.padding, conv_dw_26_params.kernel_size,

                (elem_t*)conv_25_out, (elem_t*)conv_dw_26_w, (acc_t*)conv_dw_26_b, (elem_t*)conv_dw_26_out,

                RELU, conv_dw_26_params.output_scale,
                conv_dw_26_params.pool_size, 0, conv_dw_26_params.pool_padding,

                tiled_matmul_type);
        }

        // ====================== conv_27 ======================
        tiled_matmul_nn_auto(conv_27_params.I, conv_27_params.J, conv_27_params.K,
            conv_dw_26_out, conv_27_w, conv_27_b, conv_27_out,
            NO_ACTIVATION, conv_27_params.output_scale, true,
            tiled_matmul_type, check, "conv_27");

        // ====================== res_add (conv_24 + conv_27) ======================
        tiled_resadd_auto(conv_27_params.I, conv_27_params.J,
            conv_27_params.res_scale,
            MVIN_SCALE_IDENTITY,
            ACC_SCALE_IDENTITY,
            conv_24_out,
            conv_27_out,
            conv_27_out,
            false,
            tiled_matmul_type == CPU ? CPU : WS);

        // ====================== conv_28 ======================
        tiled_matmul_nn_auto(conv_28_params.I, conv_28_params.J, conv_28_params.K,
            conv_27_out, conv_28_w, conv_28_b, conv_28_out,
            RELU, conv_28_params.output_scale, true,
            tiled_matmul_type, check, "conv_28");

        // ====================== conv_dw_29 ======================
        if (!conv) {
            conv_dw_with_col2im(conv_28_params.I, conv_28_params.J, conv_dw_29_params.I, conv_dw_29_params.J,
                conv_dw_29_params.batch_size, conv_dw_29_params.in_channels,
                conv_dw_29_params.out_row_dim, conv_dw_29_params.out_col_dim,
                conv_dw_29_params.kernel_size,
                conv_28_out, conv_dw_29_w, conv_dw_29_b, conv_dw_29_out, &conv_dw_29_params);
        } else {
            tiled_conv_dw_auto(
                conv_dw_29_params.batch_size, conv_dw_29_params.in_row_dim, conv_dw_29_params.in_col_dim,
                conv_dw_29_params.in_channels,
                conv_dw_29_params.out_row_dim, conv_dw_29_params.out_col_dim,
                conv_dw_29_params.stride, conv_dw_29_params.padding, conv_dw_29_params.kernel_size,

                (elem_t*)conv_28_out, (elem_t*)conv_dw_29_w, (acc_t*)conv_dw_29_b, (elem_t*)conv_dw_29_out,

                RELU, conv_dw_29_params.output_scale,
                conv_dw_29_params.pool_size, 0, conv_dw_29_params.pool_padding,

                tiled_matmul_type);
        }

        // ====================== conv_30 ======================
        tiled_matmul_nn_auto(conv_30_params.I, conv_30_params.J, conv_30_params.K,
            conv_dw_29_out, conv_30_w, conv_30_b, conv_30_out,
            NO_ACTIVATION, conv_30_params.output_scale, true,
            tiled_matmul_type, check, "conv_30");

        // ====================== res_add (conv_27 + conv_30) ======================
        tiled_resadd_auto(conv_30_params.I, conv_30_params.J,
            conv_30_params.res_scale,
            MVIN_SCALE_IDENTITY,
            ACC_SCALE_IDENTITY,
            conv_27_out,
            conv_30_out,
            conv_30_out,
            false,
            tiled_matmul_type == CPU ? CPU : WS);

        // ====================== conv_31 ======================
        tiled_matmul_nn_auto(conv_31_params.I, conv_31_params.J, conv_31_params.K,
            conv_30_out, conv_31_w, conv_31_b, conv_31_out,
            RELU, conv_31_params.output_scale, true,
            tiled_matmul_type, check, "conv_31");

        // ====================== conv_dw_32 ======================
        if (!conv) {
            conv_dw_with_col2im(conv_31_params.I, conv_31_params.J, conv_dw_32_params.I, conv_dw_32_params.J,
                conv_dw_32_params.batch_size, conv_dw_32_params.in_channels,
                conv_dw_32_params.out_row_dim, conv_dw_32_params.out_col_dim,
                conv_dw_32_params.kernel_size,
                conv_31_out, conv_dw_32_w, conv_dw_32_b, conv_dw_32_out, &conv_dw_32_params);
        } else {
            tiled_conv_dw_auto(
                conv_dw_32_params.batch_size, conv_dw_32_params.in_row_dim, conv_dw_32_params.in_col_dim,
                conv_dw_32_params.in_channels,
                conv_dw_32_params.out_row_dim, conv_dw_32_params.out_col_dim,
                conv_dw_32_params.stride, conv_dw_32_params.padding, conv_dw_32_params.kernel_size,

                (elem_t*)conv_31_out, (elem_t*)conv_dw_32_w, (acc_t*)conv_dw_32_b, (elem_t*)conv_dw_32_out,

                RELU, conv_dw_32_params.output_scale,
                conv_dw_32_params.pool_size, 0, conv_dw_32_params.pool_padding,

                tiled_matmul_type);
        }

        // ====================== conv_33 ======================
        tiled_matmul_nn_auto(conv_33_params.I, conv_33_params.J, conv_33_params.K,
            conv_dw_32_out, conv_33_w, conv_33_b, conv_33_out,
            NO_ACTIVATION, conv_33_params.output_scale, true,
            tiled_matmul_type, check, "conv_33");

        // ====================== conv_34 ======================
        tiled_matmul_nn_auto(conv_34_params.I, conv_34_params.J, conv_34_params.K,
            conv_33_out, conv_34_w, conv_34_b, conv_34_out,
            RELU, conv_34_params.output_scale, true,
            tiled_matmul_type, check, "conv_34");

        // ====================== conv_dw_35 ======================
        if (!conv) {
            conv_dw_with_col2im(conv_34_params.I, conv_34_params.J, conv_dw_35_params.I, conv_dw_35_params.J,
                conv_dw_35_params.batch_size, conv_dw_35_params.in_channels,
                conv_dw_35_params.out_row_dim, conv_dw_35_params.out_col_dim,
                conv_dw_35_params.kernel_size,
                conv_34_out, conv_dw_35_w, conv_dw_35_b, conv_dw_35_out, &conv_dw_35_params);
        } else {
            tiled_conv_dw_auto(
                conv_dw_35_params.batch_size, conv_dw_35_params.in_row_dim, conv_dw_35_params.in_col_dim,
                conv_dw_35_params.in_channels,
                conv_dw_35_params.out_row_dim, conv_dw_35_params.out_col_dim,
                conv_dw_35_params.stride, conv_dw_35_params.padding, conv_dw_35_params.kernel_size,

                (elem_t*)conv_34_out, (elem_t*)conv_dw_35_w, (acc_t*)conv_dw_35_b, (elem_t*)conv_dw_35_out,

                RELU, conv_dw_35_params.output_scale,
                conv_dw_35_params.pool_size, 0, conv_dw_35_params.pool_padding,

                tiled_matmul_type);
        }

        // ====================== conv_36 ======================
        tiled_matmul_nn_auto(conv_36_params.I, conv_36_params.J, conv_36_params.K,
            conv_dw_35_out, conv_36_w, conv_36_b, conv_36_out,
            NO_ACTIVATION, conv_36_params.output_scale, true,
            tiled_matmul_type, check, "conv_36");

        // ====================== res_add (conv_33 + conv_36) ======================
        tiled_resadd_auto(conv_36_params.I, conv_36_params.J,
            conv_36_params.res_scale,
            MVIN_SCALE_IDENTITY,
            ACC_SCALE_IDENTITY,
            conv_33_out,
            conv_36_out,
            conv_36_out,
            false,
            tiled_matmul_type == CPU ? CPU : WS);

        // ====================== conv_37 ======================
        tiled_matmul_nn_auto(conv_37_params.I, conv_37_params.J, conv_37_params.K,
            conv_36_out, conv_37_w, conv_37_b, conv_37_out,
            RELU, conv_37_params.output_scale, true,
            tiled_matmul_type, check, "conv_37");

        // ====================== conv_dw_38 ======================
        if (!conv) {
            conv_dw_with_col2im(conv_37_params.I, conv_37_params.J, conv_dw_38_params.I, conv_dw_38_params.J,
                conv_dw_38_params.batch_size, conv_dw_38_params.in_channels,
                conv_dw_38_params.out_row_dim, conv_dw_38_params.out_col_dim,
                conv_dw_38_params.kernel_size,
                conv_37_out, conv_dw_38_w, conv_dw_38_b, conv_dw_38_out, &conv_dw_38_params);
        } else {
            tiled_conv_dw_auto(
                conv_dw_38_params.batch_size, conv_dw_38_params.in_row_dim, conv_dw_38_params.in_col_dim,
                conv_dw_38_params.in_channels,
                conv_dw_38_params.out_row_dim, conv_dw_38_params.out_col_dim,
                conv_dw_38_params.stride, conv_dw_38_params.padding, conv_dw_38_params.kernel_size,

                (elem_t*)conv_37_out, (elem_t*)conv_dw_38_w, (acc_t*)conv_dw_38_b, (elem_t*)conv_dw_38_out,

                RELU, conv_dw_38_params.output_scale,
                conv_dw_38_params.pool_size, 0, conv_dw_38_params.pool_padding,

                tiled_matmul_type);
        }

        // ====================== conv_39 ======================
        tiled_matmul_nn_auto(conv_39_params.I, conv_39_params.J, conv_39_params.K,
            conv_dw_38_out, conv_39_w, conv_39_b, conv_39_out,
            NO_ACTIVATION, conv_39_params.output_scale, true,
            tiled_matmul_type, check, "conv_39");

        // ====================== res_add (conv_36 + conv_39) ======================
        tiled_resadd_auto(conv_39_params.I, conv_39_params.J,
            conv_39_params.res_scale,
            MVIN_SCALE_IDENTITY,
            ACC_SCALE_IDENTITY,
            conv_36_out,
            conv_39_out,
            conv_39_out,
            false,
            tiled_matmul_type == CPU ? CPU : WS);

        // ====================== conv_40 ======================
        tiled_matmul_nn_auto(conv_40_params.I, conv_40_params.J, conv_40_params.K,
            conv_39_out, conv_40_w, conv_40_b, conv_40_out,
            RELU, conv_40_params.output_scale, true,
            tiled_matmul_type, check, "conv_40");

        // ====================== conv_dw_41 ======================
        if (!conv) {
            conv_dw_with_col2im(conv_40_params.I, conv_40_params.J, conv_dw_41_params.I, conv_dw_41_params.J,
                conv_dw_41_params.batch_size, conv_dw_41_params.in_channels,
                conv_dw_41_params.out_row_dim, conv_dw_41_params.out_col_dim,
                conv_dw_41_params.kernel_size,
                conv_40_out, conv_dw_41_w, conv_dw_41_b, conv_dw_41_out, &conv_dw_41_params);
        } else {
            tiled_conv_dw_auto(
                conv_dw_41_params.batch_size, conv_dw_41_params.in_row_dim, conv_dw_41_params.in_col_dim,
                conv_dw_41_params.in_channels,
                conv_dw_41_params.out_row_dim, conv_dw_41_params.out_col_dim,
                conv_dw_41_params.stride, conv_dw_41_params.padding, conv_dw_41_params.kernel_size,

                (elem_t*)conv_40_out, (elem_t*)conv_dw_41_w, (acc_t*)conv_dw_41_b, (elem_t*)conv_dw_41_out,

                RELU, conv_dw_41_params.output_scale,
                conv_dw_41_params.pool_size, 0, conv_dw_41_params.pool_padding,

                tiled_matmul_type);
        }

        // ====================== conv_42 ======================
        tiled_matmul_nn_auto(conv_42_params.I, conv_42_params.J, conv_42_params.K,
            conv_dw_41_out, conv_42_w, conv_42_b, conv_42_out,
            NO_ACTIVATION, conv_42_params.output_scale, true,
            tiled_matmul_type, check, "conv_42");

        // ====================== conv_43 ======================
        tiled_matmul_nn_auto(conv_43_params.I, conv_43_params.J, conv_43_params.K,
            conv_42_out, conv_43_w, conv_43_b, conv_43_out,
            RELU, conv_43_params.output_scale, true,
            tiled_matmul_type, check, "conv_43");

        // ====================== conv_dw_44 ======================
        if (!conv) {
            conv_dw_with_col2im(conv_43_params.I, conv_43_params.J, conv_dw_44_params.I, conv_dw_44_params.J,
                conv_dw_44_params.batch_size, conv_dw_44_params.in_channels,
                conv_dw_44_params.out_row_dim, conv_dw_44_params.out_col_dim,
                conv_dw_44_params.kernel_size,
                conv_43_out, conv_dw_44_w, conv_dw_44_b, conv_dw_44_out, &conv_dw_44_params);
        } else {
            tiled_conv_dw_auto(
                conv_dw_44_params.batch_size, conv_dw_44_params.in_row_dim, conv_dw_44_params.in_col_dim,
                conv_dw_44_params.in_channels,
                conv_dw_44_params.out_row_dim, conv_dw_44_params.out_col_dim,
                conv_dw_44_params.stride, conv_dw_44_params.padding, conv_dw_44_params.kernel_size,

                (elem_t*)conv_43_out, (elem_t*)conv_dw_44_w, (acc_t*)conv_dw_44_b, (elem_t*)conv_dw_44_out,

                RELU, conv_dw_44_params.output_scale,
                conv_dw_44_params.pool_size, 0, conv_dw_44_params.pool_padding,

                tiled_matmul_type);
        }

        // ====================== conv_45 ======================
        tiled_matmul_nn_auto(conv_45_params.I, conv_45_params.J, conv_45_params.K,
            conv_dw_44_out, conv_45_w, conv_45_b, conv_45_out,
            NO_ACTIVATION, conv_45_params.output_scale, true,
            tiled_matmul_type, check, "conv_45");

        // ====================== res_add (conv_42 + conv_45) ======================
        tiled_resadd_auto(conv_45_params.I, conv_45_params.J,
            conv_45_params.res_scale,
            MVIN_SCALE_IDENTITY,
            ACC_SCALE_IDENTITY,
            conv_42_out,
            conv_45_out,
            conv_45_out,
            false,
            tiled_matmul_type == CPU ? CPU : WS);

        // ====================== conv_46 ======================
        tiled_matmul_nn_auto(conv_46_params.I, conv_46_params.J, conv_46_params.K,
            conv_45_out, conv_46_w, conv_46_b, conv_46_out,
            RELU, conv_46_params.output_scale, true,
            tiled_matmul_type, check, "conv_46");

        // ====================== conv_dw_47 ======================
        if (!conv) {
            conv_dw_with_col2im(conv_46_params.I, conv_46_params.J, conv_dw_47_params.I, conv_dw_47_params.J,
                conv_dw_47_params.batch_size, conv_dw_47_params.in_channels,
                conv_dw_47_params.out_row_dim, conv_dw_47_params.out_col_dim,
                conv_dw_47_params.kernel_size,
                conv_46_out, conv_dw_47_w, conv_dw_47_b, conv_dw_47_out, &conv_dw_47_params);
        } else {
            tiled_conv_dw_auto(
                conv_dw_47_params.batch_size, conv_dw_47_params.in_row_dim, conv_dw_47_params.in_col_dim,
                conv_dw_47_params.in_channels,
                conv_dw_47_params.out_row_dim, conv_dw_47_params.out_col_dim,
                conv_dw_47_params.stride, conv_dw_47_params.padding, conv_dw_47_params.kernel_size,

                (elem_t*)conv_46_out, (elem_t*)conv_dw_47_w, (acc_t*)conv_dw_47_b, (elem_t*)conv_dw_47_out,

                RELU, conv_dw_47_params.output_scale,
                conv_dw_47_params.pool_size, 0, conv_dw_47_params.pool_padding,

                tiled_matmul_type);
        }

        // ====================== conv_48 ======================
        tiled_matmul_nn_auto(conv_48_params.I, conv_48_params.J, conv_48_params.K,
            conv_dw_47_out, conv_48_w, conv_48_b, conv_48_out,
            NO_ACTIVATION, conv_48_params.output_scale, true,
            tiled_matmul_type, check, "conv_48");

        // ====================== res_add (conv_45 + conv_48) ======================
        tiled_resadd_auto(conv_48_params.I, conv_48_params.J,
            conv_48_params.res_scale,
            MVIN_SCALE_IDENTITY,
            ACC_SCALE_IDENTITY,
            conv_45_out,
            conv_48_out,
            conv_48_out,
            false,
            tiled_matmul_type == CPU ? CPU : WS);

        // ====================== conv_49 ======================
        tiled_matmul_nn_auto(conv_49_params.I, conv_49_params.J, conv_49_params.K,
            conv_48_out, conv_49_w, conv_49_b, conv_49_out,
            RELU, conv_49_params.output_scale, true,
            tiled_matmul_type, check, "conv_49");

        // ====================== conv_dw_50 ======================
        if (!conv) {
            conv_dw_with_col2im(conv_49_params.I, conv_49_params.J, conv_dw_50_params.I, conv_dw_50_params.J,
                conv_dw_50_params.batch_size, conv_dw_50_params.in_channels,
                conv_dw_50_params.out_row_dim, conv_dw_50_params.out_col_dim,
                conv_dw_50_params.kernel_size,
                conv_49_out, conv_dw_50_w, conv_dw_50_b, conv_dw_50_out, &conv_dw_50_params);
        } else {
            tiled_conv_dw_auto(
                conv_dw_50_params.batch_size, conv_dw_50_params.in_row_dim, conv_dw_50_params.in_col_dim,
                conv_dw_50_params.in_channels,
                conv_dw_50_params.out_row_dim, conv_dw_50_params.out_col_dim,
                conv_dw_50_params.stride, conv_dw_50_params.padding, conv_dw_50_params.kernel_size,

                (elem_t*)conv_49_out, (elem_t*)conv_dw_50_w, (acc_t*)conv_dw_50_b, (elem_t*)conv_dw_50_out,

                RELU, conv_dw_50_params.output_scale,
                conv_dw_50_params.pool_size, 0, conv_dw_50_params.pool_padding,

                tiled_matmul_type);
        }

        // ====================== conv_51 ======================
        tiled_matmul_nn_auto(conv_51_params.I, conv_51_params.J, conv_51_params.K,
            conv_dw_50_out, conv_51_w, conv_51_b, conv_51_out,
            NO_ACTIVATION, conv_51_params.output_scale, true,
            tiled_matmul_type, check, "conv_51");

        // ====================== conv_52 ======================
        tiled_matmul_nn_auto(conv_52_params.I, conv_52_params.J, conv_52_params.K,
            conv_51_out, conv_52_w, conv_52_b, conv_52_out,
            RELU, conv_52_params.output_scale, true,
            tiled_matmul_type, check, "conv_52");

        // ====================== Global averaging ======================
        static elem_t average[1280][4] row_align(1);

        for (int batch = 0; batch < conv_52_params.batch_size; batch++) {
            for (int channel = 0; channel < conv_52_params.out_channels; channel++) {
                int sum = 0;
                for (int row = 0; row < conv_52_params.out_row_dim; row++) {
                    for (int col = 0; col < conv_52_params.out_col_dim; col++) {
                        size_t r = batch * conv_52_params.out_row_dim * conv_52_params.out_col_dim + row * conv_52_params.out_col_dim + col;
                        sum += conv_52_out[r][channel];
                    }
                }
                const int count = conv_52_params.out_row_dim * conv_52_params.out_col_dim;
                average[channel][batch] = (sum + count/2) / count;
            }
        }

        // ====================== fc_53 ======================
        tiled_matmul_nn_auto(fc_53_params.I, fc_53_params.J, fc_53_params.K,
            fc_53_w, average, fc_53_b, fc_53_out,
            NO_ACTIVATION, fc_53_params.output_scale, false,
            tiled_matmul_type, check, "fc_53");

        uint64_t end = get_time_ns();

        printf("Batch %d/%d  Time: %llu ns\n", batch_idx + 1, num_batches,
               (unsigned long long)(end - start));

        // ====================== Activation diagnostics (first batch only) ======================
        if (batch_idx == 0) {
            // Input pixel stats (batch 0 only)
            int in_min = 127, in_max = -128;
            long in_sum = 0;
            for (int p = 0; p < IMAGE_SIZE; p++) {
                int v = current_images[p];
                if (v < in_min) in_min = v;
                if (v > in_max) in_max = v;
                in_sum += v;
            }
            printf("  [DIAG] Input pixels (img 0): min=%d, max=%d, mean=%.1f\n",
                   in_min, in_max, (float)in_sum / IMAGE_SIZE);

            // conv_1 output stats
            int c1_min = 127, c1_max = -128, c1_nonzero = 0;
            for (int r = 0; r < conv_1_params.I / 4; r++) {
                for (int c = 0; c < conv_1_params.J; c++) {
                    int v = conv_1_out[r][c];
                    if (v < c1_min) c1_min = v;
                    if (v > c1_max) c1_max = v;
                    if (v != 0) c1_nonzero++;
                }
            }
            printf("  [DIAG] conv_1_out (img 0): min=%d, max=%d, nonzero=%d/%d\n",
                   c1_min, c1_max, c1_nonzero, (conv_1_params.I / 4) * conv_1_params.J);

            // conv_52 output stats
            int c52_min = 127, c52_max = -128, c52_nonzero = 0;
            for (int r = 0; r < conv_52_params.I / 4; r++) {
                for (int c = 0; c < conv_52_params.J; c++) {
                    int v = conv_52_out[r][c];
                    if (v < c52_min) c52_min = v;
                    if (v > c52_max) c52_max = v;
                    if (v != 0) c52_nonzero++;
                }
            }
            printf("  [DIAG] conv_52_out (img 0): min=%d, max=%d, nonzero=%d/%d\n",
                   c52_min, c52_max, c52_nonzero, (conv_52_params.I / 4) * conv_52_params.J);

            // average stats
            int avg_min = 127, avg_max = -128, avg_nonzero = 0;
            for (int c = 0; c < 1280; c++) {
                int v = average[c][0];
                if (v < avg_min) avg_min = v;
                if (v > avg_max) avg_max = v;
                if (v != 0) avg_nonzero++;
            }
            printf("  [DIAG] average (img 0): min=%d, max=%d, nonzero=%d/1280\n",
                   avg_min, avg_max, avg_nonzero);

            // fc_53 output stats
            int fc_min = 127, fc_max = -128;
            for (int i = 0; i < 1000; i++) {
                int v = fc_53_out[i][0];
                if (v < fc_min) fc_min = v;
                if (v > fc_max) fc_max = v;
            }
            printf("  [DIAG] fc_53_out (img 0): min=%d, max=%d\n", fc_min, fc_max);
        }

        // ====================== Top-K accuracy ======================
        for (int batch = 0; batch < BATCH_SIZE; batch++) {
            int label = labels[batch_idx * BATCH_SIZE + batch];

            float top_scores[TOP_K];
            int top_indices[TOP_K];
            for (int k = 0; k < TOP_K; k++) {
                top_scores[k] = -1e9f;
                top_indices[k] = -1;
            }

            for (int i = 0; i < fc_53_params.out_features; i++) {
                float score = fc_53_out[i][batch];
                for (int k = 0; k < TOP_K; k++) {
                    if (score > top_scores[k]) {
                        for (int j = TOP_K - 1; j > k; j--) {
                            top_scores[j] = top_scores[j - 1];
                            top_indices[j] = top_indices[j - 1];
                        }
                        top_scores[k] = score;
                        top_indices[k] = i;
                        break;
                    }
                }
            }

            if (top_indices[0] == label) { top1_correct++; window_top1++; }
            for (int k = 0; k < 5; k++) {
                if (top_indices[k] == label) { top5_correct++; window_top5++; break; }
            }
            for (int k = 0; k < 10; k++) {
                if (top_indices[k] == label) { top10_correct++; window_top10++; break; }
            }

            // Debug: print top-10 predictions for first 2 batches (8 images)
            if (batch_idx < 2) {
                int img_idx = batch_idx * BATCH_SIZE + batch;
                printf("\n  Image %d (true label=%d):\n", img_idx, label);
                printf("    Top-10 predictions:\n");
                for (int k = 0; k < TOP_K; k++) {
                    printf("      %2d. class %4d  (score: %.1f)%s\n",
                           k + 1, top_indices[k], top_scores[k],
                           top_indices[k] == label ? "  <-- CORRECT" : "");
                }
            }
        }

        // Progress every 100 images (= 25 batches)
        int imgs_done = (batch_idx + 1) * BATCH_SIZE;
        if (imgs_done % 100 == 0) {
            int window_start = imgs_done - 100 + 1;
            printf("  [Images %d-%d] Window Top-1: %d/100 (%.1f%%), Top-5: %d/100 (%.1f%%), Top-10: %d/100 (%.1f%%)\n",
                   window_start, imgs_done,
                   window_top1, 100.0f * window_top1 / 100,
                   window_top5, 100.0f * window_top5 / 100,
                   window_top10, 100.0f * window_top10 / 100);
            printf("  [Cumulative %d/%d] Top-1: %.2f%%, Top-5: %.2f%%, Top-10: %.2f%%\n",
                   imgs_done, num_batches * BATCH_SIZE,
                   100.0f * top1_correct / imgs_done,
                   100.0f * top5_correct / imgs_done,
                   100.0f * top10_correct / imgs_done);
            // Reset window counters
            window_top1 = 0;
            window_top5 = 0;
            window_top10 = 0;
        }
    }

    fclose(fp_images);

    int total_images = num_batches * BATCH_SIZE;
    printf("\n--- Final Results (%d images) ---\n", total_images);
    printf("Top-1  correct: %d / %d = %.2f%%\n", top1_correct, total_images,
           100.0f * top1_correct / total_images);
    printf("Top-5  correct: %d / %d = %.2f%%\n", top5_correct, total_images,
           100.0f * top5_correct / total_images);
    printf("Top-10 correct: %d / %d = %.2f%%\n", top10_correct, total_images,
           100.0f * top10_correct / total_images);

    exit(0);
}
