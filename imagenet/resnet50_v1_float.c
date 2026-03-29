#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini.h"
#include "include/gemmini_nn.h"

#include "resnet50_params_float.h"
// #include "resnet50_params_1batch.h"
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

static inline uint64_t bench_read_cycles(void) {
    uint64_t c;
    asm volatile ("rdcycle %0" : "=r"(c));
    return c;
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
        printf("usage: %s [-h] matmul_option [check]\n  matmul_option may be 'os', 'ws', or cpu'\n", argv[0]);
        exit(0);
    } else {
        printf("Unknown command-line argument\n");
        printf("usage: %s [-h] matmul_option [check]\n  matmul_option may be 'os', 'ws', or cpu'\n", argv[0]);
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
        printf("usage: %s [-h] matmul_option [check] [conv]\n  matmul_option may be 'os', 'ws', or cpu'\n", argv[0]);
        exit(1);
    }

    bool check = false;

    if (argc < 4) {
        check = false;
    } else if (strcmp(argv[3], "check") == 0) {
        check = true;
    } else {
        printf("Unknown command-line argument\n");
        printf("usage: %s [-h] matmul_option [check] [conv]\n  matmul_option may be 'os', 'ws', or cpu'\n", argv[0]);
        exit(1);
    }

    printf("\n--- ResNet50 V1 Float Streaming Inference ---\n");
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

    uint64_t min_batch_cycles = UINT64_MAX, min_batch_wall = UINT64_MAX;
    uint64_t sum_batch_cycles = 0, sum_batch_wall = 0;
    float best_window_top1 = 0.0f, best_window_top5 = 0.0f;

    setvbuf(stdout, NULL, _IONBF, 0);

    // ===== A/B TEST: Run images.h first to establish baseline =====
    printf("\n===== A/B TEST: Running images.h reference data =====\n");
    {
        elem_t *current_images = (elem_t*)images;  // from images.h
        uint64_t start = get_time_ns();

        // --- Run the full network with images.h data ---
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

            pool_with_col2im(conv_1_params.I, conv_1_params.J,
                conv_1_params.batch_size, conv_1_params.out_channels,
                conv_1_params.out_dim_pooled,
                conv_1_params.out_dim_pooled,
                conv_1_out, conv_1_out_pooled, &conv_1_params);
        } else {
            tiled_conv_auto(
                conv_1_params.batch_size, conv_1_params.in_row_dim, conv_1_params.in_col_dim,
                conv_1_params.in_channels,
                conv_1_params.out_channels, conv_1_params.out_row_dim, conv_1_params.out_col_dim,
                conv_1_params.stride, 1, 1, conv_1_params.padding, conv_1_params.kernel_size,
                false, false, false, false, false,

                (elem_t*)current_images, (elem_t*)conv_1_w, (acc_t*)conv_1_b, (elem_t*)conv_1_out_pooled,

                RELU, conv_1_params.output_scale,
                conv_1_params.pool_size, conv_1_params.pool_stride, conv_1_params.pool_padding,

                tiled_matmul_type);
        }

    // conv_2

        tiled_matmul_nn_auto(conv_2_params.I, conv_2_params.J, conv_2_params.K,
            conv_1_out_pooled, conv_2_w, conv_2_b, conv_2_out,
            RELU, conv_2_params.output_scale, true,
            tiled_matmul_type, check, "conv_2");


    // conv_3

        tiled_conv_auto(
            conv_3_params.batch_size, conv_3_params.in_row_dim, conv_3_params.in_col_dim,
            conv_3_params.in_channels,
            conv_3_params.out_channels, conv_3_params.out_row_dim, conv_3_params.out_col_dim,
            conv_3_params.stride, 1, 1, conv_3_params.padding, conv_3_params.kernel_size,
            false, false, false, false, false,

            (elem_t*)conv_2_out, (elem_t*)conv_3_w, (acc_t*)conv_3_b, (elem_t*)conv_3_out,

            RELU, conv_3_params.output_scale,
            conv_3_params.pool_size, 0, conv_3_params.pool_padding,

            tiled_matmul_type);


    // conv_4

        tiled_matmul_nn_auto(conv_4_params.I, conv_4_params.J, conv_4_params.K,
            conv_3_out, conv_4_w, conv_4_b, conv_4_out,
            NO_ACTIVATION, conv_4_params.output_scale, true,
            tiled_matmul_type, check, "conv_4");


    // Downsampling conv_1_out_pooled
    // conv_5

        tiled_matmul_nn_auto(conv_5_params.I, conv_5_params.J, conv_5_params.K,
            conv_1_out_pooled, conv_5_w, conv_5_b, conv_5_out,
            NO_ACTIVATION, conv_5_params.output_scale, true,
            tiled_matmul_type, check, "conv_5");


    // Add residuals

    tiled_resadd_auto(conv_4_params.I, conv_4_params.J,
        conv_4_params.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        conv_5_out,
        conv_4_out,
        conv_4_out,
        true,
        tiled_matmul_type == CPU ? CPU : WS);

    // conv_6

        tiled_matmul_nn_auto(conv_6_params.I, conv_6_params.J, conv_6_params.K,
            conv_4_out, conv_6_w, conv_6_b, conv_6_out,
            RELU, conv_6_params.output_scale, true,
            tiled_matmul_type, check, "conv_6");


    // conv_7

        tiled_conv_auto(
            conv_7_params.batch_size, conv_7_params.in_row_dim, conv_7_params.in_col_dim,
            conv_7_params.in_channels,
            conv_7_params.out_channels, conv_7_params.out_row_dim, conv_7_params.out_col_dim,
            conv_7_params.stride, 1, 1, conv_7_params.padding, conv_7_params.kernel_size,
            false, false, false, false, false,

            (elem_t*)conv_6_out, (elem_t*)conv_7_w, (acc_t*)conv_7_b, (elem_t*)conv_7_out,

            RELU, conv_7_params.output_scale,
            conv_7_params.pool_size, 0, conv_7_params.pool_padding,

            tiled_matmul_type);


    // conv_8

        tiled_matmul_nn_auto(conv_8_params.I, conv_8_params.J, conv_8_params.K,
            conv_7_out, conv_8_w, conv_8_b, conv_8_out,
            NO_ACTIVATION, conv_8_params.output_scale, true,
            tiled_matmul_type, check, "conv_8");


    // Add residuals

    tiled_resadd_auto(conv_8_params.I, conv_8_params.J,
        conv_8_params.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        conv_4_out,
        conv_8_out,
        conv_8_out,
        true,
        tiled_matmul_type == CPU ? CPU : WS);

    // conv_9

        tiled_matmul_nn_auto(conv_9_params.I, conv_9_params.J, conv_9_params.K,
            conv_8_out, conv_9_w, conv_9_b, conv_9_out,
            RELU, conv_9_params.output_scale, true,
            tiled_matmul_type, check, "conv_9");


    // conv_10

        tiled_conv_auto(
            conv_10_params.batch_size, conv_10_params.in_row_dim, conv_10_params.in_col_dim,
            conv_10_params.in_channels,
            conv_10_params.out_channels, conv_10_params.out_row_dim, conv_10_params.out_col_dim,
            conv_10_params.stride, 1, 1, conv_10_params.padding, conv_10_params.kernel_size,
            false, false, false, false, false,

            (elem_t*)conv_9_out, (elem_t*)conv_10_w, (acc_t*)conv_10_b, (elem_t*)conv_10_out,

            RELU, conv_10_params.output_scale,
            conv_10_params.pool_size, 0, conv_10_params.pool_padding,

            tiled_matmul_type);


    // conv_11

        tiled_matmul_nn_auto(conv_11_params.I, conv_11_params.J, conv_11_params.K,
            conv_10_out, conv_11_w, conv_11_b, conv_11_out,
            NO_ACTIVATION, conv_11_params.output_scale, true,
            tiled_matmul_type, check, "conv_11");


    // Add residuals

    tiled_resadd_auto(conv_11_params.I, conv_11_params.J,
        conv_11_params.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        conv_8_out,
        conv_11_out,
        conv_11_out,
        true,
        tiled_matmul_type == CPU ? CPU : WS);

    // conv_12

        tiled_matmul_nn_auto(conv_12_params.I, conv_12_params.J, conv_12_params.K,
            conv_11_out, conv_12_w, conv_12_b, conv_12_out,
            RELU, conv_12_params.output_scale, true,
            tiled_matmul_type, check, "conv_12");


    // conv_13

        tiled_conv_auto(
            conv_13_params.batch_size, conv_13_params.in_row_dim, conv_13_params.in_col_dim,
            conv_13_params.in_channels,
            conv_13_params.out_channels, conv_13_params.out_row_dim, conv_13_params.out_col_dim,
            conv_13_params.stride, 1, 1, conv_13_params.padding, conv_13_params.kernel_size,
            false, false, false, false, false,

            (elem_t*)conv_12_out, (elem_t*)conv_13_w, (acc_t*)conv_13_b, (elem_t*)conv_13_out,

            RELU, conv_13_params.output_scale,
            conv_13_params.pool_size, 0, conv_13_params.pool_padding,

            tiled_matmul_type);


    // conv_14

        tiled_matmul_nn_auto(conv_14_params.I, conv_14_params.J, conv_14_params.K,
            conv_13_out, conv_14_w, conv_14_b, conv_14_out,
            NO_ACTIVATION, conv_14_params.output_scale, true,
            tiled_matmul_type, check, "conv_14");


    // Downsampling conv_11_out
    // conv_15

        // tiled_conv_auto(
        tiled_conv_downsample(
            conv_15_params.batch_size, conv_15_params.in_row_dim, conv_15_params.in_col_dim,
            conv_15_params.in_channels,
            conv_15_params.out_channels, conv_15_params.out_row_dim, conv_15_params.out_col_dim,
            conv_15_params.in_channels, conv_15_params.out_channels, conv_15_params.out_channels,
            // conv_15_params.stride, 1, 1, conv_15_params.padding, conv_15_params.kernel_size,
            // false, false, false, false, false,

            (elem_t*)conv_11_out, (elem_t*)conv_15_w, (acc_t*)conv_15_b, (elem_t*)conv_15_out,

            NO_ACTIVATION, conv_15_params.output_scale,
            // conv_15_params.pool_size, 0, conv_15_params.pool_padding,

            tiled_matmul_type);


    // Add residuals

    tiled_resadd_auto(conv_14_params.I, conv_14_params.J,
        conv_14_params.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        conv_15_out,
        conv_14_out,
        conv_14_out,
        true,
        tiled_matmul_type == CPU ? CPU : WS);

    
    // conv_16

        tiled_matmul_nn_auto(conv_16_params.I, conv_16_params.J, conv_16_params.K,
            conv_14_out, conv_16_w, conv_16_b, conv_16_out,
            RELU, conv_16_params.output_scale, true,
            tiled_matmul_type, check, "conv_16");


    // conv_17

        tiled_conv_auto(
            conv_17_params.batch_size, conv_17_params.in_row_dim, conv_17_params.in_col_dim,
            conv_17_params.in_channels,
            conv_17_params.out_channels, conv_17_params.out_row_dim, conv_17_params.out_col_dim,
            conv_17_params.stride, 1, 1, conv_17_params.padding, conv_17_params.kernel_size,
            false, false, false, false, false,

            (elem_t*)conv_16_out, (elem_t*)conv_17_w, (acc_t*)conv_17_b, (elem_t*)conv_17_out,

            RELU, conv_17_params.output_scale,
            conv_17_params.pool_size, 0, conv_17_params.pool_padding,

            tiled_matmul_type);


    // conv_18

        tiled_matmul_nn_auto(conv_18_params.I, conv_18_params.J, conv_18_params.K,
            conv_17_out, conv_18_w, conv_18_b, conv_18_out,
            NO_ACTIVATION, conv_18_params.output_scale, true,
            tiled_matmul_type, check, "conv_18");


    // Add residuals

    tiled_resadd_auto(conv_18_params.I, conv_18_params.J,
        conv_18_params.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        conv_14_out,
        conv_18_out,
        conv_18_out,
        true,
        tiled_matmul_type == CPU ? CPU : WS);

    
    // conv_19

        tiled_matmul_nn_auto(conv_19_params.I, conv_19_params.J, conv_19_params.K,
            conv_18_out, conv_19_w, conv_19_b, conv_19_out,
            RELU, conv_19_params.output_scale, true,
            tiled_matmul_type, check, "conv_19");


    // conv_20

        tiled_conv_auto(
            conv_20_params.batch_size, conv_20_params.in_row_dim, conv_20_params.in_col_dim,
            conv_20_params.in_channels,
            conv_20_params.out_channels, conv_20_params.out_row_dim, conv_20_params.out_col_dim,
            conv_20_params.stride, 1, 1, conv_20_params.padding, conv_20_params.kernel_size,
            false, false, false, false, false,

            (elem_t*)conv_19_out, (elem_t*)conv_20_w, (acc_t*)conv_20_b, (elem_t*)conv_20_out,

            RELU, conv_20_params.output_scale,
            conv_20_params.pool_size, 0, conv_20_params.pool_padding,

            tiled_matmul_type);


    // conv_21

        tiled_matmul_nn_auto(conv_21_params.I, conv_21_params.J, conv_21_params.K,
            conv_20_out, conv_21_w, conv_21_b, conv_21_out,
            NO_ACTIVATION, conv_21_params.output_scale, true,
            tiled_matmul_type, check, "conv_21");


    // Add residuals

    tiled_resadd_auto(conv_21_params.I, conv_21_params.J,
        conv_21_params.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        conv_18_out,
        conv_21_out,
        conv_21_out,
        true,
        tiled_matmul_type == CPU ? CPU : WS);

    
    // conv_22

        tiled_matmul_nn_auto(conv_22_params.I, conv_22_params.J, conv_22_params.K,
            conv_21_out, conv_22_w, conv_22_b, conv_22_out,
            RELU, conv_22_params.output_scale, true,
            tiled_matmul_type, check, "conv_22");


    // conv_23

        tiled_conv_auto(
            conv_23_params.batch_size, conv_23_params.in_row_dim, conv_23_params.in_col_dim,
            conv_23_params.in_channels,
            conv_23_params.out_channels, conv_23_params.out_row_dim, conv_23_params.out_col_dim,
            conv_23_params.stride, 1, 1, conv_23_params.padding, conv_23_params.kernel_size,
            false, false, false, false, false,

            (elem_t*)conv_22_out, (elem_t*)conv_23_w, (acc_t*)conv_23_b, (elem_t*)conv_23_out,

            RELU, conv_23_params.output_scale,
            conv_23_params.pool_size, 0, conv_23_params.pool_padding,

            tiled_matmul_type);


    // conv_24

        tiled_matmul_nn_auto(conv_24_params.I, conv_24_params.J, conv_24_params.K,
            conv_23_out, conv_24_w, conv_24_b, conv_24_out,
            NO_ACTIVATION, conv_24_params.output_scale, true,
            tiled_matmul_type, check, "conv_24");


    // Add residuals

    tiled_resadd_auto(conv_24_params.I, conv_24_params.J,
        conv_24_params.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        conv_21_out,
        conv_24_out,
        conv_24_out,
        true,
        tiled_matmul_type == CPU ? CPU : WS);

    
    // conv_25

        tiled_matmul_nn_auto(conv_25_params.I, conv_25_params.J, conv_25_params.K,
            conv_24_out, conv_25_w, conv_25_b, conv_25_out,
            RELU, conv_25_params.output_scale, true,
            tiled_matmul_type, check, "conv_25");


    // conv_26

        tiled_conv_auto(
            conv_26_params.batch_size, conv_26_params.in_row_dim, conv_26_params.in_col_dim,
            conv_26_params.in_channels,
            conv_26_params.out_channels, conv_26_params.out_row_dim, conv_26_params.out_col_dim,
            conv_26_params.stride, 1, 1, conv_26_params.padding, conv_26_params.kernel_size,
            false, false, false, false, false,

            (elem_t*)conv_25_out, (elem_t*)conv_26_w, (acc_t*)conv_26_b, (elem_t*)conv_26_out,

            RELU, conv_26_params.output_scale,
            conv_26_params.pool_size, 0, conv_26_params.pool_padding,

            tiled_matmul_type);


    // conv_27

        tiled_matmul_nn_auto(conv_27_params.I, conv_27_params.J, conv_27_params.K,
            conv_26_out, conv_27_w, conv_27_b, conv_27_out,
            NO_ACTIVATION, conv_27_params.output_scale, true,
            tiled_matmul_type, check, "conv_27");


    // Downsampling conv_24_out
    // conv_28

        // tiled_conv_auto(
        tiled_conv_downsample(
            conv_28_params.batch_size, conv_28_params.in_row_dim, conv_28_params.in_col_dim,
            conv_28_params.in_channels,
            conv_28_params.out_channels, conv_28_params.out_row_dim, conv_28_params.out_col_dim,
            conv_28_params.in_channels, conv_28_params.out_channels, conv_28_params.out_channels,
            // conv_28_params.stride, 1, 1, conv_28_params.padding, conv_28_params.kernel_size,
            // false, false, false, false, false,

            (elem_t*)conv_24_out, (elem_t*)conv_28_w, (acc_t*)conv_28_b, (elem_t*)conv_28_out,

            NO_ACTIVATION, conv_28_params.output_scale,
            // conv_28_params.pool_size, 0, conv_28_params.pool_padding,

            tiled_matmul_type);


    // Add residuals

    tiled_resadd_auto(conv_27_params.I, conv_27_params.J,
        conv_27_params.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        conv_28_out,
        conv_27_out,
        conv_27_out,
        true,
        tiled_matmul_type == CPU ? CPU : WS);

    
    // conv_29

        tiled_matmul_nn_auto(conv_29_params.I, conv_29_params.J, conv_29_params.K,
            conv_27_out, conv_29_w, conv_29_b, conv_29_out,
            RELU, conv_29_params.output_scale, true,
            tiled_matmul_type, check, "conv_29");


    // conv_30

        tiled_conv_auto(
            conv_30_params.batch_size, conv_30_params.in_row_dim, conv_30_params.in_col_dim,
            conv_30_params.in_channels,
            conv_30_params.out_channels, conv_30_params.out_row_dim, conv_30_params.out_col_dim,
            conv_30_params.stride, 1, 1, conv_30_params.padding, conv_30_params.kernel_size,
            false, false, false, false, false,

            (elem_t*)conv_29_out, (elem_t*)conv_30_w, (acc_t*)conv_30_b, (elem_t*)conv_30_out,

            RELU, conv_30_params.output_scale,
            conv_30_params.pool_size, 0, conv_30_params.pool_padding,

            tiled_matmul_type);


    // conv_31

        tiled_matmul_nn_auto(conv_31_params.I, conv_31_params.J, conv_31_params.K,
            conv_30_out, conv_31_w, conv_31_b, conv_31_out,
            NO_ACTIVATION, conv_31_params.output_scale, true,
            tiled_matmul_type, check, "conv_31");


    // Add residuals

    tiled_resadd_auto(conv_31_params.I, conv_31_params.J,
        conv_31_params.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        conv_27_out,
        conv_31_out,
        conv_31_out,
        true,
        tiled_matmul_type == CPU ? CPU : WS);

    
    // conv_32

        tiled_matmul_nn_auto(conv_32_params.I, conv_32_params.J, conv_32_params.K,
            conv_31_out, conv_32_w, conv_32_b, conv_32_out,
            RELU, conv_32_params.output_scale, true,
            tiled_matmul_type, check, "conv_32");


    // conv_33

        tiled_conv_auto(
            conv_33_params.batch_size, conv_33_params.in_row_dim, conv_33_params.in_col_dim,
            conv_33_params.in_channels,
            conv_33_params.out_channels, conv_33_params.out_row_dim, conv_33_params.out_col_dim,
            conv_33_params.stride, 1, 1, conv_33_params.padding, conv_33_params.kernel_size,
            false, false, false, false, false,

            (elem_t*)conv_32_out, (elem_t*)conv_33_w, (acc_t*)conv_33_b, (elem_t*)conv_33_out,

            RELU, conv_33_params.output_scale,
            conv_33_params.pool_size, 0, conv_33_params.pool_padding,

            tiled_matmul_type);


    // conv_34

        tiled_matmul_nn_auto(conv_34_params.I, conv_34_params.J, conv_34_params.K,
            conv_33_out, conv_34_w, conv_34_b, conv_34_out,
            NO_ACTIVATION, conv_34_params.output_scale, true,
            tiled_matmul_type, check, "conv_34");


    // Add residuals

    tiled_resadd_auto(conv_34_params.I, conv_34_params.J,
        conv_34_params.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        conv_31_out,
        conv_34_out,
        conv_34_out,
        true,
        tiled_matmul_type == CPU ? CPU : WS);

    
    // conv_35

        tiled_matmul_nn_auto(conv_35_params.I, conv_35_params.J, conv_35_params.K,
            conv_34_out, conv_35_w, conv_35_b, conv_35_out,
            RELU, conv_35_params.output_scale, true,
            tiled_matmul_type, check, "conv_35");


    // conv_36

        tiled_conv_auto(
            conv_36_params.batch_size, conv_36_params.in_row_dim, conv_36_params.in_col_dim,
            conv_36_params.in_channels,
            conv_36_params.out_channels, conv_36_params.out_row_dim, conv_36_params.out_col_dim,
            conv_36_params.stride, 1, 1, conv_36_params.padding, conv_36_params.kernel_size,
            false, false, false, false, false,

            (elem_t*)conv_35_out, (elem_t*)conv_36_w, (acc_t*)conv_36_b, (elem_t*)conv_36_out,

            RELU, conv_36_params.output_scale,
            conv_36_params.pool_size, 0, conv_36_params.pool_padding,

            tiled_matmul_type);


    // conv_37

        tiled_matmul_nn_auto(conv_37_params.I, conv_37_params.J, conv_37_params.K,
            conv_36_out, conv_37_w, conv_37_b, conv_37_out,
            NO_ACTIVATION, conv_37_params.output_scale, true,
            tiled_matmul_type, check, "conv_37");


    // Add residuals

    tiled_resadd_auto(conv_37_params.I, conv_37_params.J,
        conv_37_params.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        conv_34_out,
        conv_37_out,
        conv_37_out,
        true,
        tiled_matmul_type == CPU ? CPU : WS);

    
    // conv_38

        tiled_matmul_nn_auto(conv_38_params.I, conv_38_params.J, conv_38_params.K,
            conv_37_out, conv_38_w, conv_38_b, conv_38_out,
            RELU, conv_38_params.output_scale, true,
            tiled_matmul_type, check, "conv_38");


    // conv_39

        tiled_conv_auto(
            conv_39_params.batch_size, conv_39_params.in_row_dim, conv_39_params.in_col_dim,
            conv_39_params.in_channels,
            conv_39_params.out_channels, conv_39_params.out_row_dim, conv_39_params.out_col_dim,
            conv_39_params.stride, 1, 1, conv_39_params.padding, conv_39_params.kernel_size,
            false, false, false, false, false,

            (elem_t*)conv_38_out, (elem_t*)conv_39_w, (acc_t*)conv_39_b, (elem_t*)conv_39_out,

            RELU, conv_39_params.output_scale,
            conv_39_params.pool_size, 0, conv_39_params.pool_padding,

            tiled_matmul_type);


    // conv_40

        tiled_matmul_nn_auto(conv_40_params.I, conv_40_params.J, conv_40_params.K,
            conv_39_out, conv_40_w, conv_40_b, conv_40_out,
            NO_ACTIVATION, conv_40_params.output_scale, true,
            tiled_matmul_type, check, "conv_40");


    // Add residuals

    tiled_resadd_auto(conv_40_params.I, conv_40_params.J,
        conv_40_params.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        conv_37_out,
        conv_40_out,
        conv_40_out,
        true,
        tiled_matmul_type == CPU ? CPU : WS);

    
    // conv_41

        tiled_matmul_nn_auto(conv_41_params.I, conv_41_params.J, conv_41_params.K,
            conv_40_out, conv_41_w, conv_41_b, conv_41_out,
            RELU, conv_41_params.output_scale, true,
            tiled_matmul_type, check, "conv_41");


    // conv_42

        tiled_conv_auto(
            conv_42_params.batch_size, conv_42_params.in_row_dim, conv_42_params.in_col_dim,
            conv_42_params.in_channels,
            conv_42_params.out_channels, conv_42_params.out_row_dim, conv_42_params.out_col_dim,
            conv_42_params.stride, 1, 1, conv_42_params.padding, conv_42_params.kernel_size,
            false, false, false, false, false,

            (elem_t*)conv_41_out, (elem_t*)conv_42_w, (acc_t*)conv_42_b, (elem_t*)conv_42_out,

            RELU, conv_42_params.output_scale,
            conv_42_params.pool_size, 0, conv_42_params.pool_padding,

            tiled_matmul_type);


    // conv_43

        tiled_matmul_nn_auto(conv_43_params.I, conv_43_params.J, conv_43_params.K,
            conv_42_out, conv_43_w, conv_43_b, conv_43_out,
            NO_ACTIVATION, conv_43_params.output_scale, true,
            tiled_matmul_type, check, "conv_43");


    // Add residuals

    tiled_resadd_auto(conv_43_params.I, conv_43_params.J,
        conv_43_params.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        conv_40_out,
        conv_43_out,
        conv_43_out,
        true,
        tiled_matmul_type == CPU ? CPU : WS);

    
    // conv_44

        tiled_matmul_nn_auto(conv_44_params.I, conv_44_params.J, conv_44_params.K,
            conv_43_out, conv_44_w, conv_44_b, conv_44_out,
            RELU, conv_44_params.output_scale, true,
            tiled_matmul_type, check, "conv_44");


    // conv_45

        tiled_conv_auto(
            conv_45_params.batch_size, conv_45_params.in_row_dim, conv_45_params.in_col_dim,
            conv_45_params.in_channels,
            conv_45_params.out_channels, conv_45_params.out_row_dim, conv_45_params.out_col_dim,
            conv_45_params.stride, 1, 1, conv_45_params.padding, conv_45_params.kernel_size,
            false, false, false, false, false,

            (elem_t*)conv_44_out, (elem_t*)conv_45_w, (acc_t*)conv_45_b, (elem_t*)conv_45_out,

            RELU, conv_45_params.output_scale,
            conv_45_params.pool_size, 0, conv_45_params.pool_padding,

            tiled_matmul_type);


    // conv_46

        tiled_matmul_nn_auto(conv_46_params.I, conv_46_params.J, conv_46_params.K,
            conv_45_out, conv_46_w, conv_46_b, conv_46_out,
            NO_ACTIVATION, conv_46_params.output_scale, true,
            tiled_matmul_type, check, "conv_46");


    // Downsampling conv_43_out
    // conv_47

        tiled_conv_auto(
            conv_47_params.batch_size, conv_47_params.in_row_dim, conv_47_params.in_col_dim,
            conv_47_params.in_channels,
            conv_47_params.out_channels, conv_47_params.out_row_dim, conv_47_params.out_col_dim,
            conv_47_params.stride, 1, 1, conv_47_params.padding, conv_47_params.kernel_size,
            false, false, false, false, false,

            (elem_t*)conv_43_out, (elem_t*)conv_47_w, (acc_t*)conv_47_b, (elem_t*)conv_47_out,

            NO_ACTIVATION, conv_47_params.output_scale,
            conv_47_params.pool_size, 0, conv_47_params.pool_padding,

            tiled_matmul_type);


    // Add residuals

    tiled_resadd_auto(conv_46_params.I, conv_46_params.J,
        conv_46_params.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        conv_47_out,
        conv_46_out,
        conv_46_out,
        true,
        tiled_matmul_type == CPU ? CPU : WS);

    
    // conv_48

        tiled_matmul_nn_auto(conv_48_params.I, conv_48_params.J, conv_48_params.K,
            conv_46_out, conv_48_w, conv_48_b, conv_48_out,
            RELU, conv_48_params.output_scale, true,
            tiled_matmul_type, check, "conv_48");


    // conv_49

        tiled_conv_auto(
            conv_49_params.batch_size, conv_49_params.in_row_dim, conv_49_params.in_col_dim,
            conv_49_params.in_channels,
            conv_49_params.out_channels, conv_49_params.out_row_dim, conv_49_params.out_col_dim,
            conv_49_params.stride, 1, 1, conv_49_params.padding, conv_49_params.kernel_size,
            false, false, false, false, false,

            (elem_t*)conv_48_out, (elem_t*)conv_49_w, (acc_t*)conv_49_b, (elem_t*)conv_49_out,

            RELU, conv_49_params.output_scale,
            conv_49_params.pool_size, 0, conv_49_params.pool_padding,

            tiled_matmul_type);


    // conv_50

        tiled_matmul_nn_auto(conv_50_params.I, conv_50_params.J, conv_50_params.K,
            conv_49_out, conv_50_w, conv_50_b, conv_50_out,
            NO_ACTIVATION, conv_50_params.output_scale, true,
            tiled_matmul_type, check, "conv_50");


    // Add residuals

    tiled_resadd_auto(conv_50_params.I, conv_50_params.J,
        conv_50_params.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        conv_46_out,
        conv_50_out,
        conv_50_out,
        true,
        tiled_matmul_type == CPU ? CPU : WS);

    
    // conv_51

        tiled_matmul_nn_auto(conv_51_params.I, conv_51_params.J, conv_51_params.K,
            conv_50_out, conv_51_w, conv_51_b, conv_51_out,
            RELU, conv_51_params.output_scale, true,
            tiled_matmul_type, check, "conv_51");


    // conv_52

        tiled_conv_auto(
            conv_52_params.batch_size, conv_52_params.in_row_dim, conv_52_params.in_col_dim,
            conv_52_params.in_channels,
            conv_52_params.out_channels, conv_52_params.out_row_dim, conv_52_params.out_col_dim,
            conv_52_params.stride, 1, 1, conv_52_params.padding, conv_52_params.kernel_size,
            false, false, false, false, false,

            (elem_t*)conv_51_out, (elem_t*)conv_52_w, (acc_t*)conv_52_b, (elem_t*)conv_52_out,

            RELU, conv_52_params.output_scale,
            conv_52_params.pool_size, 0, conv_52_params.pool_padding,

            tiled_matmul_type);


    // conv_53

        tiled_matmul_nn_auto(conv_53_params.I, conv_53_params.J, conv_53_params.K,
            conv_52_out, conv_53_w, conv_53_b, conv_53_out,
            NO_ACTIVATION, conv_53_params.output_scale, true,
            tiled_matmul_type, check, "conv_53");


    // Add residuals

    tiled_resadd_auto(conv_53_params.I, conv_53_params.J,
        conv_53_params.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        conv_50_out,
        conv_53_out,
        conv_53_out,
        true,
        tiled_matmul_type == CPU ? CPU : WS);

    // Global averaging
    static elem_t ref_average[4][2048] row_align(1);

    tiled_global_average_auto(conv_53_out, ref_average, conv_53_params.batch_size,
        conv_53_params.out_channels, conv_53_params.out_row_dim, WS);

    // fc_54

    tiled_matmul_nn_auto(fc_54_params.I, fc_54_params.J, fc_54_params.K,
        ref_average, fc_54_w, fc_54_b, fc_54_out,
        NO_ACTIVATION, fc_54_params.output_scale, false,
        tiled_matmul_type, check, "fc_54");

        // Print fc_54 stats
        float fc_min = 1e30f, fc_max = -1e30f;
        for (int i = 0; i < 1000; i++) {
            float v = fc_54_out[0][i];
            if (v < fc_min) fc_min = v;
            if (v > fc_max) fc_max = v;
        }
        printf("  [REF] fc_54_out: min=%.4f, max=%.4f\n", fc_min, fc_max);

        // Print predictions for images.h
        for (int batch = 0; batch < BATCH_SIZE; batch++) {
            int max_idx = 0;
            elem_t max_val = fc_54_out[batch][0];
            for (int i = 1; i < fc_54_params.out_features; i++) {
                if (fc_54_out[batch][i] > max_val) {
                    max_val = fc_54_out[batch][i];
                    max_idx = i;
                }
            }
            printf("  [REF] Image %d: pred=%d (score=%.4f)\n", batch, max_idx, (float)max_val);
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

        uint64_t cycle_start = bench_read_cycles();
        uint64_t batch_start = get_time_ns();

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

            pool_with_col2im(conv_1_params.I, conv_1_params.J,
                conv_1_params.batch_size, conv_1_params.out_channels,
                conv_1_params.out_dim_pooled,
                conv_1_params.out_dim_pooled,
                conv_1_out, conv_1_out_pooled, &conv_1_params);
        } else {
            tiled_conv_auto(
                conv_1_params.batch_size, conv_1_params.in_row_dim, conv_1_params.in_col_dim,
                conv_1_params.in_channels,
                conv_1_params.out_channels, conv_1_params.out_row_dim, conv_1_params.out_col_dim,
                conv_1_params.stride, 1, 1, conv_1_params.padding, conv_1_params.kernel_size,
                false, false, false, false, false,

                (elem_t*)current_images, (elem_t*)conv_1_w, (acc_t*)conv_1_b, (elem_t*)conv_1_out_pooled,

                RELU, conv_1_params.output_scale,
                conv_1_params.pool_size, conv_1_params.pool_stride, conv_1_params.pool_padding,

                tiled_matmul_type);
        }

    // conv_2
    if (!conv) {

        im2col(conv_2_params.batch_size, conv_2_params.in_channels,
            conv_2_params.in_row_dim, conv_2_params.in_col_dim,
            conv_2_params.I, conv_2_params.K,
            conv_1_out_pooled, conv_2_in, &conv_2_params);

        tiled_matmul_nn_auto(conv_2_params.I, conv_2_params.J, conv_2_params.K,
            conv_2_in, conv_2_w, conv_2_b, conv_2_out,
            RELU, conv_2_params.output_scale, true,
            tiled_matmul_type, check, "conv_2");

    } else {

        tiled_matmul_nn_auto(conv_2_params.I, conv_2_params.J, conv_2_params.K,
            conv_1_out_pooled, conv_2_w, conv_2_b, conv_2_out,
            RELU, conv_2_params.output_scale, true,
            tiled_matmul_type, check, "conv_2");

    }

    // conv_3
    if (!conv) {

        im2col_with_col2im(conv_2_params.I, conv_2_params.J,
            conv_3_params.I, conv_3_params.K,
            conv_2_out, conv_3_in, &conv_3_params);

        tiled_matmul_nn_auto(conv_3_params.I, conv_3_params.J, conv_3_params.K,
            conv_3_in, conv_3_w, conv_3_b, conv_3_out,
            RELU, conv_3_params.output_scale, true,
            tiled_matmul_type, check, "conv_3");

    } else {

        tiled_conv_auto(
            conv_3_params.batch_size, conv_3_params.in_row_dim, conv_3_params.in_col_dim,
            conv_3_params.in_channels,
            conv_3_params.out_channels, conv_3_params.out_row_dim, conv_3_params.out_col_dim,
            conv_3_params.stride, 1, 1, conv_3_params.padding, conv_3_params.kernel_size,
            false, false, false, false, false,

            (elem_t*)conv_2_out, (elem_t*)conv_3_w, (acc_t*)conv_3_b, (elem_t*)conv_3_out,

            RELU, conv_3_params.output_scale,
            conv_3_params.pool_size, 0, conv_3_params.pool_padding,

            tiled_matmul_type);

    }

    // conv_4

        tiled_matmul_nn_auto(conv_4_params.I, conv_4_params.J, conv_4_params.K,
            conv_3_out, conv_4_w, conv_4_b, conv_4_out,
            NO_ACTIVATION, conv_4_params.output_scale, true,
            tiled_matmul_type, check, "conv_4");


    // Downsampling conv_1_out_pooled
    // conv_5
    if (!conv) {

        im2col(conv_5_params.batch_size, conv_5_params.in_channels,
            conv_5_params.in_row_dim, conv_5_params.in_col_dim,
            conv_5_params.I, conv_5_params.K,
            conv_1_out_pooled, conv_5_in, &conv_5_params);

        tiled_matmul_nn_auto(conv_5_params.I, conv_5_params.J, conv_5_params.K,
            conv_5_in, conv_5_w, conv_5_b, conv_5_out,
            NO_ACTIVATION, conv_5_params.output_scale, true,
            tiled_matmul_type, check, "conv_5");

    } else {

        tiled_matmul_nn_auto(conv_5_params.I, conv_5_params.J, conv_5_params.K,
            conv_1_out_pooled, conv_5_w, conv_5_b, conv_5_out,
            NO_ACTIVATION, conv_5_params.output_scale, true,
            tiled_matmul_type, check, "conv_5");

    }

    // Add residuals

    tiled_resadd_auto(conv_4_params.I, conv_4_params.J,
        conv_4_params.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        conv_5_out,
        conv_4_out,
        conv_4_out,
        true,
        tiled_matmul_type == CPU ? CPU : WS);

    // conv_6

        tiled_matmul_nn_auto(conv_6_params.I, conv_6_params.J, conv_6_params.K,
            conv_4_out, conv_6_w, conv_6_b, conv_6_out,
            RELU, conv_6_params.output_scale, true,
            tiled_matmul_type, check, "conv_6");


    // conv_7
    if (!conv) {

        im2col_with_col2im(conv_6_params.I, conv_6_params.J,
            conv_7_params.I, conv_7_params.K,
            conv_6_out, conv_7_in, &conv_7_params);

        tiled_matmul_nn_auto(conv_7_params.I, conv_7_params.J, conv_7_params.K,
            conv_7_in, conv_7_w, conv_7_b, conv_7_out,
            RELU, conv_7_params.output_scale, true,
            tiled_matmul_type, check, "conv_7");

    } else {

        tiled_conv_auto(
            conv_7_params.batch_size, conv_7_params.in_row_dim, conv_7_params.in_col_dim,
            conv_7_params.in_channels,
            conv_7_params.out_channels, conv_7_params.out_row_dim, conv_7_params.out_col_dim,
            conv_7_params.stride, 1, 1, conv_7_params.padding, conv_7_params.kernel_size,
            false, false, false, false, false,

            (elem_t*)conv_6_out, (elem_t*)conv_7_w, (acc_t*)conv_7_b, (elem_t*)conv_7_out,

            RELU, conv_7_params.output_scale,
            conv_7_params.pool_size, 0, conv_7_params.pool_padding,

            tiled_matmul_type);

    }

    // conv_8

        tiled_matmul_nn_auto(conv_8_params.I, conv_8_params.J, conv_8_params.K,
            conv_7_out, conv_8_w, conv_8_b, conv_8_out,
            NO_ACTIVATION, conv_8_params.output_scale, true,
            tiled_matmul_type, check, "conv_8");


    // Add residuals

    tiled_resadd_auto(conv_8_params.I, conv_8_params.J,
        conv_8_params.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        conv_4_out,
        conv_8_out,
        conv_8_out,
        true,
        tiled_matmul_type == CPU ? CPU : WS);

    // conv_9

        tiled_matmul_nn_auto(conv_9_params.I, conv_9_params.J, conv_9_params.K,
            conv_8_out, conv_9_w, conv_9_b, conv_9_out,
            RELU, conv_9_params.output_scale, true,
            tiled_matmul_type, check, "conv_9");


    // conv_10
    if (!conv) {

        im2col_with_col2im(conv_9_params.I, conv_9_params.J,
            conv_10_params.I, conv_10_params.K,
            conv_9_out, conv_10_in, &conv_10_params);

        tiled_matmul_nn_auto(conv_10_params.I, conv_10_params.J, conv_10_params.K,
            conv_10_in, conv_10_w, conv_10_b, conv_10_out,
            RELU, conv_10_params.output_scale, true,
            tiled_matmul_type, check, "conv_10");

    } else {

        tiled_conv_auto(
            conv_10_params.batch_size, conv_10_params.in_row_dim, conv_10_params.in_col_dim,
            conv_10_params.in_channels,
            conv_10_params.out_channels, conv_10_params.out_row_dim, conv_10_params.out_col_dim,
            conv_10_params.stride, 1, 1, conv_10_params.padding, conv_10_params.kernel_size,
            false, false, false, false, false,

            (elem_t*)conv_9_out, (elem_t*)conv_10_w, (acc_t*)conv_10_b, (elem_t*)conv_10_out,

            RELU, conv_10_params.output_scale,
            conv_10_params.pool_size, 0, conv_10_params.pool_padding,

            tiled_matmul_type);

    }

    // conv_11

        tiled_matmul_nn_auto(conv_11_params.I, conv_11_params.J, conv_11_params.K,
            conv_10_out, conv_11_w, conv_11_b, conv_11_out,
            NO_ACTIVATION, conv_11_params.output_scale, true,
            tiled_matmul_type, check, "conv_11");


    // Add residuals

    tiled_resadd_auto(conv_11_params.I, conv_11_params.J,
        conv_11_params.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        conv_8_out,
        conv_11_out,
        conv_11_out,
        true,
        tiled_matmul_type == CPU ? CPU : WS);

    // conv_12

        tiled_matmul_nn_auto(conv_12_params.I, conv_12_params.J, conv_12_params.K,
            conv_11_out, conv_12_w, conv_12_b, conv_12_out,
            RELU, conv_12_params.output_scale, true,
            tiled_matmul_type, check, "conv_12");


    // conv_13
    if (!conv) {

        im2col_with_col2im(conv_12_params.I, conv_12_params.J,
            conv_13_params.I, conv_13_params.K,
            conv_12_out, conv_13_in, &conv_13_params);

        tiled_matmul_nn_auto(conv_13_params.I, conv_13_params.J, conv_13_params.K,
            conv_13_in, conv_13_w, conv_13_b, conv_13_out,
            RELU, conv_13_params.output_scale, true,
            tiled_matmul_type, check, "conv_13");

    } else {

        tiled_conv_auto(
            conv_13_params.batch_size, conv_13_params.in_row_dim, conv_13_params.in_col_dim,
            conv_13_params.in_channels,
            conv_13_params.out_channels, conv_13_params.out_row_dim, conv_13_params.out_col_dim,
            conv_13_params.stride, 1, 1, conv_13_params.padding, conv_13_params.kernel_size,
            false, false, false, false, false,

            (elem_t*)conv_12_out, (elem_t*)conv_13_w, (acc_t*)conv_13_b, (elem_t*)conv_13_out,

            RELU, conv_13_params.output_scale,
            conv_13_params.pool_size, 0, conv_13_params.pool_padding,

            tiled_matmul_type);

    }

    // conv_14

        tiled_matmul_nn_auto(conv_14_params.I, conv_14_params.J, conv_14_params.K,
            conv_13_out, conv_14_w, conv_14_b, conv_14_out,
            NO_ACTIVATION, conv_14_params.output_scale, true,
            tiled_matmul_type, check, "conv_14");


    // Downsampling conv_11_out
    // conv_15
    if (!conv) {

        im2col_with_col2im(conv_11_params.I, conv_11_params.J,
            conv_15_params.I, conv_15_params.K,
            conv_11_out, conv_15_in, &conv_15_params);

        tiled_matmul_nn_auto(conv_15_params.I, conv_15_params.J, conv_15_params.K,
            conv_15_in, conv_15_w, conv_15_b, conv_15_out,
            NO_ACTIVATION, conv_15_params.output_scale, true,
            tiled_matmul_type, check, "conv_15");

    } else {

        // tiled_conv_auto(
        tiled_conv_downsample(
            conv_15_params.batch_size, conv_15_params.in_row_dim, conv_15_params.in_col_dim,
            conv_15_params.in_channels,
            conv_15_params.out_channels, conv_15_params.out_row_dim, conv_15_params.out_col_dim,
            conv_15_params.in_channels, conv_15_params.out_channels, conv_15_params.out_channels,
            // conv_15_params.stride, 1, 1, conv_15_params.padding, conv_15_params.kernel_size,
            // false, false, false, false, false,

            (elem_t*)conv_11_out, (elem_t*)conv_15_w, (acc_t*)conv_15_b, (elem_t*)conv_15_out,

            NO_ACTIVATION, conv_15_params.output_scale,
            // conv_15_params.pool_size, 0, conv_15_params.pool_padding,

            tiled_matmul_type);

    }

    // Add residuals

    tiled_resadd_auto(conv_14_params.I, conv_14_params.J,
        conv_14_params.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        conv_15_out,
        conv_14_out,
        conv_14_out,
        true,
        tiled_matmul_type == CPU ? CPU : WS);

    
    // conv_16

        tiled_matmul_nn_auto(conv_16_params.I, conv_16_params.J, conv_16_params.K,
            conv_14_out, conv_16_w, conv_16_b, conv_16_out,
            RELU, conv_16_params.output_scale, true,
            tiled_matmul_type, check, "conv_16");


    // conv_17
    if (!conv) {

        im2col_with_col2im(conv_16_params.I, conv_16_params.J,
            conv_17_params.I, conv_17_params.K,
            conv_16_out, conv_17_in, &conv_17_params);

        tiled_matmul_nn_auto(conv_17_params.I, conv_17_params.J, conv_17_params.K,
            conv_17_in, conv_17_w, conv_17_b, conv_17_out,
            RELU, conv_17_params.output_scale, true,
            tiled_matmul_type, check, "conv_17");

    } else {

        tiled_conv_auto(
            conv_17_params.batch_size, conv_17_params.in_row_dim, conv_17_params.in_col_dim,
            conv_17_params.in_channels,
            conv_17_params.out_channels, conv_17_params.out_row_dim, conv_17_params.out_col_dim,
            conv_17_params.stride, 1, 1, conv_17_params.padding, conv_17_params.kernel_size,
            false, false, false, false, false,

            (elem_t*)conv_16_out, (elem_t*)conv_17_w, (acc_t*)conv_17_b, (elem_t*)conv_17_out,

            RELU, conv_17_params.output_scale,
            conv_17_params.pool_size, 0, conv_17_params.pool_padding,

            tiled_matmul_type);

    }

    // conv_18

        tiled_matmul_nn_auto(conv_18_params.I, conv_18_params.J, conv_18_params.K,
            conv_17_out, conv_18_w, conv_18_b, conv_18_out,
            NO_ACTIVATION, conv_18_params.output_scale, true,
            tiled_matmul_type, check, "conv_18");


    // Add residuals

    tiled_resadd_auto(conv_18_params.I, conv_18_params.J,
        conv_18_params.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        conv_14_out,
        conv_18_out,
        conv_18_out,
        true,
        tiled_matmul_type == CPU ? CPU : WS);

    
    // conv_19

        tiled_matmul_nn_auto(conv_19_params.I, conv_19_params.J, conv_19_params.K,
            conv_18_out, conv_19_w, conv_19_b, conv_19_out,
            RELU, conv_19_params.output_scale, true,
            tiled_matmul_type, check, "conv_19");


    // conv_20
    if (!conv) {

        im2col_with_col2im(conv_19_params.I, conv_19_params.J,
            conv_20_params.I, conv_20_params.K,
            conv_19_out, conv_20_in, &conv_20_params);

        tiled_matmul_nn_auto(conv_20_params.I, conv_20_params.J, conv_20_params.K,
            conv_20_in, conv_20_w, conv_20_b, conv_20_out,
            RELU, conv_20_params.output_scale, true,
            tiled_matmul_type, check, "conv_20");

    } else {

        tiled_conv_auto(
            conv_20_params.batch_size, conv_20_params.in_row_dim, conv_20_params.in_col_dim,
            conv_20_params.in_channels,
            conv_20_params.out_channels, conv_20_params.out_row_dim, conv_20_params.out_col_dim,
            conv_20_params.stride, 1, 1, conv_20_params.padding, conv_20_params.kernel_size,
            false, false, false, false, false,

            (elem_t*)conv_19_out, (elem_t*)conv_20_w, (acc_t*)conv_20_b, (elem_t*)conv_20_out,

            RELU, conv_20_params.output_scale,
            conv_20_params.pool_size, 0, conv_20_params.pool_padding,

            tiled_matmul_type);

    }

    // conv_21

        tiled_matmul_nn_auto(conv_21_params.I, conv_21_params.J, conv_21_params.K,
            conv_20_out, conv_21_w, conv_21_b, conv_21_out,
            NO_ACTIVATION, conv_21_params.output_scale, true,
            tiled_matmul_type, check, "conv_21");


    // Add residuals

    tiled_resadd_auto(conv_21_params.I, conv_21_params.J,
        conv_21_params.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        conv_18_out,
        conv_21_out,
        conv_21_out,
        true,
        tiled_matmul_type == CPU ? CPU : WS);

    
    // conv_22

        tiled_matmul_nn_auto(conv_22_params.I, conv_22_params.J, conv_22_params.K,
            conv_21_out, conv_22_w, conv_22_b, conv_22_out,
            RELU, conv_22_params.output_scale, true,
            tiled_matmul_type, check, "conv_22");


    // conv_23
    if (!conv) {

        im2col_with_col2im(conv_22_params.I, conv_22_params.J,
            conv_23_params.I, conv_23_params.K,
            conv_22_out, conv_23_in, &conv_23_params);

        tiled_matmul_nn_auto(conv_23_params.I, conv_23_params.J, conv_23_params.K,
            conv_23_in, conv_23_w, conv_23_b, conv_23_out,
            RELU, conv_23_params.output_scale, true,
            tiled_matmul_type, check, "conv_23");

    } else {

        tiled_conv_auto(
            conv_23_params.batch_size, conv_23_params.in_row_dim, conv_23_params.in_col_dim,
            conv_23_params.in_channels,
            conv_23_params.out_channels, conv_23_params.out_row_dim, conv_23_params.out_col_dim,
            conv_23_params.stride, 1, 1, conv_23_params.padding, conv_23_params.kernel_size,
            false, false, false, false, false,

            (elem_t*)conv_22_out, (elem_t*)conv_23_w, (acc_t*)conv_23_b, (elem_t*)conv_23_out,

            RELU, conv_23_params.output_scale,
            conv_23_params.pool_size, 0, conv_23_params.pool_padding,

            tiled_matmul_type);

    }

    // conv_24

        tiled_matmul_nn_auto(conv_24_params.I, conv_24_params.J, conv_24_params.K,
            conv_23_out, conv_24_w, conv_24_b, conv_24_out,
            NO_ACTIVATION, conv_24_params.output_scale, true,
            tiled_matmul_type, check, "conv_24");


    // Add residuals

    tiled_resadd_auto(conv_24_params.I, conv_24_params.J,
        conv_24_params.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        conv_21_out,
        conv_24_out,
        conv_24_out,
        true,
        tiled_matmul_type == CPU ? CPU : WS);

    
    // conv_25

        tiled_matmul_nn_auto(conv_25_params.I, conv_25_params.J, conv_25_params.K,
            conv_24_out, conv_25_w, conv_25_b, conv_25_out,
            RELU, conv_25_params.output_scale, true,
            tiled_matmul_type, check, "conv_25");


    // conv_26
    if (!conv) {

        im2col_with_col2im(conv_25_params.I, conv_25_params.J,
            conv_26_params.I, conv_26_params.K,
            conv_25_out, conv_26_in, &conv_26_params);

        tiled_matmul_nn_auto(conv_26_params.I, conv_26_params.J, conv_26_params.K,
            conv_26_in, conv_26_w, conv_26_b, conv_26_out,
            RELU, conv_26_params.output_scale, true,
            tiled_matmul_type, check, "conv_26");

    } else {

        tiled_conv_auto(
            conv_26_params.batch_size, conv_26_params.in_row_dim, conv_26_params.in_col_dim,
            conv_26_params.in_channels,
            conv_26_params.out_channels, conv_26_params.out_row_dim, conv_26_params.out_col_dim,
            conv_26_params.stride, 1, 1, conv_26_params.padding, conv_26_params.kernel_size,
            false, false, false, false, false,

            (elem_t*)conv_25_out, (elem_t*)conv_26_w, (acc_t*)conv_26_b, (elem_t*)conv_26_out,

            RELU, conv_26_params.output_scale,
            conv_26_params.pool_size, 0, conv_26_params.pool_padding,

            tiled_matmul_type);

    }

    // conv_27

        tiled_matmul_nn_auto(conv_27_params.I, conv_27_params.J, conv_27_params.K,
            conv_26_out, conv_27_w, conv_27_b, conv_27_out,
            NO_ACTIVATION, conv_27_params.output_scale, true,
            tiled_matmul_type, check, "conv_27");


    // Downsampling conv_24_out
    // conv_28
    if (!conv) {

        im2col_with_col2im(conv_24_params.I, conv_24_params.J,
            conv_28_params.I, conv_28_params.K,
            conv_24_out, conv_28_in, &conv_28_params);

        tiled_matmul_nn_auto(conv_28_params.I, conv_28_params.J, conv_28_params.K,
            conv_28_in, conv_28_w, conv_28_b, conv_28_out,
            NO_ACTIVATION, conv_28_params.output_scale, true,
            tiled_matmul_type, check, "conv_28");

    } else {

        // tiled_conv_auto(
        tiled_conv_downsample(
            conv_28_params.batch_size, conv_28_params.in_row_dim, conv_28_params.in_col_dim,
            conv_28_params.in_channels,
            conv_28_params.out_channels, conv_28_params.out_row_dim, conv_28_params.out_col_dim,
            conv_28_params.in_channels, conv_28_params.out_channels, conv_28_params.out_channels,
            // conv_28_params.stride, 1, 1, conv_28_params.padding, conv_28_params.kernel_size,
            // false, false, false, false, false,

            (elem_t*)conv_24_out, (elem_t*)conv_28_w, (acc_t*)conv_28_b, (elem_t*)conv_28_out,

            NO_ACTIVATION, conv_28_params.output_scale,
            // conv_28_params.pool_size, 0, conv_28_params.pool_padding,

            tiled_matmul_type);

    }

    // Add residuals

    tiled_resadd_auto(conv_27_params.I, conv_27_params.J,
        conv_27_params.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        conv_28_out,
        conv_27_out,
        conv_27_out,
        true,
        tiled_matmul_type == CPU ? CPU : WS);

    
    // conv_29

        tiled_matmul_nn_auto(conv_29_params.I, conv_29_params.J, conv_29_params.K,
            conv_27_out, conv_29_w, conv_29_b, conv_29_out,
            RELU, conv_29_params.output_scale, true,
            tiled_matmul_type, check, "conv_29");


    // conv_30
    if (!conv) {

        im2col_with_col2im(conv_29_params.I, conv_29_params.J,
            conv_30_params.I, conv_30_params.K,
            conv_29_out, conv_30_in, &conv_30_params);

        tiled_matmul_nn_auto(conv_30_params.I, conv_30_params.J, conv_30_params.K,
            conv_30_in, conv_30_w, conv_30_b, conv_30_out,
            RELU, conv_30_params.output_scale, true,
            tiled_matmul_type, check, "conv_30");

    } else {

        tiled_conv_auto(
            conv_30_params.batch_size, conv_30_params.in_row_dim, conv_30_params.in_col_dim,
            conv_30_params.in_channels,
            conv_30_params.out_channels, conv_30_params.out_row_dim, conv_30_params.out_col_dim,
            conv_30_params.stride, 1, 1, conv_30_params.padding, conv_30_params.kernel_size,
            false, false, false, false, false,

            (elem_t*)conv_29_out, (elem_t*)conv_30_w, (acc_t*)conv_30_b, (elem_t*)conv_30_out,

            RELU, conv_30_params.output_scale,
            conv_30_params.pool_size, 0, conv_30_params.pool_padding,

            tiled_matmul_type);

    }

    // conv_31

        tiled_matmul_nn_auto(conv_31_params.I, conv_31_params.J, conv_31_params.K,
            conv_30_out, conv_31_w, conv_31_b, conv_31_out,
            NO_ACTIVATION, conv_31_params.output_scale, true,
            tiled_matmul_type, check, "conv_31");


    // Add residuals

    tiled_resadd_auto(conv_31_params.I, conv_31_params.J,
        conv_31_params.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        conv_27_out,
        conv_31_out,
        conv_31_out,
        true,
        tiled_matmul_type == CPU ? CPU : WS);

    
    // conv_32

        tiled_matmul_nn_auto(conv_32_params.I, conv_32_params.J, conv_32_params.K,
            conv_31_out, conv_32_w, conv_32_b, conv_32_out,
            RELU, conv_32_params.output_scale, true,
            tiled_matmul_type, check, "conv_32");


    // conv_33
    if (!conv) {

        im2col_with_col2im(conv_32_params.I, conv_32_params.J,
            conv_33_params.I, conv_33_params.K,
            conv_32_out, conv_33_in, &conv_33_params);

        tiled_matmul_nn_auto(conv_33_params.I, conv_33_params.J, conv_33_params.K,
            conv_33_in, conv_33_w, conv_33_b, conv_33_out,
            RELU, conv_33_params.output_scale, true,
            tiled_matmul_type, check, "conv_33");

    } else {

        tiled_conv_auto(
            conv_33_params.batch_size, conv_33_params.in_row_dim, conv_33_params.in_col_dim,
            conv_33_params.in_channels,
            conv_33_params.out_channels, conv_33_params.out_row_dim, conv_33_params.out_col_dim,
            conv_33_params.stride, 1, 1, conv_33_params.padding, conv_33_params.kernel_size,
            false, false, false, false, false,

            (elem_t*)conv_32_out, (elem_t*)conv_33_w, (acc_t*)conv_33_b, (elem_t*)conv_33_out,

            RELU, conv_33_params.output_scale,
            conv_33_params.pool_size, 0, conv_33_params.pool_padding,

            tiled_matmul_type);

    }

    // conv_34

        tiled_matmul_nn_auto(conv_34_params.I, conv_34_params.J, conv_34_params.K,
            conv_33_out, conv_34_w, conv_34_b, conv_34_out,
            NO_ACTIVATION, conv_34_params.output_scale, true,
            tiled_matmul_type, check, "conv_34");


    // Add residuals

    tiled_resadd_auto(conv_34_params.I, conv_34_params.J,
        conv_34_params.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        conv_31_out,
        conv_34_out,
        conv_34_out,
        true,
        tiled_matmul_type == CPU ? CPU : WS);

    
    // conv_35

        tiled_matmul_nn_auto(conv_35_params.I, conv_35_params.J, conv_35_params.K,
            conv_34_out, conv_35_w, conv_35_b, conv_35_out,
            RELU, conv_35_params.output_scale, true,
            tiled_matmul_type, check, "conv_35");


    // conv_36
    if (!conv) {

        im2col_with_col2im(conv_35_params.I, conv_35_params.J,
            conv_36_params.I, conv_36_params.K,
            conv_35_out, conv_36_in, &conv_36_params);

        tiled_matmul_nn_auto(conv_36_params.I, conv_36_params.J, conv_36_params.K,
            conv_36_in, conv_36_w, conv_36_b, conv_36_out,
            RELU, conv_36_params.output_scale, true,
            tiled_matmul_type, check, "conv_36");

    } else {

        tiled_conv_auto(
            conv_36_params.batch_size, conv_36_params.in_row_dim, conv_36_params.in_col_dim,
            conv_36_params.in_channels,
            conv_36_params.out_channels, conv_36_params.out_row_dim, conv_36_params.out_col_dim,
            conv_36_params.stride, 1, 1, conv_36_params.padding, conv_36_params.kernel_size,
            false, false, false, false, false,

            (elem_t*)conv_35_out, (elem_t*)conv_36_w, (acc_t*)conv_36_b, (elem_t*)conv_36_out,

            RELU, conv_36_params.output_scale,
            conv_36_params.pool_size, 0, conv_36_params.pool_padding,

            tiled_matmul_type);

    }

    // conv_37

        tiled_matmul_nn_auto(conv_37_params.I, conv_37_params.J, conv_37_params.K,
            conv_36_out, conv_37_w, conv_37_b, conv_37_out,
            NO_ACTIVATION, conv_37_params.output_scale, true,
            tiled_matmul_type, check, "conv_37");


    // Add residuals

    tiled_resadd_auto(conv_37_params.I, conv_37_params.J,
        conv_37_params.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        conv_34_out,
        conv_37_out,
        conv_37_out,
        true,
        tiled_matmul_type == CPU ? CPU : WS);

    
    // conv_38

        tiled_matmul_nn_auto(conv_38_params.I, conv_38_params.J, conv_38_params.K,
            conv_37_out, conv_38_w, conv_38_b, conv_38_out,
            RELU, conv_38_params.output_scale, true,
            tiled_matmul_type, check, "conv_38");


    // conv_39
    if (!conv) {

        im2col_with_col2im(conv_38_params.I, conv_38_params.J,
            conv_39_params.I, conv_39_params.K,
            conv_38_out, conv_39_in, &conv_39_params);

        tiled_matmul_nn_auto(conv_39_params.I, conv_39_params.J, conv_39_params.K,
            conv_39_in, conv_39_w, conv_39_b, conv_39_out,
            RELU, conv_39_params.output_scale, true,
            tiled_matmul_type, check, "conv_39");

    } else {

        tiled_conv_auto(
            conv_39_params.batch_size, conv_39_params.in_row_dim, conv_39_params.in_col_dim,
            conv_39_params.in_channels,
            conv_39_params.out_channels, conv_39_params.out_row_dim, conv_39_params.out_col_dim,
            conv_39_params.stride, 1, 1, conv_39_params.padding, conv_39_params.kernel_size,
            false, false, false, false, false,

            (elem_t*)conv_38_out, (elem_t*)conv_39_w, (acc_t*)conv_39_b, (elem_t*)conv_39_out,

            RELU, conv_39_params.output_scale,
            conv_39_params.pool_size, 0, conv_39_params.pool_padding,

            tiled_matmul_type);

    }

    // conv_40

        tiled_matmul_nn_auto(conv_40_params.I, conv_40_params.J, conv_40_params.K,
            conv_39_out, conv_40_w, conv_40_b, conv_40_out,
            NO_ACTIVATION, conv_40_params.output_scale, true,
            tiled_matmul_type, check, "conv_40");


    // Add residuals

    tiled_resadd_auto(conv_40_params.I, conv_40_params.J,
        conv_40_params.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        conv_37_out,
        conv_40_out,
        conv_40_out,
        true,
        tiled_matmul_type == CPU ? CPU : WS);

    
    // conv_41

        tiled_matmul_nn_auto(conv_41_params.I, conv_41_params.J, conv_41_params.K,
            conv_40_out, conv_41_w, conv_41_b, conv_41_out,
            RELU, conv_41_params.output_scale, true,
            tiled_matmul_type, check, "conv_41");


    // conv_42
    if (!conv) {

        im2col_with_col2im(conv_41_params.I, conv_41_params.J,
            conv_42_params.I, conv_42_params.K,
            conv_41_out, conv_42_in, &conv_42_params);

        tiled_matmul_nn_auto(conv_42_params.I, conv_42_params.J, conv_42_params.K,
            conv_42_in, conv_42_w, conv_42_b, conv_42_out,
            RELU, conv_42_params.output_scale, true,
            tiled_matmul_type, check, "conv_42");

    } else {

        tiled_conv_auto(
            conv_42_params.batch_size, conv_42_params.in_row_dim, conv_42_params.in_col_dim,
            conv_42_params.in_channels,
            conv_42_params.out_channels, conv_42_params.out_row_dim, conv_42_params.out_col_dim,
            conv_42_params.stride, 1, 1, conv_42_params.padding, conv_42_params.kernel_size,
            false, false, false, false, false,

            (elem_t*)conv_41_out, (elem_t*)conv_42_w, (acc_t*)conv_42_b, (elem_t*)conv_42_out,

            RELU, conv_42_params.output_scale,
            conv_42_params.pool_size, 0, conv_42_params.pool_padding,

            tiled_matmul_type);

    }

    // conv_43

        tiled_matmul_nn_auto(conv_43_params.I, conv_43_params.J, conv_43_params.K,
            conv_42_out, conv_43_w, conv_43_b, conv_43_out,
            NO_ACTIVATION, conv_43_params.output_scale, true,
            tiled_matmul_type, check, "conv_43");


    // Add residuals

    tiled_resadd_auto(conv_43_params.I, conv_43_params.J,
        conv_43_params.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        conv_40_out,
        conv_43_out,
        conv_43_out,
        true,
        tiled_matmul_type == CPU ? CPU : WS);

    
    // conv_44

        tiled_matmul_nn_auto(conv_44_params.I, conv_44_params.J, conv_44_params.K,
            conv_43_out, conv_44_w, conv_44_b, conv_44_out,
            RELU, conv_44_params.output_scale, true,
            tiled_matmul_type, check, "conv_44");


    // conv_45
    if (!conv) {

        im2col_with_col2im(conv_44_params.I, conv_44_params.J,
            conv_45_params.I, conv_45_params.K,
            conv_44_out, conv_45_in, &conv_45_params);

        tiled_matmul_nn_auto(conv_45_params.I, conv_45_params.J, conv_45_params.K,
            conv_45_in, conv_45_w, conv_45_b, conv_45_out,
            RELU, conv_45_params.output_scale, true,
            tiled_matmul_type, check, "conv_45");

    } else {

        tiled_conv_auto(
            conv_45_params.batch_size, conv_45_params.in_row_dim, conv_45_params.in_col_dim,
            conv_45_params.in_channels,
            conv_45_params.out_channels, conv_45_params.out_row_dim, conv_45_params.out_col_dim,
            conv_45_params.stride, 1, 1, conv_45_params.padding, conv_45_params.kernel_size,
            false, false, false, false, false,

            (elem_t*)conv_44_out, (elem_t*)conv_45_w, (acc_t*)conv_45_b, (elem_t*)conv_45_out,

            RELU, conv_45_params.output_scale,
            conv_45_params.pool_size, 0, conv_45_params.pool_padding,

            tiled_matmul_type);

    }

    // conv_46

        tiled_matmul_nn_auto(conv_46_params.I, conv_46_params.J, conv_46_params.K,
            conv_45_out, conv_46_w, conv_46_b, conv_46_out,
            NO_ACTIVATION, conv_46_params.output_scale, true,
            tiled_matmul_type, check, "conv_46");


    // Downsampling conv_43_out
    // conv_47
    if (!conv) {

        im2col_with_col2im(conv_43_params.I, conv_43_params.J,
            conv_47_params.I, conv_47_params.K,
            conv_43_out, conv_47_in, &conv_47_params);

        tiled_matmul_nn_auto(conv_47_params.I, conv_47_params.J, conv_47_params.K,
            conv_47_in, conv_47_w, conv_47_b, conv_47_out,
            NO_ACTIVATION, conv_47_params.output_scale, true,
            tiled_matmul_type, check, "conv_47");

    } else {

        tiled_conv_auto(
            conv_47_params.batch_size, conv_47_params.in_row_dim, conv_47_params.in_col_dim,
            conv_47_params.in_channels,
            conv_47_params.out_channels, conv_47_params.out_row_dim, conv_47_params.out_col_dim,
            conv_47_params.stride, 1, 1, conv_47_params.padding, conv_47_params.kernel_size,
            false, false, false, false, false,

            (elem_t*)conv_43_out, (elem_t*)conv_47_w, (acc_t*)conv_47_b, (elem_t*)conv_47_out,

            NO_ACTIVATION, conv_47_params.output_scale,
            conv_47_params.pool_size, 0, conv_47_params.pool_padding,

            tiled_matmul_type);

    }

    // Add residuals

    tiled_resadd_auto(conv_46_params.I, conv_46_params.J,
        conv_46_params.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        conv_47_out,
        conv_46_out,
        conv_46_out,
        true,
        tiled_matmul_type == CPU ? CPU : WS);

    
    // conv_48

        tiled_matmul_nn_auto(conv_48_params.I, conv_48_params.J, conv_48_params.K,
            conv_46_out, conv_48_w, conv_48_b, conv_48_out,
            RELU, conv_48_params.output_scale, true,
            tiled_matmul_type, check, "conv_48");


    // conv_49
    if (!conv) {

        im2col_with_col2im(conv_48_params.I, conv_48_params.J,
            conv_49_params.I, conv_49_params.K,
            conv_48_out, conv_49_in, &conv_49_params);

        tiled_matmul_nn_auto(conv_49_params.I, conv_49_params.J, conv_49_params.K,
            conv_49_in, conv_49_w, conv_49_b, conv_49_out,
            RELU, conv_49_params.output_scale, true,
            tiled_matmul_type, check, "conv_49");

    } else {

        tiled_conv_auto(
            conv_49_params.batch_size, conv_49_params.in_row_dim, conv_49_params.in_col_dim,
            conv_49_params.in_channels,
            conv_49_params.out_channels, conv_49_params.out_row_dim, conv_49_params.out_col_dim,
            conv_49_params.stride, 1, 1, conv_49_params.padding, conv_49_params.kernel_size,
            false, false, false, false, false,

            (elem_t*)conv_48_out, (elem_t*)conv_49_w, (acc_t*)conv_49_b, (elem_t*)conv_49_out,

            RELU, conv_49_params.output_scale,
            conv_49_params.pool_size, 0, conv_49_params.pool_padding,

            tiled_matmul_type);

    }

    // conv_50

        tiled_matmul_nn_auto(conv_50_params.I, conv_50_params.J, conv_50_params.K,
            conv_49_out, conv_50_w, conv_50_b, conv_50_out,
            NO_ACTIVATION, conv_50_params.output_scale, true,
            tiled_matmul_type, check, "conv_50");


    // Add residuals

    tiled_resadd_auto(conv_50_params.I, conv_50_params.J,
        conv_50_params.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        conv_46_out,
        conv_50_out,
        conv_50_out,
        true,
        tiled_matmul_type == CPU ? CPU : WS);

    
    // conv_51

        tiled_matmul_nn_auto(conv_51_params.I, conv_51_params.J, conv_51_params.K,
            conv_50_out, conv_51_w, conv_51_b, conv_51_out,
            RELU, conv_51_params.output_scale, true,
            tiled_matmul_type, check, "conv_51");


    // conv_52
    if (!conv) {

        im2col_with_col2im(conv_51_params.I, conv_51_params.J,
            conv_52_params.I, conv_52_params.K,
            conv_51_out, conv_52_in, &conv_52_params);

        tiled_matmul_nn_auto(conv_52_params.I, conv_52_params.J, conv_52_params.K,
            conv_52_in, conv_52_w, conv_52_b, conv_52_out,
            RELU, conv_52_params.output_scale, true,
            tiled_matmul_type, check, "conv_52");

    } else {

        tiled_conv_auto(
            conv_52_params.batch_size, conv_52_params.in_row_dim, conv_52_params.in_col_dim,
            conv_52_params.in_channels,
            conv_52_params.out_channels, conv_52_params.out_row_dim, conv_52_params.out_col_dim,
            conv_52_params.stride, 1, 1, conv_52_params.padding, conv_52_params.kernel_size,
            false, false, false, false, false,

            (elem_t*)conv_51_out, (elem_t*)conv_52_w, (acc_t*)conv_52_b, (elem_t*)conv_52_out,

            RELU, conv_52_params.output_scale,
            conv_52_params.pool_size, 0, conv_52_params.pool_padding,

            tiled_matmul_type);

    }

    // conv_53

        tiled_matmul_nn_auto(conv_53_params.I, conv_53_params.J, conv_53_params.K,
            conv_52_out, conv_53_w, conv_53_b, conv_53_out,
            NO_ACTIVATION, conv_53_params.output_scale, true,
            tiled_matmul_type, check, "conv_53");


    // Add residuals

    tiled_resadd_auto(conv_53_params.I, conv_53_params.J,
        conv_53_params.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        conv_50_out,
        conv_53_out,
        conv_53_out,
        true,
        tiled_matmul_type == CPU ? CPU : WS);

    // Global averaging
    static elem_t average[4][2048] row_align(1);

    tiled_global_average_auto(conv_53_out, average, conv_53_params.batch_size,
        conv_53_params.out_channels, conv_53_params.out_row_dim, WS);

    // fc_54

    tiled_matmul_nn_auto(fc_54_params.I, fc_54_params.J, fc_54_params.K,
        average, fc_54_w, fc_54_b, fc_54_out,
        NO_ACTIVATION, fc_54_params.output_scale, false,
        tiled_matmul_type, check, "fc_54");

        uint64_t cycle_end = bench_read_cycles();
        uint64_t batch_end = get_time_ns();

        uint64_t batch_cycles = cycle_end - cycle_start;
        uint64_t batch_wall = batch_end - batch_start;
        if (batch_cycles < min_batch_cycles) min_batch_cycles = batch_cycles;
        if (batch_wall < min_batch_wall) min_batch_wall = batch_wall;
        sum_batch_cycles += batch_cycles;
        sum_batch_wall += batch_wall;

        printf("Batch %d/%d  Cycles: %llu  Time: %llu ns\n", batch_idx + 1, num_batches,
               (unsigned long long)batch_cycles, (unsigned long long)batch_wall);

        // ====================== Top-K accuracy ======================
        for (int batch = 0; batch < BATCH_SIZE; batch++) {
            int label = labels[batch_idx * BATCH_SIZE + batch];

            float top_scores[TOP_K];
            int top_indices[TOP_K];
            for (int k = 0; k < TOP_K; k++) {
                top_scores[k] = -1e9f;
                top_indices[k] = -1;
            }

            for (int i = 0; i < fc_54_params.out_features; i++) {
                float score = fc_54_out[batch][i];
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
            float w_top1 = 100.0f * window_top1 / 100;
            float w_top5 = 100.0f * window_top5 / 100;
            if (w_top1 > best_window_top1) best_window_top1 = w_top1;
            if (w_top5 > best_window_top5) best_window_top5 = w_top5;
            // Reset window counters
            window_top1 = 0;
            window_top5 = 0;
            window_top10 = 0;
        }
    }

    fclose(fp_images);

    int total_images = num_batches * BATCH_SIZE;
    uint64_t avg_batch_cycles = (num_batches > 0) ? sum_batch_cycles / num_batches : 0;
    uint64_t avg_batch_wall = (num_batches > 0) ? sum_batch_wall / num_batches : 0;
    float final_top1 = (total_images > 0) ? 100.0f * top1_correct / total_images : 0;
    float final_top5 = (total_images > 0) ? 100.0f * top5_correct / total_images : 0;
    float final_top10 = (total_images > 0) ? 100.0f * top10_correct / total_images : 0;
    double imgs_per_sec = (avg_batch_wall > 0) ? (double)BATCH_SIZE * 1e9 / avg_batch_wall : 0;

    printf("\n--- Final Results (%d images) ---\n", total_images);
    printf("Top-1  correct: %d / %d = %.2f%%\n", top1_correct, total_images, final_top1);
    printf("Top-5  correct: %d / %d = %.2f%%\n", top5_correct, total_images, final_top5);
    printf("Top-10 correct: %d / %d = %.2f%%\n", top10_correct, total_images, final_top10);
    printf("Best window top-1: %.1f%%  top-5: %.1f%%\n", best_window_top1, best_window_top5);
    printf("Min batch cycles: %llu  Avg batch cycles: %llu\n",
           (unsigned long long)min_batch_cycles, (unsigned long long)avg_batch_cycles);
    printf("Min batch wall: %llu ns  Avg batch wall: %llu ns\n",
           (unsigned long long)min_batch_wall, (unsigned long long)avg_batch_wall);
    printf("Throughput: %.2f images/sec\n", imgs_per_sec);

    printf("\nCSV,ResNet50-ImageNet-Float,imagenet,224x224,%d,%.2f,%.2f,%.2f,%.1f,%.1f,%llu,%llu,%llu,%llu,%.2f\n",
           total_images, final_top1, final_top5, final_top10,
           best_window_top1, best_window_top5,
           (unsigned long long)min_batch_cycles, (unsigned long long)avg_batch_cycles,
           (unsigned long long)min_batch_wall, (unsigned long long)avg_batch_wall,
           imgs_per_sec);

    exit(0);
}
