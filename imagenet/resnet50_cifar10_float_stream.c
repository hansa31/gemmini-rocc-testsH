// resnet50_cifar10_float_stream.c — Streaming Float (FP32) inference for ResNet-50 on CIFAR-10
//
// Architecture (edadaltocg/resnet50_cifar10):
//   Input: 32x32x3 RGB,  stem: 3x3/s1/p1 conv (no maxpool),  10 output classes
//   4 ResNet stages with bottleneck blocks:
//     Stage0: 3 blocks, 64/64/256 channels,  32x32 spatial
//     Stage1: 4 blocks, 128/128/512 channels, 32x32->16x16 spatial
//     Stage2: 6 blocks, 256/256/1024 channels, 16x16->8x8 spatial
//     Stage3: 3 blocks, 512/512/2048 channels, 8x8->4x4 spatial
//   Global average pool -> FC (2048->10)
//
// Weights extracted via extract_resnet50_cifar10_float.py:
//   FP32 weights with BatchNorm folded in, output_scale=1.0f for all layers.
//
// Streams 10000 CIFAR-10 test images in batches of 4 from a binary file.
// Reports top-1 and top-5 accuracy every 100 images and at the end.
//
// Usage: ./resnet50_cifar10_float_stream [ws|os|cpu]

#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include <time.h>

#include "include/gemmini.h"
#include "include/gemmini_nn.h"
#include "include/gemmini_float_convert.h"

#include "resnet50_cifar10_params_float.h"
#include "cifar10_images.h"

// ---- Configuration ----
#define TOP_K         5
#define NUM_IMAGES    10000
#define BATCH_SIZE    4
#define IMAGE_SIZE    (32 * 32 * 3)

#define IMAGES_BIN_FILE   "cifar10_test_10000.bin"
#define LABELS_TXT_FILE   "cifar10_test_10000_labels.txt"

#ifndef CLOCK_MONOTONIC
    #define CLOCK_MONOTONIC CLOCK_REALTIME
#endif

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

// CIFAR-10 class names
static const char *class_names[10] = {
    "airplane", "automobile", "bird", "cat",  "deer",
    "dog",      "frog",       "horse", "ship", "truck"
};

// ---------------------------------------------------------------------------
// Global average pool (manual loops, gemmini layout)
//   in_data[n_patches][channels] (n_patches = batch * out_row * out_col)
//   average[channels][batch_size]
// ---------------------------------------------------------------------------
static elem_t average[2048][BATCH_SIZE] row_align(1);

static void global_average_pool(
    const elem_t *in_data,
    int batch_size,
    int out_channels,
    int out_row_dim,
    int out_col_dim)
{
    int spatial = out_row_dim * out_col_dim;
    for (int b = 0; b < batch_size; b++) {
        for (int c = 0; c < out_channels; c++) {
            float sum = 0.0f;
            for (int r = 0; r < out_row_dim; r++) {
                for (int col = 0; col < out_col_dim; col++) {
                    int patch_idx = b * spatial + r * out_col_dim + col;
                    sum += elem_bits_to_float(in_data[patch_idx * out_channels + c]);
                }
            }
            average[c][b] = float_to_elem_bits(sum / (float)spatial);
        }
    }
}

// ---------------------------------------------------------------------------
// Run one batch of images through the full ResNet-50 network
// ---------------------------------------------------------------------------
static void run_resnet50_batch(
    const elem_t *current_images,
    enum tiled_matmul_type_t matmul_type)
{
    // ---- Stem ----
    // conv_1: 3x3, stride 1, padding 1, 3->64, 32x32->32x32, RELU
    tiled_conv_auto(
        conv_1_params.batch_size,
        conv_1_params.in_row_dim, conv_1_params.in_col_dim,
        conv_1_params.in_channels,
        conv_1_params.out_channels,
        conv_1_params.out_row_dim, conv_1_params.out_col_dim,
        conv_1_params.stride, 1, 1,
        conv_1_params.padding,
        conv_1_params.kernel_size,
        false, false, false, false, false,
        (elem_t *)current_images,
        (elem_t *)conv_1_w,
        (acc_t  *)conv_1_b,
        (elem_t *)conv_1_out,
        RELU, conv_1_params.output_scale,
        conv_1_params.pool_size, 0, conv_1_params.pool_padding,
        matmul_type);

    // ======================================================================
    // Stage 0: 3 bottleneck blocks, 32x32, 64/64/256 channels
    // ======================================================================

    // -- Block 0 (projection shortcut: conv_1_out->conv_5_out) --
    // conv_2: 1x1, 64->64, RELU
    tiled_matmul_nn_auto(
        conv_2_params.I, conv_2_params.J, conv_2_params.K,
        (elem_t *)conv_1_out, (elem_t *)conv_2_w, (acc_t *)conv_2_b,
        (elem_t *)conv_2_out,
        RELU, conv_2_params.output_scale, true, matmul_type, false, "conv_2");

    // conv_3: 3x3, 64->64, RELU
    tiled_conv_auto(
        conv_3_params.batch_size,
        conv_3_params.in_row_dim, conv_3_params.in_col_dim,
        conv_3_params.in_channels, conv_3_params.out_channels,
        conv_3_params.out_row_dim, conv_3_params.out_col_dim,
        conv_3_params.stride, 1, 1,
        conv_3_params.padding, conv_3_params.kernel_size,
        false, false, false, false, false,
        (elem_t *)conv_2_out, (elem_t *)conv_3_w, (acc_t *)conv_3_b,
        (elem_t *)conv_3_out,
        RELU, conv_3_params.output_scale,
        conv_3_params.pool_size, 0, conv_3_params.pool_padding,
        matmul_type);

    // conv_4: 1x1, 64->256, NO_ACTIVATION
    tiled_matmul_nn_auto(
        conv_4_params.I, conv_4_params.J, conv_4_params.K,
        (elem_t *)conv_3_out, (elem_t *)conv_4_w, (acc_t *)conv_4_b,
        (elem_t *)conv_4_out,
        NO_ACTIVATION, conv_4_params.output_scale, true, matmul_type, false, "conv_4");

    // conv_5: 1x1 projection shortcut, stride=1, 64->256, NO_ACTIVATION
    tiled_matmul_nn_auto(
        conv_5_params.I, conv_5_params.J, conv_5_params.K,
        (elem_t *)conv_1_out, (elem_t *)conv_5_w, (acc_t *)conv_5_b,
        (elem_t *)conv_5_out,
        NO_ACTIVATION, conv_5_params.output_scale, true, matmul_type, false, "conv_5");

    // resadd: conv_5_out + conv_4_out -> conv_4_out, in-place
    tiled_resadd_auto(
        conv_4_params.I, conv_4_params.J,
        conv_4_params.res_scale, MVIN_SCALE_IDENTITY, ACC_SCALE_IDENTITY,
        (elem_t *)conv_5_out, (elem_t *)conv_4_out, (elem_t *)conv_4_out,
        true, WS);

    // -- Block 1 (identity shortcut: conv_4_out->conv_8_out) --
    tiled_matmul_nn_auto(
        conv_6_params.I, conv_6_params.J, conv_6_params.K,
        (elem_t *)conv_4_out, (elem_t *)conv_6_w, (acc_t *)conv_6_b,
        (elem_t *)conv_6_out,
        RELU, conv_6_params.output_scale, true, matmul_type, false, "conv_6");

    tiled_conv_auto(
        conv_7_params.batch_size,
        conv_7_params.in_row_dim, conv_7_params.in_col_dim,
        conv_7_params.in_channels, conv_7_params.out_channels,
        conv_7_params.out_row_dim, conv_7_params.out_col_dim,
        conv_7_params.stride, 1, 1,
        conv_7_params.padding, conv_7_params.kernel_size,
        false, false, false, false, false,
        (elem_t *)conv_6_out, (elem_t *)conv_7_w, (acc_t *)conv_7_b,
        (elem_t *)conv_7_out,
        RELU, conv_7_params.output_scale,
        conv_7_params.pool_size, 0, conv_7_params.pool_padding,
        matmul_type);

    tiled_matmul_nn_auto(
        conv_8_params.I, conv_8_params.J, conv_8_params.K,
        (elem_t *)conv_7_out, (elem_t *)conv_8_w, (acc_t *)conv_8_b,
        (elem_t *)conv_8_out,
        NO_ACTIVATION, conv_8_params.output_scale, true, matmul_type, false, "conv_8");

    tiled_resadd_auto(
        conv_8_params.I, conv_8_params.J,
        conv_8_params.res_scale, MVIN_SCALE_IDENTITY, ACC_SCALE_IDENTITY,
        (elem_t *)conv_4_out, (elem_t *)conv_8_out, (elem_t *)conv_8_out,
        true, WS);

    // -- Block 2 (identity shortcut: conv_8_out->conv_11_out) --
    tiled_matmul_nn_auto(
        conv_9_params.I, conv_9_params.J, conv_9_params.K,
        (elem_t *)conv_8_out, (elem_t *)conv_9_w, (acc_t *)conv_9_b,
        (elem_t *)conv_9_out,
        RELU, conv_9_params.output_scale, true, matmul_type, false, "conv_9");

    tiled_conv_auto(
        conv_10_params.batch_size,
        conv_10_params.in_row_dim, conv_10_params.in_col_dim,
        conv_10_params.in_channels, conv_10_params.out_channels,
        conv_10_params.out_row_dim, conv_10_params.out_col_dim,
        conv_10_params.stride, 1, 1,
        conv_10_params.padding, conv_10_params.kernel_size,
        false, false, false, false, false,
        (elem_t *)conv_9_out, (elem_t *)conv_10_w, (acc_t *)conv_10_b,
        (elem_t *)conv_10_out,
        RELU, conv_10_params.output_scale,
        conv_10_params.pool_size, 0, conv_10_params.pool_padding,
        matmul_type);

    tiled_matmul_nn_auto(
        conv_11_params.I, conv_11_params.J, conv_11_params.K,
        (elem_t *)conv_10_out, (elem_t *)conv_11_w, (acc_t *)conv_11_b,
        (elem_t *)conv_11_out,
        NO_ACTIVATION, conv_11_params.output_scale, true, matmul_type, false, "conv_11");

    tiled_resadd_auto(
        conv_11_params.I, conv_11_params.J,
        conv_11_params.res_scale, MVIN_SCALE_IDENTITY, ACC_SCALE_IDENTITY,
        (elem_t *)conv_8_out, (elem_t *)conv_11_out, (elem_t *)conv_11_out,
        true, WS);

    // ======================================================================
    // Stage 1: 4 bottleneck blocks, 128/128/512 channels
    //   Block0 input: 32x32 (conv_11_out), output: 16x16 (via stride-2)
    //   Blocks 1-3: 16x16
    // ======================================================================

    // -- Block 0 (stride-2 3x3, projection shortcut conv_15) --
    // conv_12: 1x1, 256->128, RELU, stride 1
    tiled_matmul_nn_auto(
        conv_12_params.I, conv_12_params.J, conv_12_params.K,
        (elem_t *)conv_11_out, (elem_t *)conv_12_w, (acc_t *)conv_12_b,
        (elem_t *)conv_12_out,
        RELU, conv_12_params.output_scale, true, matmul_type, false, "conv_12");

    // conv_13: 3x3, stride 2, 128->128, RELU (32x32->16x16)
    tiled_conv_auto(
        conv_13_params.batch_size,
        conv_13_params.in_row_dim, conv_13_params.in_col_dim,
        conv_13_params.in_channels, conv_13_params.out_channels,
        conv_13_params.out_row_dim, conv_13_params.out_col_dim,
        conv_13_params.stride, 1, 1,
        conv_13_params.padding, conv_13_params.kernel_size,
        false, false, false, false, false,
        (elem_t *)conv_12_out, (elem_t *)conv_13_w, (acc_t *)conv_13_b,
        (elem_t *)conv_13_out,
        RELU, conv_13_params.output_scale,
        conv_13_params.pool_size, 0, conv_13_params.pool_padding,
        matmul_type);

    // conv_14: 1x1, 128->512, NO_ACTIVATION
    tiled_matmul_nn_auto(
        conv_14_params.I, conv_14_params.J, conv_14_params.K,
        (elem_t *)conv_13_out, (elem_t *)conv_14_w, (acc_t *)conv_14_b,
        (elem_t *)conv_14_out,
        NO_ACTIVATION, conv_14_params.output_scale, true, matmul_type, false, "conv_14");

    // conv_15: 1x1 stride-2 projection shortcut, 256->512 (32x32->16x16)
    tiled_conv_downsample(
        conv_15_params.batch_size,
        conv_15_params.in_row_dim, conv_15_params.in_col_dim,
        conv_15_params.in_channels,
        conv_15_params.out_channels,
        conv_15_params.out_row_dim, conv_15_params.out_col_dim,
        conv_15_params.in_channels, conv_15_params.out_channels, conv_15_params.out_channels,
        (elem_t *)conv_11_out, (elem_t *)conv_15_w, (acc_t *)conv_15_b,
        (elem_t *)conv_15_out,
        NO_ACTIVATION, conv_15_params.output_scale,
        matmul_type);

    tiled_resadd_auto(
        conv_14_params.I, conv_14_params.J,
        conv_14_params.res_scale, MVIN_SCALE_IDENTITY, ACC_SCALE_IDENTITY,
        (elem_t *)conv_15_out, (elem_t *)conv_14_out, (elem_t *)conv_14_out,
        true, WS);

    // -- Block 1 (identity shortcut: conv_14_out->conv_18_out) --
    tiled_matmul_nn_auto(
        conv_16_params.I, conv_16_params.J, conv_16_params.K,
        (elem_t *)conv_14_out, (elem_t *)conv_16_w, (acc_t *)conv_16_b,
        (elem_t *)conv_16_out,
        RELU, conv_16_params.output_scale, true, matmul_type, false, "conv_16");

    tiled_conv_auto(
        conv_17_params.batch_size,
        conv_17_params.in_row_dim, conv_17_params.in_col_dim,
        conv_17_params.in_channels, conv_17_params.out_channels,
        conv_17_params.out_row_dim, conv_17_params.out_col_dim,
        conv_17_params.stride, 1, 1,
        conv_17_params.padding, conv_17_params.kernel_size,
        false, false, false, false, false,
        (elem_t *)conv_16_out, (elem_t *)conv_17_w, (acc_t *)conv_17_b,
        (elem_t *)conv_17_out,
        RELU, conv_17_params.output_scale,
        conv_17_params.pool_size, 0, conv_17_params.pool_padding,
        matmul_type);

    tiled_matmul_nn_auto(
        conv_18_params.I, conv_18_params.J, conv_18_params.K,
        (elem_t *)conv_17_out, (elem_t *)conv_18_w, (acc_t *)conv_18_b,
        (elem_t *)conv_18_out,
        NO_ACTIVATION, conv_18_params.output_scale, true, matmul_type, false, "conv_18");

    tiled_resadd_auto(
        conv_18_params.I, conv_18_params.J,
        conv_18_params.res_scale, MVIN_SCALE_IDENTITY, ACC_SCALE_IDENTITY,
        (elem_t *)conv_14_out, (elem_t *)conv_18_out, (elem_t *)conv_18_out,
        true, WS);

    // -- Block 2 (identity shortcut: conv_18_out->conv_21_out) --
    tiled_matmul_nn_auto(
        conv_19_params.I, conv_19_params.J, conv_19_params.K,
        (elem_t *)conv_18_out, (elem_t *)conv_19_w, (acc_t *)conv_19_b,
        (elem_t *)conv_19_out,
        RELU, conv_19_params.output_scale, true, matmul_type, false, "conv_19");

    tiled_conv_auto(
        conv_20_params.batch_size,
        conv_20_params.in_row_dim, conv_20_params.in_col_dim,
        conv_20_params.in_channels, conv_20_params.out_channels,
        conv_20_params.out_row_dim, conv_20_params.out_col_dim,
        conv_20_params.stride, 1, 1,
        conv_20_params.padding, conv_20_params.kernel_size,
        false, false, false, false, false,
        (elem_t *)conv_19_out, (elem_t *)conv_20_w, (acc_t *)conv_20_b,
        (elem_t *)conv_20_out,
        RELU, conv_20_params.output_scale,
        conv_20_params.pool_size, 0, conv_20_params.pool_padding,
        matmul_type);

    tiled_matmul_nn_auto(
        conv_21_params.I, conv_21_params.J, conv_21_params.K,
        (elem_t *)conv_20_out, (elem_t *)conv_21_w, (acc_t *)conv_21_b,
        (elem_t *)conv_21_out,
        NO_ACTIVATION, conv_21_params.output_scale, true, matmul_type, false, "conv_21");

    tiled_resadd_auto(
        conv_21_params.I, conv_21_params.J,
        conv_21_params.res_scale, MVIN_SCALE_IDENTITY, ACC_SCALE_IDENTITY,
        (elem_t *)conv_18_out, (elem_t *)conv_21_out, (elem_t *)conv_21_out,
        true, WS);

    // -- Block 3 (identity shortcut: conv_21_out->conv_24_out) --
    tiled_matmul_nn_auto(
        conv_22_params.I, conv_22_params.J, conv_22_params.K,
        (elem_t *)conv_21_out, (elem_t *)conv_22_w, (acc_t *)conv_22_b,
        (elem_t *)conv_22_out,
        RELU, conv_22_params.output_scale, true, matmul_type, false, "conv_22");

    tiled_conv_auto(
        conv_23_params.batch_size,
        conv_23_params.in_row_dim, conv_23_params.in_col_dim,
        conv_23_params.in_channels, conv_23_params.out_channels,
        conv_23_params.out_row_dim, conv_23_params.out_col_dim,
        conv_23_params.stride, 1, 1,
        conv_23_params.padding, conv_23_params.kernel_size,
        false, false, false, false, false,
        (elem_t *)conv_22_out, (elem_t *)conv_23_w, (acc_t *)conv_23_b,
        (elem_t *)conv_23_out,
        RELU, conv_23_params.output_scale,
        conv_23_params.pool_size, 0, conv_23_params.pool_padding,
        matmul_type);

    tiled_matmul_nn_auto(
        conv_24_params.I, conv_24_params.J, conv_24_params.K,
        (elem_t *)conv_23_out, (elem_t *)conv_24_w, (acc_t *)conv_24_b,
        (elem_t *)conv_24_out,
        NO_ACTIVATION, conv_24_params.output_scale, true, matmul_type, false, "conv_24");

    tiled_resadd_auto(
        conv_24_params.I, conv_24_params.J,
        conv_24_params.res_scale, MVIN_SCALE_IDENTITY, ACC_SCALE_IDENTITY,
        (elem_t *)conv_21_out, (elem_t *)conv_24_out, (elem_t *)conv_24_out,
        true, WS);

    // ======================================================================
    // Stage 2: 6 bottleneck blocks, 256/256/1024 channels
    //   Block0 input: 16x16 (conv_24_out), output: 8x8
    //   Blocks 1-5: 8x8
    // ======================================================================

    // -- Block 0 (stride-2 3x3, projection shortcut conv_28) --
    // conv_25: 1x1, 512->256, RELU
    tiled_matmul_nn_auto(
        conv_25_params.I, conv_25_params.J, conv_25_params.K,
        (elem_t *)conv_24_out, (elem_t *)conv_25_w, (acc_t *)conv_25_b,
        (elem_t *)conv_25_out,
        RELU, conv_25_params.output_scale, true, matmul_type, false, "conv_25");

    // conv_26: 3x3, stride 2, 256->256, RELU (16x16->8x8)
    tiled_conv_auto(
        conv_26_params.batch_size,
        conv_26_params.in_row_dim, conv_26_params.in_col_dim,
        conv_26_params.in_channels, conv_26_params.out_channels,
        conv_26_params.out_row_dim, conv_26_params.out_col_dim,
        conv_26_params.stride, 1, 1,
        conv_26_params.padding, conv_26_params.kernel_size,
        false, false, false, false, false,
        (elem_t *)conv_25_out, (elem_t *)conv_26_w, (acc_t *)conv_26_b,
        (elem_t *)conv_26_out,
        RELU, conv_26_params.output_scale,
        conv_26_params.pool_size, 0, conv_26_params.pool_padding,
        matmul_type);

    // conv_27: 1x1, 256->1024, NO_ACTIVATION
    tiled_matmul_nn_auto(
        conv_27_params.I, conv_27_params.J, conv_27_params.K,
        (elem_t *)conv_26_out, (elem_t *)conv_27_w, (acc_t *)conv_27_b,
        (elem_t *)conv_27_out,
        NO_ACTIVATION, conv_27_params.output_scale, true, matmul_type, false, "conv_27");

    // conv_28: 1x1 stride-2 projection shortcut, 512->1024 (16x16->8x8)
    tiled_conv_downsample(
        conv_28_params.batch_size,
        conv_28_params.in_row_dim, conv_28_params.in_col_dim,
        conv_28_params.in_channels,
        conv_28_params.out_channels,
        conv_28_params.out_row_dim, conv_28_params.out_col_dim,
        conv_28_params.in_channels, conv_28_params.out_channels, conv_28_params.out_channels,
        (elem_t *)conv_24_out, (elem_t *)conv_28_w, (acc_t *)conv_28_b,
        (elem_t *)conv_28_out,
        NO_ACTIVATION, conv_28_params.output_scale,
        matmul_type);

    tiled_resadd_auto(
        conv_27_params.I, conv_27_params.J,
        conv_27_params.res_scale, MVIN_SCALE_IDENTITY, ACC_SCALE_IDENTITY,
        (elem_t *)conv_28_out, (elem_t *)conv_27_out, (elem_t *)conv_27_out,
        true, WS);

    // -- Block 1 (identity shortcut: conv_27_out->conv_31_out) --
    tiled_matmul_nn_auto(
        conv_29_params.I, conv_29_params.J, conv_29_params.K,
        (elem_t *)conv_27_out, (elem_t *)conv_29_w, (acc_t *)conv_29_b,
        (elem_t *)conv_29_out,
        RELU, conv_29_params.output_scale, true, matmul_type, false, "conv_29");

    tiled_conv_auto(
        conv_30_params.batch_size,
        conv_30_params.in_row_dim, conv_30_params.in_col_dim,
        conv_30_params.in_channels, conv_30_params.out_channels,
        conv_30_params.out_row_dim, conv_30_params.out_col_dim,
        conv_30_params.stride, 1, 1,
        conv_30_params.padding, conv_30_params.kernel_size,
        false, false, false, false, false,
        (elem_t *)conv_29_out, (elem_t *)conv_30_w, (acc_t *)conv_30_b,
        (elem_t *)conv_30_out,
        RELU, conv_30_params.output_scale,
        conv_30_params.pool_size, 0, conv_30_params.pool_padding,
        matmul_type);

    tiled_matmul_nn_auto(
        conv_31_params.I, conv_31_params.J, conv_31_params.K,
        (elem_t *)conv_30_out, (elem_t *)conv_31_w, (acc_t *)conv_31_b,
        (elem_t *)conv_31_out,
        NO_ACTIVATION, conv_31_params.output_scale, true, matmul_type, false, "conv_31");

    tiled_resadd_auto(
        conv_31_params.I, conv_31_params.J,
        conv_31_params.res_scale, MVIN_SCALE_IDENTITY, ACC_SCALE_IDENTITY,
        (elem_t *)conv_27_out, (elem_t *)conv_31_out, (elem_t *)conv_31_out,
        true, WS);

    // -- Block 2 (identity shortcut: conv_31_out->conv_34_out) --
    tiled_matmul_nn_auto(
        conv_32_params.I, conv_32_params.J, conv_32_params.K,
        (elem_t *)conv_31_out, (elem_t *)conv_32_w, (acc_t *)conv_32_b,
        (elem_t *)conv_32_out,
        RELU, conv_32_params.output_scale, true, matmul_type, false, "conv_32");

    tiled_conv_auto(
        conv_33_params.batch_size,
        conv_33_params.in_row_dim, conv_33_params.in_col_dim,
        conv_33_params.in_channels, conv_33_params.out_channels,
        conv_33_params.out_row_dim, conv_33_params.out_col_dim,
        conv_33_params.stride, 1, 1,
        conv_33_params.padding, conv_33_params.kernel_size,
        false, false, false, false, false,
        (elem_t *)conv_32_out, (elem_t *)conv_33_w, (acc_t *)conv_33_b,
        (elem_t *)conv_33_out,
        RELU, conv_33_params.output_scale,
        conv_33_params.pool_size, 0, conv_33_params.pool_padding,
        matmul_type);

    tiled_matmul_nn_auto(
        conv_34_params.I, conv_34_params.J, conv_34_params.K,
        (elem_t *)conv_33_out, (elem_t *)conv_34_w, (acc_t *)conv_34_b,
        (elem_t *)conv_34_out,
        NO_ACTIVATION, conv_34_params.output_scale, true, matmul_type, false, "conv_34");

    tiled_resadd_auto(
        conv_34_params.I, conv_34_params.J,
        conv_34_params.res_scale, MVIN_SCALE_IDENTITY, ACC_SCALE_IDENTITY,
        (elem_t *)conv_31_out, (elem_t *)conv_34_out, (elem_t *)conv_34_out,
        true, WS);

    // -- Block 3 (identity shortcut: conv_34_out->conv_37_out) --
    tiled_matmul_nn_auto(
        conv_35_params.I, conv_35_params.J, conv_35_params.K,
        (elem_t *)conv_34_out, (elem_t *)conv_35_w, (acc_t *)conv_35_b,
        (elem_t *)conv_35_out,
        RELU, conv_35_params.output_scale, true, matmul_type, false, "conv_35");

    tiled_conv_auto(
        conv_36_params.batch_size,
        conv_36_params.in_row_dim, conv_36_params.in_col_dim,
        conv_36_params.in_channels, conv_36_params.out_channels,
        conv_36_params.out_row_dim, conv_36_params.out_col_dim,
        conv_36_params.stride, 1, 1,
        conv_36_params.padding, conv_36_params.kernel_size,
        false, false, false, false, false,
        (elem_t *)conv_35_out, (elem_t *)conv_36_w, (acc_t *)conv_36_b,
        (elem_t *)conv_36_out,
        RELU, conv_36_params.output_scale,
        conv_36_params.pool_size, 0, conv_36_params.pool_padding,
        matmul_type);

    tiled_matmul_nn_auto(
        conv_37_params.I, conv_37_params.J, conv_37_params.K,
        (elem_t *)conv_36_out, (elem_t *)conv_37_w, (acc_t *)conv_37_b,
        (elem_t *)conv_37_out,
        NO_ACTIVATION, conv_37_params.output_scale, true, matmul_type, false, "conv_37");

    tiled_resadd_auto(
        conv_37_params.I, conv_37_params.J,
        conv_37_params.res_scale, MVIN_SCALE_IDENTITY, ACC_SCALE_IDENTITY,
        (elem_t *)conv_34_out, (elem_t *)conv_37_out, (elem_t *)conv_37_out,
        true, WS);

    // -- Block 4 (identity shortcut: conv_37_out->conv_40_out) --
    tiled_matmul_nn_auto(
        conv_38_params.I, conv_38_params.J, conv_38_params.K,
        (elem_t *)conv_37_out, (elem_t *)conv_38_w, (acc_t *)conv_38_b,
        (elem_t *)conv_38_out,
        RELU, conv_38_params.output_scale, true, matmul_type, false, "conv_38");

    tiled_conv_auto(
        conv_39_params.batch_size,
        conv_39_params.in_row_dim, conv_39_params.in_col_dim,
        conv_39_params.in_channels, conv_39_params.out_channels,
        conv_39_params.out_row_dim, conv_39_params.out_col_dim,
        conv_39_params.stride, 1, 1,
        conv_39_params.padding, conv_39_params.kernel_size,
        false, false, false, false, false,
        (elem_t *)conv_38_out, (elem_t *)conv_39_w, (acc_t *)conv_39_b,
        (elem_t *)conv_39_out,
        RELU, conv_39_params.output_scale,
        conv_39_params.pool_size, 0, conv_39_params.pool_padding,
        matmul_type);

    tiled_matmul_nn_auto(
        conv_40_params.I, conv_40_params.J, conv_40_params.K,
        (elem_t *)conv_39_out, (elem_t *)conv_40_w, (acc_t *)conv_40_b,
        (elem_t *)conv_40_out,
        NO_ACTIVATION, conv_40_params.output_scale, true, matmul_type, false, "conv_40");

    tiled_resadd_auto(
        conv_40_params.I, conv_40_params.J,
        conv_40_params.res_scale, MVIN_SCALE_IDENTITY, ACC_SCALE_IDENTITY,
        (elem_t *)conv_37_out, (elem_t *)conv_40_out, (elem_t *)conv_40_out,
        true, WS);

    // -- Block 5 (identity shortcut: conv_40_out->conv_43_out) --
    tiled_matmul_nn_auto(
        conv_41_params.I, conv_41_params.J, conv_41_params.K,
        (elem_t *)conv_40_out, (elem_t *)conv_41_w, (acc_t *)conv_41_b,
        (elem_t *)conv_41_out,
        RELU, conv_41_params.output_scale, true, matmul_type, false, "conv_41");

    tiled_conv_auto(
        conv_42_params.batch_size,
        conv_42_params.in_row_dim, conv_42_params.in_col_dim,
        conv_42_params.in_channels, conv_42_params.out_channels,
        conv_42_params.out_row_dim, conv_42_params.out_col_dim,
        conv_42_params.stride, 1, 1,
        conv_42_params.padding, conv_42_params.kernel_size,
        false, false, false, false, false,
        (elem_t *)conv_41_out, (elem_t *)conv_42_w, (acc_t *)conv_42_b,
        (elem_t *)conv_42_out,
        RELU, conv_42_params.output_scale,
        conv_42_params.pool_size, 0, conv_42_params.pool_padding,
        matmul_type);

    tiled_matmul_nn_auto(
        conv_43_params.I, conv_43_params.J, conv_43_params.K,
        (elem_t *)conv_42_out, (elem_t *)conv_43_w, (acc_t *)conv_43_b,
        (elem_t *)conv_43_out,
        NO_ACTIVATION, conv_43_params.output_scale, true, matmul_type, false, "conv_43");

    tiled_resadd_auto(
        conv_43_params.I, conv_43_params.J,
        conv_43_params.res_scale, MVIN_SCALE_IDENTITY, ACC_SCALE_IDENTITY,
        (elem_t *)conv_40_out, (elem_t *)conv_43_out, (elem_t *)conv_43_out,
        true, WS);

    // ======================================================================
    // Stage 3: 3 bottleneck blocks, 512/512/2048 channels
    //   Block0 input: 8x8 (conv_43_out), output: 4x4
    //   Blocks 1-2: 4x4
    // ======================================================================

    // -- Block 0 (stride-2 3x3, projection shortcut conv_47) --
    // conv_44: 1x1, 1024->512, RELU
    tiled_matmul_nn_auto(
        conv_44_params.I, conv_44_params.J, conv_44_params.K,
        (elem_t *)conv_43_out, (elem_t *)conv_44_w, (acc_t *)conv_44_b,
        (elem_t *)conv_44_out,
        RELU, conv_44_params.output_scale, true, matmul_type, false, "conv_44");

    // conv_45: 3x3, stride 2, 512->512, RELU (8x8->4x4)
    tiled_conv_auto(
        conv_45_params.batch_size,
        conv_45_params.in_row_dim, conv_45_params.in_col_dim,
        conv_45_params.in_channels, conv_45_params.out_channels,
        conv_45_params.out_row_dim, conv_45_params.out_col_dim,
        conv_45_params.stride, 1, 1,
        conv_45_params.padding, conv_45_params.kernel_size,
        false, false, false, false, false,
        (elem_t *)conv_44_out, (elem_t *)conv_45_w, (acc_t *)conv_45_b,
        (elem_t *)conv_45_out,
        RELU, conv_45_params.output_scale,
        conv_45_params.pool_size, 0, conv_45_params.pool_padding,
        matmul_type);

    // conv_46: 1x1, 512->2048, NO_ACTIVATION
    tiled_matmul_nn_auto(
        conv_46_params.I, conv_46_params.J, conv_46_params.K,
        (elem_t *)conv_45_out, (elem_t *)conv_46_w, (acc_t *)conv_46_b,
        (elem_t *)conv_46_out,
        NO_ACTIVATION, conv_46_params.output_scale, true, matmul_type, false, "conv_46");

    // conv_47: 1x1 stride-2 projection shortcut, 1024->2048 (8x8->4x4)
    tiled_conv_downsample(
        conv_47_params.batch_size,
        conv_47_params.in_row_dim, conv_47_params.in_col_dim,
        conv_47_params.in_channels,
        conv_47_params.out_channels,
        conv_47_params.out_row_dim, conv_47_params.out_col_dim,
        conv_47_params.in_channels, conv_47_params.out_channels, conv_47_params.out_channels,
        (elem_t *)conv_43_out, (elem_t *)conv_47_w, (acc_t *)conv_47_b,
        (elem_t *)conv_47_out,
        NO_ACTIVATION, conv_47_params.output_scale,
        matmul_type);

    tiled_resadd_auto(
        conv_46_params.I, conv_46_params.J,
        conv_46_params.res_scale, MVIN_SCALE_IDENTITY, ACC_SCALE_IDENTITY,
        (elem_t *)conv_47_out, (elem_t *)conv_46_out, (elem_t *)conv_46_out,
        true, WS);

    // -- Block 1 (identity shortcut: conv_46_out->conv_50_out) --
    tiled_matmul_nn_auto(
        conv_48_params.I, conv_48_params.J, conv_48_params.K,
        (elem_t *)conv_46_out, (elem_t *)conv_48_w, (acc_t *)conv_48_b,
        (elem_t *)conv_48_out,
        RELU, conv_48_params.output_scale, true, matmul_type, false, "conv_48");

    tiled_conv_auto(
        conv_49_params.batch_size,
        conv_49_params.in_row_dim, conv_49_params.in_col_dim,
        conv_49_params.in_channels, conv_49_params.out_channels,
        conv_49_params.out_row_dim, conv_49_params.out_col_dim,
        conv_49_params.stride, 1, 1,
        conv_49_params.padding, conv_49_params.kernel_size,
        false, false, false, false, false,
        (elem_t *)conv_48_out, (elem_t *)conv_49_w, (acc_t *)conv_49_b,
        (elem_t *)conv_49_out,
        RELU, conv_49_params.output_scale,
        conv_49_params.pool_size, 0, conv_49_params.pool_padding,
        matmul_type);

    tiled_matmul_nn_auto(
        conv_50_params.I, conv_50_params.J, conv_50_params.K,
        (elem_t *)conv_49_out, (elem_t *)conv_50_w, (acc_t *)conv_50_b,
        (elem_t *)conv_50_out,
        NO_ACTIVATION, conv_50_params.output_scale, true, matmul_type, false, "conv_50");

    tiled_resadd_auto(
        conv_50_params.I, conv_50_params.J,
        conv_50_params.res_scale, MVIN_SCALE_IDENTITY, ACC_SCALE_IDENTITY,
        (elem_t *)conv_46_out, (elem_t *)conv_50_out, (elem_t *)conv_50_out,
        true, WS);

    // -- Block 2 (identity shortcut: conv_50_out->conv_53_out) --
    tiled_matmul_nn_auto(
        conv_51_params.I, conv_51_params.J, conv_51_params.K,
        (elem_t *)conv_50_out, (elem_t *)conv_51_w, (acc_t *)conv_51_b,
        (elem_t *)conv_51_out,
        RELU, conv_51_params.output_scale, true, matmul_type, false, "conv_51");

    tiled_conv_auto(
        conv_52_params.batch_size,
        conv_52_params.in_row_dim, conv_52_params.in_col_dim,
        conv_52_params.in_channels, conv_52_params.out_channels,
        conv_52_params.out_row_dim, conv_52_params.out_col_dim,
        conv_52_params.stride, 1, 1,
        conv_52_params.padding, conv_52_params.kernel_size,
        false, false, false, false, false,
        (elem_t *)conv_51_out, (elem_t *)conv_52_w, (acc_t *)conv_52_b,
        (elem_t *)conv_52_out,
        RELU, conv_52_params.output_scale,
        conv_52_params.pool_size, 0, conv_52_params.pool_padding,
        matmul_type);

    tiled_matmul_nn_auto(
        conv_53_params.I, conv_53_params.J, conv_53_params.K,
        (elem_t *)conv_52_out, (elem_t *)conv_53_w, (acc_t *)conv_53_b,
        (elem_t *)conv_53_out,
        NO_ACTIVATION, conv_53_params.output_scale, true, matmul_type, false, "conv_53");

    tiled_resadd_auto(
        conv_53_params.I, conv_53_params.J,
        conv_53_params.res_scale, MVIN_SCALE_IDENTITY, ACC_SCALE_IDENTITY,
        (elem_t *)conv_50_out, (elem_t *)conv_53_out, (elem_t *)conv_53_out,
        true, WS);

    // ======================================================================
    // Global Average Pooling: conv_53_out[64][2048] -> average[2048][4]
    //   conv_53_out layout: [n_patches][channels] where n_patches = batch*4*4
    //   average layout: [channels][batch]
    // ======================================================================
    global_average_pool(
        (const elem_t *)conv_53_out,
        conv_53_params.batch_size,
        conv_53_params.out_channels,
        conv_53_params.out_row_dim,
        conv_53_params.out_col_dim);

    // ======================================================================
    // FC: fc_54_w[10][2048], average[2048][4] -> fc_54_out[10][4]
    // ======================================================================
    tiled_matmul_nn_auto(
        fc_54_params.I, fc_54_params.J, fc_54_params.K,
        (elem_t *)fc_54_w, (elem_t *)average, (acc_t *)fc_54_b,
        (elem_t *)fc_54_out,
        NO_ACTIVATION, fc_54_params.output_scale, false, matmul_type, false, "fc_54");
}

// ---------------------------------------------------------------------------
// Top-K prediction helper (float comparison)
// ---------------------------------------------------------------------------
static int topk_correct(int batch, int true_label, int k)
{
    int num_classes = fc_54_params.out_features;
    static int sorted_idx[10];
    for (int i = 0; i < num_classes; i++)
        sorted_idx[i] = i;
    // Simple insertion sort (num_classes=10, negligible overhead)
    for (int i = 1; i < num_classes; i++) {
        int key = sorted_idx[i];
        float key_score = elem_bits_to_float(fc_54_out[key][batch]);
        int j = i - 1;
        while (j >= 0 && elem_bits_to_float(fc_54_out[sorted_idx[j]][batch]) < key_score) {
            sorted_idx[j + 1] = sorted_idx[j];
            j--;
        }
        sorted_idx[j + 1] = key;
    }
    for (int i = 0; i < k; i++) {
        if (sorted_idx[i] == true_label)
            return 1;
    }
    return 0;
}

static int argmax_batch(int batch)
{
    int num_classes = fc_54_params.out_features;
    int best = 0;
    for (int i = 1; i < num_classes; i++) {
        if (elem_bits_to_float(fc_54_out[i][batch]) > elem_bits_to_float(fc_54_out[best][batch]))
            best = i;
    }
    return best;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char *argv[])
{
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
        perror("mlockall failed");
        exit(1);
    }
#endif

    enum tiled_matmul_type_t matmul_type = WS;
    if (argc > 1) {
        if      (!strcmp(argv[1], "os"))  matmul_type = OS;
        else if (!strcmp(argv[1], "cpu")) matmul_type = CPU;
        else                              matmul_type = WS;
    }

    gemmini_flush(0);

    printf("\n");
    printf("=================================================================\n");
    printf("  ResNet50 CIFAR-10 Streaming Inference (Float/FP32)\n");
    printf("  Model: edadaltocg/resnet50_cifar10 (BN folded, output_scale=1.0f)\n");
    printf("  Input: 32x32x3,  Classes: 10,  Batch: %d\n", BATCH_SIZE);
    printf("  Total test images: %d\n", NUM_IMAGES);
    printf("  matmul_type: %s\n",
           matmul_type == CPU ? "CPU" : matmul_type == OS ? "OS" : "WS");
    printf("=================================================================\n\n");

    // ------------------------------------------------------------------
    // A/B test on 4 known CIFAR-10 images (from cifar10_images.h)
    // Expected labels: cat=3, ship=8, ship=8, airplane=0
    // ------------------------------------------------------------------
    const int expected_labels[BATCH_SIZE] = {3, 8, 8, 0};
    printf("--- A/B Test on 4 known CIFAR-10 images (cifar10_images.h) ---\n");
    printf("Expected: cat, ship, ship, airplane\n\n");

    uint64_t ab_start = get_time_ns();
    run_resnet50_batch((const elem_t *)cifar10_images, matmul_type);
    uint64_t ab_elapsed = get_time_ns() - ab_start;

    for (int b = 0; b < BATCH_SIZE; b++) {
        int pred = argmax_batch(b);
        const char *result = (pred == expected_labels[b]) ? "PASS" : "FAIL";
        printf("  Image %d: predicted=%s (%d), expected=%s (%d)  [%s]\n",
               b, class_names[pred], pred,
               class_names[expected_labels[b]], expected_labels[b], result);
    }
    printf("  A/B batch time: %.3f ms\n\n", ab_elapsed / 1e6);

    // ------------------------------------------------------------------
    // Streaming inference over CIFAR-10 test set
    // ------------------------------------------------------------------
    FILE *img_fp = fopen(IMAGES_BIN_FILE, "rb");
    if (!img_fp) {
        fprintf(stderr, "ERROR: Cannot open %s\n", IMAGES_BIN_FILE);
        return 1;
    }
    FILE *lbl_fp = fopen(LABELS_TXT_FILE, "r");
    if (!lbl_fp) {
        fclose(img_fp);
        fprintf(stderr, "ERROR: Cannot open %s\n", LABELS_TXT_FILE);
        return 1;
    }

    // Only one batch of images in memory at a time (~48 KB for float CIFAR-10)
    static elem_t image_batch[BATCH_SIZE * IMAGE_SIZE];

    int top1_correct_window = 0;
    int top5_correct_window = 0;
    int top1_correct_total  = 0;
    int top5_correct_total  = 0;
    int images_processed    = 0;

    int num_batches = NUM_IMAGES / BATCH_SIZE;

    uint64_t min_batch_cycles = UINT64_MAX, min_batch_wall = UINT64_MAX;
    uint64_t sum_batch_cycles = 0, sum_batch_wall = 0;
    float best_window_top1 = 0.0f, best_window_top5 = 0.0f;

    setvbuf(stdout, NULL, _IONBF, 0);

    for (int batch_idx = 0; batch_idx < num_batches; batch_idx++) {
        int true_labels[BATCH_SIZE];

        // Load batch from binary file (each image: IMAGE_SIZE bytes, uint8)
        // Normalize uint8 [0,255] -> float [0,1] for FP32 inference
        for (int b = 0; b < BATCH_SIZE; b++) {
            uint8_t raw[IMAGE_SIZE];
            size_t n = fread(raw, 1, IMAGE_SIZE, img_fp);
            if (n != IMAGE_SIZE) {
                fprintf(stderr, "ERROR: Unexpected EOF at batch %d image %d\n",
                        batch_idx, b);
                goto done;
            }
            for (int i = 0; i < IMAGE_SIZE; i++)
                image_batch[b * IMAGE_SIZE + i] = float_to_elem_bits(raw[i] / 255.0f);
        }

        // Load labels
        for (int b = 0; b < BATCH_SIZE; b++) {
            if (fscanf(lbl_fp, "%d", &true_labels[b]) != 1) {
                fprintf(stderr, "ERROR: Cannot read label at batch %d image %d\n",
                        batch_idx, b);
                goto done;
            }
        }

        uint64_t cycle_start = bench_read_cycles();
        uint64_t wall_start = get_time_ns();

        run_resnet50_batch((const elem_t *)image_batch, matmul_type);

        uint64_t cycle_end = bench_read_cycles();
        uint64_t wall_end = get_time_ns();

        uint64_t batch_cycles = cycle_end - cycle_start;
        uint64_t batch_wall = wall_end - wall_start;
        if (batch_cycles < min_batch_cycles) min_batch_cycles = batch_cycles;
        if (batch_wall < min_batch_wall) min_batch_wall = batch_wall;
        sum_batch_cycles += batch_cycles;
        sum_batch_wall += batch_wall;

        printf("Batch %d/%d  Cycles: %llu  Time: %llu ns\n",
               batch_idx + 1, num_batches,
               (unsigned long long)batch_cycles, (unsigned long long)batch_wall);

        // Evaluate predictions
        for (int b = 0; b < BATCH_SIZE; b++) {
            int t1 = topk_correct(b, true_labels[b], 1);
            int t5 = topk_correct(b, true_labels[b], TOP_K);
            top1_correct_window += t1;
            top5_correct_window += t5;
            top1_correct_total  += t1;
            top5_correct_total  += t5;
        }
        images_processed += BATCH_SIZE;

        // Report every 100 images (25 batches)
        if ((batch_idx + 1) % 25 == 0) {
            int window_size = 25 * BATCH_SIZE;
            printf("[%5d/%d]  Window top-1: %3d/%d (%.1f%%)  top-5: %3d/%d (%.1f%%)  "
                   "Cumulative top-1: %.2f%%  top-5: %.2f%%\n",
                   images_processed, NUM_IMAGES,
                   top1_correct_window, window_size,
                   100.0f * top1_correct_window / window_size,
                   top5_correct_window, window_size,
                   100.0f * top5_correct_window / window_size,
                   100.0f * top1_correct_total  / images_processed,
                   100.0f * top5_correct_total  / images_processed);
            float w_top1 = 100.0f * top1_correct_window / window_size;
            float w_top5 = 100.0f * top5_correct_window / window_size;
            if (w_top1 > best_window_top1) best_window_top1 = w_top1;
            if (w_top5 > best_window_top5) best_window_top5 = w_top5;
            top1_correct_window = 0;
            top5_correct_window = 0;
        }
    }

done:
    fclose(img_fp);
    fclose(lbl_fp);

    uint64_t avg_batch_cycles = (num_batches > 0) ? sum_batch_cycles / num_batches : 0;
    uint64_t avg_batch_wall = (num_batches > 0) ? sum_batch_wall / num_batches : 0;
    float final_top1 = (images_processed > 0) ? 100.0f * top1_correct_total / images_processed : 0;
    float final_top5 = (images_processed > 0) ? 100.0f * top5_correct_total / images_processed : 0;
    double imgs_per_sec = (avg_batch_wall > 0) ? (double)BATCH_SIZE * 1e9 / avg_batch_wall : 0;

    printf("\n=================================================================\n");
    printf("  Final Results (%d images processed)\n", images_processed);
    printf("  Top-1 accuracy: %d / %d = %.2f%%\n",
           top1_correct_total, images_processed, final_top1);
    printf("  Top-5 accuracy: %d / %d = %.2f%%\n",
           top5_correct_total, images_processed, final_top5);
    printf("  Best window top-1: %.1f%%  top-5: %.1f%%\n", best_window_top1, best_window_top5);
    printf("  Min batch cycles: %llu  Avg batch cycles: %llu\n",
           (unsigned long long)min_batch_cycles, (unsigned long long)avg_batch_cycles);
    printf("  Min batch wall: %llu ns  Avg batch wall: %llu ns\n",
           (unsigned long long)min_batch_wall, (unsigned long long)avg_batch_wall);
    printf("  Throughput: %.2f images/sec\n", imgs_per_sec);
    printf("=================================================================\n");

    printf("\nCSV,ResNet50-CIFAR10-Float,cifar10,32x32,%d,%.2f,%.2f,0.00,%.1f,%.1f,%llu,%llu,%llu,%llu,%.2f\n",
           images_processed, final_top1, final_top5,
           best_window_top1, best_window_top5,
           (unsigned long long)min_batch_cycles, (unsigned long long)avg_batch_cycles,
           (unsigned long long)min_batch_wall, (unsigned long long)avg_batch_wall,
           imgs_per_sec);

    return 0;
}
