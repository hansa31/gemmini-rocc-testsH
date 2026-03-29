#!/usr/bin/env python3
"""
Extract INT8-quantized weights from HuggingFace ResNet-50 (microsoft/resnet-50)
and generate:
  - ../resnet50_params.h  (INT8 weights, INT32 biases, power-of-2 output_scale)

This replaces the incorrectly calibrated resnet50_params.h that had a wrong
header guard (MOBILENET_PARAMETERS_H) and no known generation script.

Architecture: standard ImageNet ResNet-50
  - Stem: 7x7 conv, stride 2, pad 3 + 3x3 maxpool, stride 2
  - Stage 0 (layer1): 3 bottleneck blocks, 64->64->256 channels
  - Stage 1 (layer2): 4 bottleneck blocks, 128->128->512 channels
  - Stage 2 (layer3): 6 bottleneck blocks, 256->256->1024 channels
  - Stage 3 (layer4): 3 bottleneck blocks, 512->512->2048 channels
  - Global avg pool -> FC (2048->1000)

Quantization approach (same as extract_resnet50_cifar10_int.py):
  - Per-layer symmetric weight quantization: w_int8 = clip(round(w / scale), -128, 127)
  - INT32 biases: b_int32 = round(b_float / (w_scale * x_scale))
  - Exact float output_scale: output_scale = w_scale * x_scale / y_scale
  - Activation range from BatchNorm running statistics (3-sigma rule)
  - Scale propagation tracks x_scale sequentially through the network
  - res_scale for residual adds: y_range[skip_src] / y_range[main_layer]

Input preprocessing assumed: (pixel/255 - mean)/std -> int8  (x_scale_0 = FMAX/127 = 2.75/127)

Usage:
    pip install torch transformers numpy
    python extract_resnet50_imagenet_int8.py

Output:
    ../resnet50_params.h
"""

import os
import argparse
import numpy as np
import torch
from transformers import ResNetForImageClassification

MODEL_NAME = "microsoft/resnet-50"
INPUT_DIM = 224
BATCH_SIZE = 4
NUM_CLASSES = 1000

# Input preprocessing constants (must match prepare_imagenet.py)
# x_float = (pixel/255 - mean) / std  ->  x_int8 = clip(round(x_float/FMAX*127), -128, 127)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
FMAX = 2.75  # x_scale_0 = FMAX / 127.0

# ---------------------------------------------------------------------------
# HuggingFace-to-Gemmini layer mapping (identical to extract_resnet50_weights.py)
# ---------------------------------------------------------------------------
LAYER_MAPPING = [
    # Initial 7x7 conv + BN
    ("conv_1",  "resnet.embedder.embedder",                         "conv"),

    # Stage 0 (layer1) - 3 bottleneck blocks
    ("conv_2",  "resnet.encoder.stages.0.layers.0.layer.0",         "conv"),
    ("conv_3",  "resnet.encoder.stages.0.layers.0.layer.1",         "conv"),
    ("conv_4",  "resnet.encoder.stages.0.layers.0.layer.2",         "conv"),
    ("conv_5",  "resnet.encoder.stages.0.layers.0.shortcut",        "conv"),  # projection skip

    ("conv_6",  "resnet.encoder.stages.0.layers.1.layer.0",         "conv"),
    ("conv_7",  "resnet.encoder.stages.0.layers.1.layer.1",         "conv"),
    ("conv_8",  "resnet.encoder.stages.0.layers.1.layer.2",         "conv"),

    ("conv_9",  "resnet.encoder.stages.0.layers.2.layer.0",         "conv"),
    ("conv_10", "resnet.encoder.stages.0.layers.2.layer.1",         "conv"),
    ("conv_11", "resnet.encoder.stages.0.layers.2.layer.2",         "conv"),

    # Stage 1 (layer2) - 4 bottleneck blocks
    ("conv_12", "resnet.encoder.stages.1.layers.0.layer.0",         "conv"),
    ("conv_13", "resnet.encoder.stages.1.layers.0.layer.1",         "conv"),  # 3x3 stride 2
    ("conv_14", "resnet.encoder.stages.1.layers.0.layer.2",         "conv"),
    ("conv_15", "resnet.encoder.stages.1.layers.0.shortcut",        "conv"),  # projection skip

    ("conv_16", "resnet.encoder.stages.1.layers.1.layer.0",         "conv"),
    ("conv_17", "resnet.encoder.stages.1.layers.1.layer.1",         "conv"),
    ("conv_18", "resnet.encoder.stages.1.layers.1.layer.2",         "conv"),

    ("conv_19", "resnet.encoder.stages.1.layers.2.layer.0",         "conv"),
    ("conv_20", "resnet.encoder.stages.1.layers.2.layer.1",         "conv"),
    ("conv_21", "resnet.encoder.stages.1.layers.2.layer.2",         "conv"),

    ("conv_22", "resnet.encoder.stages.1.layers.3.layer.0",         "conv"),
    ("conv_23", "resnet.encoder.stages.1.layers.3.layer.1",         "conv"),
    ("conv_24", "resnet.encoder.stages.1.layers.3.layer.2",         "conv"),

    # Stage 2 (layer3) - 6 bottleneck blocks
    ("conv_25", "resnet.encoder.stages.2.layers.0.layer.0",         "conv"),
    ("conv_26", "resnet.encoder.stages.2.layers.0.layer.1",         "conv"),  # 3x3 stride 2
    ("conv_27", "resnet.encoder.stages.2.layers.0.layer.2",         "conv"),
    ("conv_28", "resnet.encoder.stages.2.layers.0.shortcut",        "conv"),  # projection skip

    ("conv_29", "resnet.encoder.stages.2.layers.1.layer.0",         "conv"),
    ("conv_30", "resnet.encoder.stages.2.layers.1.layer.1",         "conv"),
    ("conv_31", "resnet.encoder.stages.2.layers.1.layer.2",         "conv"),

    ("conv_32", "resnet.encoder.stages.2.layers.2.layer.0",         "conv"),
    ("conv_33", "resnet.encoder.stages.2.layers.2.layer.1",         "conv"),
    ("conv_34", "resnet.encoder.stages.2.layers.2.layer.2",         "conv"),

    ("conv_35", "resnet.encoder.stages.2.layers.3.layer.0",         "conv"),
    ("conv_36", "resnet.encoder.stages.2.layers.3.layer.1",         "conv"),
    ("conv_37", "resnet.encoder.stages.2.layers.3.layer.2",         "conv"),

    ("conv_38", "resnet.encoder.stages.2.layers.4.layer.0",         "conv"),
    ("conv_39", "resnet.encoder.stages.2.layers.4.layer.1",         "conv"),
    ("conv_40", "resnet.encoder.stages.2.layers.4.layer.2",         "conv"),

    ("conv_41", "resnet.encoder.stages.2.layers.5.layer.0",         "conv"),
    ("conv_42", "resnet.encoder.stages.2.layers.5.layer.1",         "conv"),
    ("conv_43", "resnet.encoder.stages.2.layers.5.layer.2",         "conv"),

    # Stage 3 (layer4) - 3 bottleneck blocks
    ("conv_44", "resnet.encoder.stages.3.layers.0.layer.0",         "conv"),
    ("conv_45", "resnet.encoder.stages.3.layers.0.layer.1",         "conv"),  # 3x3 stride 2
    ("conv_46", "resnet.encoder.stages.3.layers.0.layer.2",         "conv"),
    ("conv_47", "resnet.encoder.stages.3.layers.0.shortcut",        "conv"),  # projection skip

    ("conv_48", "resnet.encoder.stages.3.layers.1.layer.0",         "conv"),
    ("conv_49", "resnet.encoder.stages.3.layers.1.layer.1",         "conv"),
    ("conv_50", "resnet.encoder.stages.3.layers.1.layer.2",         "conv"),

    ("conv_51", "resnet.encoder.stages.3.layers.2.layer.0",         "conv"),
    ("conv_52", "resnet.encoder.stages.3.layers.2.layer.1",         "conv"),
    ("conv_53", "resnet.encoder.stages.3.layers.2.layer.2",         "conv"),

    ("fc_54",   "classifier",                                        "fc"),
]

# ---------------------------------------------------------------------------
# Conv layer parameters (spatial dimensions, channels, etc.)
# res_scale is computed dynamically and NOT stored here.
# fmt: off
# ---------------------------------------------------------------------------
CONV_PARAMS = {
    "conv_1":  {"batch_size": 4, "in_row_dim": 224, "in_col_dim": 224, "kernel_size": 7,  "in_channels": 3,    "out_channels": 64,   "stride": 2, "padding": 3, "bias": 1, "depthwise": 0, "out_row_dim": 112, "out_col_dim": 112, "n_patches": 50176, "patch_size": 147,  "pool_size": 3, "pool_stride": 2, "pool_padding": 1, "out_dim_pooled": 56,  "I": 50176, "J": 64,   "K": 147},
    "conv_2":  {"batch_size": 4, "in_row_dim": 56,  "in_col_dim": 56,  "kernel_size": 1,  "in_channels": 64,   "out_channels": 64,   "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 56,  "out_col_dim": 56,  "n_patches": 12544, "patch_size": 64,   "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 56,  "I": 12544, "J": 64,   "K": 64},
    "conv_3":  {"batch_size": 4, "in_row_dim": 56,  "in_col_dim": 56,  "kernel_size": 3,  "in_channels": 64,   "out_channels": 64,   "stride": 1, "padding": 1, "bias": 1, "depthwise": 0, "out_row_dim": 56,  "out_col_dim": 56,  "n_patches": 12544, "patch_size": 576,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 56,  "I": 12544, "J": 64,   "K": 576},
    "conv_4":  {"batch_size": 4, "in_row_dim": 56,  "in_col_dim": 56,  "kernel_size": 1,  "in_channels": 64,   "out_channels": 256,  "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 56,  "out_col_dim": 56,  "n_patches": 12544, "patch_size": 64,   "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 56,  "I": 12544, "J": 256,  "K": 64},
    "conv_5":  {"batch_size": 4, "in_row_dim": 56,  "in_col_dim": 56,  "kernel_size": 1,  "in_channels": 64,   "out_channels": 256,  "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 56,  "out_col_dim": 56,  "n_patches": 12544, "patch_size": 64,   "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 56,  "I": 12544, "J": 256,  "K": 64},
    "conv_6":  {"batch_size": 4, "in_row_dim": 56,  "in_col_dim": 56,  "kernel_size": 1,  "in_channels": 256,  "out_channels": 64,   "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 56,  "out_col_dim": 56,  "n_patches": 12544, "patch_size": 256,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 56,  "I": 12544, "J": 64,   "K": 256},
    "conv_7":  {"batch_size": 4, "in_row_dim": 56,  "in_col_dim": 56,  "kernel_size": 3,  "in_channels": 64,   "out_channels": 64,   "stride": 1, "padding": 1, "bias": 1, "depthwise": 0, "out_row_dim": 56,  "out_col_dim": 56,  "n_patches": 12544, "patch_size": 576,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 56,  "I": 12544, "J": 64,   "K": 576},
    "conv_8":  {"batch_size": 4, "in_row_dim": 56,  "in_col_dim": 56,  "kernel_size": 1,  "in_channels": 64,   "out_channels": 256,  "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 56,  "out_col_dim": 56,  "n_patches": 12544, "patch_size": 64,   "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 56,  "I": 12544, "J": 256,  "K": 64},
    "conv_9":  {"batch_size": 4, "in_row_dim": 56,  "in_col_dim": 56,  "kernel_size": 1,  "in_channels": 256,  "out_channels": 64,   "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 56,  "out_col_dim": 56,  "n_patches": 12544, "patch_size": 256,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 56,  "I": 12544, "J": 64,   "K": 256},
    "conv_10": {"batch_size": 4, "in_row_dim": 56,  "in_col_dim": 56,  "kernel_size": 3,  "in_channels": 64,   "out_channels": 64,   "stride": 1, "padding": 1, "bias": 1, "depthwise": 0, "out_row_dim": 56,  "out_col_dim": 56,  "n_patches": 12544, "patch_size": 576,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 56,  "I": 12544, "J": 64,   "K": 576},
    "conv_11": {"batch_size": 4, "in_row_dim": 56,  "in_col_dim": 56,  "kernel_size": 1,  "in_channels": 64,   "out_channels": 256,  "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 56,  "out_col_dim": 56,  "n_patches": 12544, "patch_size": 64,   "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 56,  "I": 12544, "J": 256,  "K": 64},
    "conv_12": {"batch_size": 4, "in_row_dim": 56,  "in_col_dim": 56,  "kernel_size": 1,  "in_channels": 256,  "out_channels": 128,  "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 56,  "out_col_dim": 56,  "n_patches": 12544, "patch_size": 256,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 56,  "I": 12544, "J": 128,  "K": 256},
    "conv_13": {"batch_size": 4, "in_row_dim": 56,  "in_col_dim": 56,  "kernel_size": 3,  "in_channels": 128,  "out_channels": 128,  "stride": 2, "padding": 1, "bias": 1, "depthwise": 0, "out_row_dim": 28,  "out_col_dim": 28,  "n_patches": 3136,  "patch_size": 1152, "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 28,  "I": 3136,  "J": 128,  "K": 1152},
    "conv_14": {"batch_size": 4, "in_row_dim": 28,  "in_col_dim": 28,  "kernel_size": 1,  "in_channels": 128,  "out_channels": 512,  "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 28,  "out_col_dim": 28,  "n_patches": 3136,  "patch_size": 128,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 28,  "I": 3136,  "J": 512,  "K": 128},
    "conv_15": {"batch_size": 4, "in_row_dim": 56,  "in_col_dim": 56,  "kernel_size": 1,  "in_channels": 256,  "out_channels": 512,  "stride": 2, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 28,  "out_col_dim": 28,  "n_patches": 3136,  "patch_size": 256,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 28,  "I": 3136,  "J": 512,  "K": 256},
    "conv_16": {"batch_size": 4, "in_row_dim": 28,  "in_col_dim": 28,  "kernel_size": 1,  "in_channels": 512,  "out_channels": 128,  "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 28,  "out_col_dim": 28,  "n_patches": 3136,  "patch_size": 512,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 28,  "I": 3136,  "J": 128,  "K": 512},
    "conv_17": {"batch_size": 4, "in_row_dim": 28,  "in_col_dim": 28,  "kernel_size": 3,  "in_channels": 128,  "out_channels": 128,  "stride": 1, "padding": 1, "bias": 1, "depthwise": 0, "out_row_dim": 28,  "out_col_dim": 28,  "n_patches": 3136,  "patch_size": 1152, "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 28,  "I": 3136,  "J": 128,  "K": 1152},
    "conv_18": {"batch_size": 4, "in_row_dim": 28,  "in_col_dim": 28,  "kernel_size": 1,  "in_channels": 128,  "out_channels": 512,  "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 28,  "out_col_dim": 28,  "n_patches": 3136,  "patch_size": 128,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 28,  "I": 3136,  "J": 512,  "K": 128},
    "conv_19": {"batch_size": 4, "in_row_dim": 28,  "in_col_dim": 28,  "kernel_size": 1,  "in_channels": 512,  "out_channels": 128,  "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 28,  "out_col_dim": 28,  "n_patches": 3136,  "patch_size": 512,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 28,  "I": 3136,  "J": 128,  "K": 512},
    "conv_20": {"batch_size": 4, "in_row_dim": 28,  "in_col_dim": 28,  "kernel_size": 3,  "in_channels": 128,  "out_channels": 128,  "stride": 1, "padding": 1, "bias": 1, "depthwise": 0, "out_row_dim": 28,  "out_col_dim": 28,  "n_patches": 3136,  "patch_size": 1152, "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 28,  "I": 3136,  "J": 128,  "K": 1152},
    "conv_21": {"batch_size": 4, "in_row_dim": 28,  "in_col_dim": 28,  "kernel_size": 1,  "in_channels": 128,  "out_channels": 512,  "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 28,  "out_col_dim": 28,  "n_patches": 3136,  "patch_size": 128,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 28,  "I": 3136,  "J": 512,  "K": 128},
    "conv_22": {"batch_size": 4, "in_row_dim": 28,  "in_col_dim": 28,  "kernel_size": 1,  "in_channels": 512,  "out_channels": 128,  "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 28,  "out_col_dim": 28,  "n_patches": 3136,  "patch_size": 512,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 28,  "I": 3136,  "J": 128,  "K": 512},
    "conv_23": {"batch_size": 4, "in_row_dim": 28,  "in_col_dim": 28,  "kernel_size": 3,  "in_channels": 128,  "out_channels": 128,  "stride": 1, "padding": 1, "bias": 1, "depthwise": 0, "out_row_dim": 28,  "out_col_dim": 28,  "n_patches": 3136,  "patch_size": 1152, "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 28,  "I": 3136,  "J": 128,  "K": 1152},
    "conv_24": {"batch_size": 4, "in_row_dim": 28,  "in_col_dim": 28,  "kernel_size": 1,  "in_channels": 128,  "out_channels": 512,  "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 28,  "out_col_dim": 28,  "n_patches": 3136,  "patch_size": 128,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 28,  "I": 3136,  "J": 512,  "K": 128},
    "conv_25": {"batch_size": 4, "in_row_dim": 28,  "in_col_dim": 28,  "kernel_size": 1,  "in_channels": 512,  "out_channels": 256,  "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 28,  "out_col_dim": 28,  "n_patches": 3136,  "patch_size": 512,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 28,  "I": 3136,  "J": 256,  "K": 512},
    "conv_26": {"batch_size": 4, "in_row_dim": 28,  "in_col_dim": 28,  "kernel_size": 3,  "in_channels": 256,  "out_channels": 256,  "stride": 2, "padding": 1, "bias": 1, "depthwise": 0, "out_row_dim": 14,  "out_col_dim": 14,  "n_patches": 784,   "patch_size": 2304, "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 14,  "I": 784,   "J": 256,  "K": 2304},
    "conv_27": {"batch_size": 4, "in_row_dim": 14,  "in_col_dim": 14,  "kernel_size": 1,  "in_channels": 256,  "out_channels": 1024, "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 14,  "out_col_dim": 14,  "n_patches": 784,   "patch_size": 256,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 14,  "I": 784,   "J": 1024, "K": 256},
    "conv_28": {"batch_size": 4, "in_row_dim": 28,  "in_col_dim": 28,  "kernel_size": 1,  "in_channels": 512,  "out_channels": 1024, "stride": 2, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 14,  "out_col_dim": 14,  "n_patches": 784,   "patch_size": 512,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 14,  "I": 784,   "J": 1024, "K": 512},
    "conv_29": {"batch_size": 4, "in_row_dim": 14,  "in_col_dim": 14,  "kernel_size": 1,  "in_channels": 1024, "out_channels": 256,  "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 14,  "out_col_dim": 14,  "n_patches": 784,   "patch_size": 1024, "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 14,  "I": 784,   "J": 256,  "K": 1024},
    "conv_30": {"batch_size": 4, "in_row_dim": 14,  "in_col_dim": 14,  "kernel_size": 3,  "in_channels": 256,  "out_channels": 256,  "stride": 1, "padding": 1, "bias": 1, "depthwise": 0, "out_row_dim": 14,  "out_col_dim": 14,  "n_patches": 784,   "patch_size": 2304, "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 14,  "I": 784,   "J": 256,  "K": 2304},
    "conv_31": {"batch_size": 4, "in_row_dim": 14,  "in_col_dim": 14,  "kernel_size": 1,  "in_channels": 256,  "out_channels": 1024, "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 14,  "out_col_dim": 14,  "n_patches": 784,   "patch_size": 256,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 14,  "I": 784,   "J": 1024, "K": 256},
    "conv_32": {"batch_size": 4, "in_row_dim": 14,  "in_col_dim": 14,  "kernel_size": 1,  "in_channels": 1024, "out_channels": 256,  "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 14,  "out_col_dim": 14,  "n_patches": 784,   "patch_size": 1024, "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 14,  "I": 784,   "J": 256,  "K": 1024},
    "conv_33": {"batch_size": 4, "in_row_dim": 14,  "in_col_dim": 14,  "kernel_size": 3,  "in_channels": 256,  "out_channels": 256,  "stride": 1, "padding": 1, "bias": 1, "depthwise": 0, "out_row_dim": 14,  "out_col_dim": 14,  "n_patches": 784,   "patch_size": 2304, "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 14,  "I": 784,   "J": 256,  "K": 2304},
    "conv_34": {"batch_size": 4, "in_row_dim": 14,  "in_col_dim": 14,  "kernel_size": 1,  "in_channels": 256,  "out_channels": 1024, "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 14,  "out_col_dim": 14,  "n_patches": 784,   "patch_size": 256,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 14,  "I": 784,   "J": 1024, "K": 256},
    "conv_35": {"batch_size": 4, "in_row_dim": 14,  "in_col_dim": 14,  "kernel_size": 1,  "in_channels": 1024, "out_channels": 256,  "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 14,  "out_col_dim": 14,  "n_patches": 784,   "patch_size": 1024, "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 14,  "I": 784,   "J": 256,  "K": 1024},
    "conv_36": {"batch_size": 4, "in_row_dim": 14,  "in_col_dim": 14,  "kernel_size": 3,  "in_channels": 256,  "out_channels": 256,  "stride": 1, "padding": 1, "bias": 1, "depthwise": 0, "out_row_dim": 14,  "out_col_dim": 14,  "n_patches": 784,   "patch_size": 2304, "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 14,  "I": 784,   "J": 256,  "K": 2304},
    "conv_37": {"batch_size": 4, "in_row_dim": 14,  "in_col_dim": 14,  "kernel_size": 1,  "in_channels": 256,  "out_channels": 1024, "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 14,  "out_col_dim": 14,  "n_patches": 784,   "patch_size": 256,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 14,  "I": 784,   "J": 1024, "K": 256},
    "conv_38": {"batch_size": 4, "in_row_dim": 14,  "in_col_dim": 14,  "kernel_size": 1,  "in_channels": 1024, "out_channels": 256,  "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 14,  "out_col_dim": 14,  "n_patches": 784,   "patch_size": 1024, "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 14,  "I": 784,   "J": 256,  "K": 1024},
    "conv_39": {"batch_size": 4, "in_row_dim": 14,  "in_col_dim": 14,  "kernel_size": 3,  "in_channels": 256,  "out_channels": 256,  "stride": 1, "padding": 1, "bias": 1, "depthwise": 0, "out_row_dim": 14,  "out_col_dim": 14,  "n_patches": 784,   "patch_size": 2304, "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 14,  "I": 784,   "J": 256,  "K": 2304},
    "conv_40": {"batch_size": 4, "in_row_dim": 14,  "in_col_dim": 14,  "kernel_size": 1,  "in_channels": 256,  "out_channels": 1024, "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 14,  "out_col_dim": 14,  "n_patches": 784,   "patch_size": 256,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 14,  "I": 784,   "J": 1024, "K": 256},
    "conv_41": {"batch_size": 4, "in_row_dim": 14,  "in_col_dim": 14,  "kernel_size": 1,  "in_channels": 1024, "out_channels": 256,  "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 14,  "out_col_dim": 14,  "n_patches": 784,   "patch_size": 1024, "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 14,  "I": 784,   "J": 256,  "K": 1024},
    "conv_42": {"batch_size": 4, "in_row_dim": 14,  "in_col_dim": 14,  "kernel_size": 3,  "in_channels": 256,  "out_channels": 256,  "stride": 1, "padding": 1, "bias": 1, "depthwise": 0, "out_row_dim": 14,  "out_col_dim": 14,  "n_patches": 784,   "patch_size": 2304, "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 14,  "I": 784,   "J": 256,  "K": 2304},
    "conv_43": {"batch_size": 4, "in_row_dim": 14,  "in_col_dim": 14,  "kernel_size": 1,  "in_channels": 256,  "out_channels": 1024, "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 14,  "out_col_dim": 14,  "n_patches": 784,   "patch_size": 256,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 14,  "I": 784,   "J": 1024, "K": 256},
    "conv_44": {"batch_size": 4, "in_row_dim": 14,  "in_col_dim": 14,  "kernel_size": 1,  "in_channels": 1024, "out_channels": 512,  "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 14,  "out_col_dim": 14,  "n_patches": 784,   "patch_size": 1024, "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 14,  "I": 784,   "J": 512,  "K": 1024},
    "conv_45": {"batch_size": 4, "in_row_dim": 14,  "in_col_dim": 14,  "kernel_size": 3,  "in_channels": 512,  "out_channels": 512,  "stride": 2, "padding": 1, "bias": 1, "depthwise": 0, "out_row_dim": 7,   "out_col_dim": 7,   "n_patches": 196,   "patch_size": 4608, "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 7,   "I": 196,   "J": 512,  "K": 4608},
    "conv_46": {"batch_size": 4, "in_row_dim": 7,   "in_col_dim": 7,   "kernel_size": 1,  "in_channels": 512,  "out_channels": 2048, "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 7,   "out_col_dim": 7,   "n_patches": 196,   "patch_size": 512,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 7,   "I": 196,   "J": 2048, "K": 512},
    "conv_47": {"batch_size": 4, "in_row_dim": 14,  "in_col_dim": 14,  "kernel_size": 1,  "in_channels": 1024, "out_channels": 2048, "stride": 2, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 7,   "out_col_dim": 7,   "n_patches": 196,   "patch_size": 1024, "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 7,   "I": 196,   "J": 2048, "K": 1024},
    "conv_48": {"batch_size": 4, "in_row_dim": 7,   "in_col_dim": 7,   "kernel_size": 1,  "in_channels": 2048, "out_channels": 512,  "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 7,   "out_col_dim": 7,   "n_patches": 196,   "patch_size": 2048, "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 7,   "I": 196,   "J": 512,  "K": 2048},
    "conv_49": {"batch_size": 4, "in_row_dim": 7,   "in_col_dim": 7,   "kernel_size": 3,  "in_channels": 512,  "out_channels": 512,  "stride": 1, "padding": 1, "bias": 1, "depthwise": 0, "out_row_dim": 7,   "out_col_dim": 7,   "n_patches": 196,   "patch_size": 4608, "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 7,   "I": 196,   "J": 512,  "K": 4608},
    "conv_50": {"batch_size": 4, "in_row_dim": 7,   "in_col_dim": 7,   "kernel_size": 1,  "in_channels": 512,  "out_channels": 2048, "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 7,   "out_col_dim": 7,   "n_patches": 196,   "patch_size": 512,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 7,   "I": 196,   "J": 2048, "K": 512},
    "conv_51": {"batch_size": 4, "in_row_dim": 7,   "in_col_dim": 7,   "kernel_size": 1,  "in_channels": 2048, "out_channels": 512,  "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 7,   "out_col_dim": 7,   "n_patches": 196,   "patch_size": 2048, "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 7,   "I": 196,   "J": 512,  "K": 2048},
    "conv_52": {"batch_size": 4, "in_row_dim": 7,   "in_col_dim": 7,   "kernel_size": 3,  "in_channels": 512,  "out_channels": 512,  "stride": 1, "padding": 1, "bias": 1, "depthwise": 0, "out_row_dim": 7,   "out_col_dim": 7,   "n_patches": 196,   "patch_size": 4608, "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 7,   "I": 196,   "J": 512,  "K": 4608},
    "conv_53": {"batch_size": 4, "in_row_dim": 7,   "in_col_dim": 7,   "kernel_size": 1,  "in_channels": 512,  "out_channels": 2048, "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 7,   "out_col_dim": 7,   "n_patches": 196,   "patch_size": 512,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 7,   "I": 196,   "J": 2048, "K": 512},
}
# fmt: on

# FC params: tiled_matmul_nn_auto(I=4, J=1000, K=2048, A=average[4][2048], B=fc_54_w[2048][1000])
FC_PARAMS = {
    "fc_54": {
        "batch_size": BATCH_SIZE,
        "in_features": 2048,
        "out_features": NUM_CLASSES,
        "bias": 1,
        "I": BATCH_SIZE,
        "J": NUM_CLASSES,
        "K": 2048,
    }
}

# Buffer declarations (from extract_resnet50_weights.py)
BUFFERS = [
    ("conv_1_in",          50176, 147),
    ("conv_1_out",         50176, 64),
    ("conv_1_out_pooled",  None,  None),  # special: [4][56][56][64]

    ("conv_2_in",   12544, 64),    ("conv_2_out",  12544, 64),
    ("conv_3_in",   12544, 576),   ("conv_3_out",  12544, 64),
    ("conv_4_in",   12544, 64),    ("conv_4_out",  12544, 256),
    ("conv_5_in",   12544, 64),    ("conv_5_out",  12544, 256),
    ("conv_6_in",   12544, 256),   ("conv_6_out",  12544, 64),
    ("conv_7_in",   12544, 576),   ("conv_7_out",  12544, 64),
    ("conv_8_in",   12544, 64),    ("conv_8_out",  12544, 256),
    ("conv_9_in",   12544, 256),   ("conv_9_out",  12544, 64),
    ("conv_10_in",  12544, 576),   ("conv_10_out", 12544, 64),
    ("conv_11_in",  12544, 64),    ("conv_11_out", 12544, 256),
    ("conv_12_in",  12544, 256),   ("conv_12_out", 12544, 128),
    ("conv_13_in",  3136,  1152),  ("conv_13_out", 3136,  128),
    ("conv_14_in",  3136,  128),   ("conv_14_out", 3136,  512),
    ("conv_15_in",  3136,  256),   ("conv_15_out", 3136,  512),
    ("conv_16_in",  3136,  512),   ("conv_16_out", 3136,  128),
    ("conv_17_in",  3136,  1152),  ("conv_17_out", 3136,  128),
    ("conv_18_in",  3136,  128),   ("conv_18_out", 3136,  512),
    ("conv_19_in",  3136,  512),   ("conv_19_out", 3136,  128),
    ("conv_20_in",  3136,  1152),  ("conv_20_out", 3136,  128),
    ("conv_21_in",  3136,  128),   ("conv_21_out", 3136,  512),
    ("conv_22_in",  3136,  512),   ("conv_22_out", 3136,  128),
    ("conv_23_in",  3136,  1152),  ("conv_23_out", 3136,  128),
    ("conv_24_in",  3136,  128),   ("conv_24_out", 3136,  512),
    ("conv_25_in",  3136,  512),   ("conv_25_out", 3136,  256),
    ("conv_26_in",  784,   2304),  ("conv_26_out", 784,   256),
    ("conv_27_in",  784,   256),   ("conv_27_out", 784,   1024),
    ("conv_28_in",  784,   512),   ("conv_28_out", 784,   1024),
    ("conv_29_in",  784,   1024),  ("conv_29_out", 784,   256),
    ("conv_30_in",  784,   2304),  ("conv_30_out", 784,   256),
    ("conv_31_in",  784,   256),   ("conv_31_out", 784,   1024),
    ("conv_32_in",  784,   1024),  ("conv_32_out", 784,   256),
    ("conv_33_in",  784,   2304),  ("conv_33_out", 784,   256),
    ("conv_34_in",  784,   256),   ("conv_34_out", 784,   1024),
    ("conv_35_in",  784,   1024),  ("conv_35_out", 784,   256),
    ("conv_36_in",  784,   2304),  ("conv_36_out", 784,   256),
    ("conv_37_in",  784,   256),   ("conv_37_out", 784,   1024),
    ("conv_38_in",  784,   1024),  ("conv_38_out", 784,   256),
    ("conv_39_in",  784,   2304),  ("conv_39_out", 784,   256),
    ("conv_40_in",  784,   256),   ("conv_40_out", 784,   1024),
    ("conv_41_in",  784,   1024),  ("conv_41_out", 784,   256),
    ("conv_42_in",  784,   2304),  ("conv_42_out", 784,   256),
    ("conv_43_in",  784,   256),   ("conv_43_out", 784,   1024),
    ("conv_44_in",  784,   1024),  ("conv_44_out", 784,   512),
    ("conv_45_in",  196,   4608),  ("conv_45_out", 196,   512),
    ("conv_46_in",  196,   512),   ("conv_46_out", 196,   2048),
    ("conv_47_in",  196,   1024),  ("conv_47_out", 196,   2048),
    ("conv_48_in",  196,   2048),  ("conv_48_out", 196,   512),
    ("conv_49_in",  196,   4608),  ("conv_49_out", 196,   512),
    ("conv_50_in",  196,   512),   ("conv_50_out", 196,   2048),
    ("conv_51_in",  196,   2048),  ("conv_51_out", 196,   512),
    ("conv_52_in",  196,   4608),  ("conv_52_out", 196,   512),
    ("conv_53_in",  196,   512),   ("conv_53_out", 196,   2048),
    ("fc_54_out",   BATCH_SIZE, NUM_CLASSES),
]

# ReLU activation per layer (True = ReLU after conv, False = linear/no act)
# Bottleneck pattern: 1x1 ReLU, 3x3 ReLU, 1x1 Linear; shortcuts are Linear
LAYER_ACTIVATION = {
    "conv_1":  True,
    "conv_2":  True,  "conv_3":  True,  "conv_4":  False, "conv_5":  False,
    "conv_6":  True,  "conv_7":  True,  "conv_8":  False,
    "conv_9":  True,  "conv_10": True,  "conv_11": False,
    "conv_12": True,  "conv_13": True,  "conv_14": False, "conv_15": False,
    "conv_16": True,  "conv_17": True,  "conv_18": False,
    "conv_19": True,  "conv_20": True,  "conv_21": False,
    "conv_22": True,  "conv_23": True,  "conv_24": False,
    "conv_25": True,  "conv_26": True,  "conv_27": False, "conv_28": False,
    "conv_29": True,  "conv_30": True,  "conv_31": False,
    "conv_32": True,  "conv_33": True,  "conv_34": False,
    "conv_35": True,  "conv_36": True,  "conv_37": False,
    "conv_38": True,  "conv_39": True,  "conv_40": False,
    "conv_41": True,  "conv_42": True,  "conv_43": False,
    "conv_44": True,  "conv_45": True,  "conv_46": False, "conv_47": False,
    "conv_48": True,  "conv_49": True,  "conv_50": False,
    "conv_51": True,  "conv_52": True,  "conv_53": False,
}

# Residual skip connections.
# key   = bottleneck output layer (the one whose params struct contains res_scale)
# value = the skip source layer passed first to tiled_resadd_auto
# From resnet50_v1.c:
#   tiled_resadd_auto(I, J, conv_X_params.res_scale, MVIN_SCALE_IDENTITY, ACC_SCALE_IDENTITY,
#                     conv_SKIP_out, conv_X_out, conv_X_out, relu, WS)
#   => res_scale = y_range[skip] / y_range[conv_X]
RESIDUAL_SKIP = {
    "conv_4":  "conv_5",   # Stage0 block0: projection shortcut
    "conv_8":  "conv_4",   # Stage0 block1: identity
    "conv_11": "conv_8",   # Stage0 block2: identity
    "conv_14": "conv_15",  # Stage1 block0: projection shortcut
    "conv_18": "conv_14",  # Stage1 block1: identity
    "conv_21": "conv_18",  # Stage1 block2: identity
    "conv_24": "conv_21",  # Stage1 block3: identity
    "conv_27": "conv_28",  # Stage2 block0: projection shortcut
    "conv_31": "conv_27",  # Stage2 block1: identity
    "conv_34": "conv_31",  # Stage2 block2: identity
    "conv_37": "conv_34",  # Stage2 block3: identity
    "conv_40": "conv_37",  # Stage2 block4: identity
    "conv_43": "conv_40",  # Stage2 block5: identity
    "conv_46": "conv_47",  # Stage3 block0: projection shortcut
    "conv_50": "conv_46",  # Stage3 block1: identity
    "conv_53": "conv_50",  # Stage3 block2: identity
}


# ---------------------------------------------------------------------------
# BN folding
# ---------------------------------------------------------------------------

def fold_bn(conv_weight, bn_weight, bn_bias, bn_mean, bn_var, eps=1e-5):
    inv_std = 1.0 / np.sqrt(bn_var + eps)
    scale = bn_weight * inv_std
    shape = [conv_weight.shape[0]] + [1] * (conv_weight.ndim - 1)
    w_folded = conv_weight * scale.reshape(shape)
    b_folded = bn_bias - bn_weight * bn_mean * inv_std
    return w_folded, b_folded


def reshape_conv_weight(w):
    """PyTorch [out_ch, in_ch, kH, kW] -> Gemmini [patch_size, out_ch]."""
    out_ch = w.shape[0]
    if w.ndim == 4:
        w = w.transpose(0, 2, 3, 1)  # [out_ch, kH, kW, in_ch]
    return w.reshape(out_ch, -1).T    # [patch_size, out_ch]


def get_conv_bn_params(state_dict, prefix):
    conv_w = state_dict[f"{prefix}.convolution.weight"].float().numpy()
    gamma  = state_dict[f"{prefix}.normalization.weight"].float().numpy()
    beta   = state_dict[f"{prefix}.normalization.bias"].float().numpy()
    mean   = state_dict[f"{prefix}.normalization.running_mean"].float().numpy()
    var    = state_dict[f"{prefix}.normalization.running_var"].float().numpy()
    return conv_w, gamma, beta, mean, var


def extract_and_fold(state_dict, hf_prefix):
    conv_w, gamma, beta, mean, var = get_conv_bn_params(state_dict, hf_prefix)
    w_folded, b_folded = fold_bn(conv_w, gamma, beta, mean, var)
    return reshape_conv_weight(w_folded), b_folded, gamma, beta


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------

def quantize_weight_int8(w_float):
    w_abs_max = np.max(np.abs(w_float))
    if w_abs_max < 1e-10:
        return np.zeros_like(w_float, dtype=np.int8), 1e-10
    w_scale = w_abs_max / 127.0
    w_int = np.clip(np.round(w_float / w_scale), -128, 127).astype(np.int8)
    return w_int, w_scale


def quantize_bias_int32(b_float, combined_scale):
    if combined_scale < 1e-10:
        return np.zeros_like(b_float, dtype=np.int32)
    return np.clip(
        np.round(b_float / combined_scale), -(2**31), 2**31 - 1
    ).astype(np.int32)


def compute_output_scale(w_scale, x_scale, y_range):
    """Exact float output_scale = w_scale * x_scale / y_scale."""
    y_scale = y_range / 127.0
    raw = (w_scale * x_scale) / y_scale
    if raw <= 0 or not np.isfinite(raw):
        raw = 1.0
    return raw, f"{raw:.8e}f"


def estimate_activation_range(bn_gamma, bn_beta, has_relu):
    """Estimate activation output range from BN statistics (3-sigma rule)."""
    gamma_max = np.max(np.abs(bn_gamma))
    beta_max  = np.max(np.abs(bn_beta))
    if has_relu:
        y_range = max(float(np.max(bn_beta + 3.0 * np.abs(bn_gamma))), 1.0)
    else:
        y_range = max(beta_max + 3.0 * gamma_max, 1.0)
    return y_range


def run_calibration(model, image_dir, num_images=100, percentile=99.99):
    """Run calibration images through the float model to measure actual
    per-layer activation ranges using a percentile (not max-abs).

    Max-abs is dominated by outliers and wastes INT8 dynamic range.
    The 99.99th percentile balances clipping avoidance with precision.
    """
    import torchvision.transforms as T
    from PIL import Image

    transform = T.Compose([
        T.Resize((INPUT_DIM, INPUT_DIM)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    # Collect per-layer absolute-value samples
    all_abs = {}

    def make_hook(gemmini_name, has_relu):
        def hook(module, inp, out):
            x = out.detach().float()
            if has_relu:
                x = torch.relu(x)
            vals = x.abs().flatten()
            if vals.numel() > 10000:
                idx = torch.randperm(vals.numel())[:10000]
                vals = vals[idx]
            if gemmini_name not in all_abs:
                all_abs[gemmini_name] = []
            all_abs[gemmini_name].append(vals.numpy())
        return hook

    hooks = []
    for gemmini_name, hf_prefix, layer_type in LAYER_MAPPING:
        if layer_type == "fc":
            continue
        has_relu = LAYER_ACTIVATION.get(gemmini_name, False)
        parts = hf_prefix.split(".")
        mod = model
        for p in parts:
            mod = getattr(mod, p)
        bn_mod = mod.normalization
        h = bn_mod.register_forward_hook(make_hook(gemmini_name, has_relu))
        hooks.append(h)

    files = sorted([
        f for f in os.listdir(image_dir)
        if f.lower().endswith((".jpeg", ".jpg", ".png"))
    ])[:num_images]
    print(f"  Running {len(files)} calibration images from {image_dir}...")

    with torch.no_grad():
        for i, fname in enumerate(files):
            img = Image.open(os.path.join(image_dir, fname)).convert("RGB")
            x = transform(img).unsqueeze(0)
            model(x)

    for h in hooks:
        h.remove()

    # Compute percentile-based ranges
    captured = {}
    for gn in all_abs:
        combined = np.concatenate(all_abs[gn])
        captured[gn] = float(np.percentile(combined, percentile))

    print(f"  Calibrated {len(captured)} layers (percentile={percentile})")
    return captured


# ---------------------------------------------------------------------------
# C formatting
# ---------------------------------------------------------------------------

def fmt_int(v):
    return str(int(v))

def fmt_int_1d(arr):
    return "{" + ",".join(fmt_int(v) for v in arr) + "}"

def fmt_int_2d(arr):
    return "{" + ",".join(fmt_int_1d(row) for row in arr) + "}"


# ---------------------------------------------------------------------------
# Header writing
# ---------------------------------------------------------------------------

def write_conv_layer(f, entry):
    name             = entry["name"]
    w_int            = entry["weight"]
    b_int            = entry["bias"]
    output_scale_str = entry["output_scale_str"]
    res_scale        = entry["res_scale"]
    p                = CONV_PARAMS[name]

    f.write(f"static const elem_t {name}_w[{p['patch_size']}][{p['out_channels']}] row_align(1) = ")
    f.write(fmt_int_2d(w_int))
    f.write(";\n")
    f.write(f"static const acc_t {name}_b[{p['out_channels']}] row_align_acc(1) = ")
    f.write(fmt_int_1d(b_int))
    f.write(";\n")
    for buf_name, d1, d2 in BUFFERS:
        if buf_name == "conv_1_out_pooled":
            if name == "conv_1":
                f.write("static elem_t conv_1_out_pooled[4][56][56][64] row_align(1);\n")
            continue
        if buf_name.startswith(name + "_") and d1 is not None:
            f.write(f"static elem_t {buf_name}[{d1}][{d2}] row_align(1);\n")
    f.write(f"static const struct ConvParams {name}_params = {{")
    f.write(f".batch_size={p['batch_size']}, ")
    f.write(f".in_row_dim={p['in_row_dim']}, .in_col_dim={p['in_col_dim']}, ")
    f.write(f".kernel_size={p['kernel_size']}, ")
    f.write(f".in_channels={p['in_channels']}, .out_channels={p['out_channels']}, ")
    f.write(f".stride={p['stride']}, .padding={p['padding']}, ")
    f.write(f".bias={p['bias']}, .depthwise={p['depthwise']}, ")
    f.write(f".out_row_dim={p['out_row_dim']}, .out_col_dim={p['out_col_dim']}, ")
    f.write(f".n_patches={p['n_patches']}, .patch_size={p['patch_size']}, ")
    f.write(f".pool_size={p['pool_size']}, .pool_stride={p['pool_stride']}, ")
    f.write(f".pool_padding={p['pool_padding']}, .out_dim_pooled={p['out_dim_pooled']}, ")
    f.write(f".output_scale={output_scale_str}, ")
    f.write(f".I={p['I']}, .J={p['J']}, .K={p['K']}, ")
    f.write(f".res_scale={res_scale:.8e}f")
    f.write("};\n\n")


def write_fc_layer(f, entry):
    name             = entry["name"]
    w_int            = entry["weight"]   # [2048][1000]
    b_int            = entry["bias"]     # [BATCH_SIZE][1000]
    output_scale_str = entry["output_scale_str"]
    p                = FC_PARAMS[name]
    in_f             = p["in_features"]
    out_f            = p["out_features"]

    # fc_54_w[in_features][out_features] for tiled_matmul_nn_auto(A=average[4][2048], B=fc_54_w)
    f.write(f"static const elem_t {name}_w[{in_f}][{out_f}] row_align(1) = ")
    f.write(fmt_int_2d(w_int))
    f.write(";\n")
    f.write(f"static const acc_t {name}_b[{BATCH_SIZE}][{out_f}] row_align_acc(1) = ")
    f.write(fmt_int_2d(b_int))
    f.write(";\n")
    f.write(f"static elem_t {name}_out[{BATCH_SIZE}][{out_f}] row_align(1);\n")
    f.write(f"static const struct FcParams {name}_params = {{")
    f.write(f".batch_size={BATCH_SIZE}, ")
    f.write(f".in_features={in_f}, .out_features={out_f}, ")
    f.write(f".bias={p['bias']}, ")
    f.write(f".output_scale={output_scale_str}, ")
    f.write(f".I={p['I']}, .J={p['J']}, .K={p['K']}")
    f.write("};\n\n")


def write_header(output_path, layers_data):
    with open(output_path, "w") as f:
        f.write("// Generated by extract_resnet50_imagenet_int8.py\n")
        f.write("// Model: microsoft/resnet-50 (HuggingFace)\n")
        f.write("// Quantization: per-layer symmetric INT8, exact float output_scale\n")
        f.write("// Input preprocessing: (pixel/255 - mean)/std -> int8  (x_scale_0 = FMAX/127 = 2.75/127)\n\n")
        f.write("#ifndef RESNET50_IMAGENET_PARAMETERS_H\n")
        f.write("#define RESNET50_IMAGENET_PARAMETERS_H\n\n")
        f.write("#include <include/gemmini_params.h>\n")
        f.write("#include <stdbool.h>\n\n")

        for entry in layers_data:
            if entry["layer_type"] == "fc":
                write_fc_layer(f, entry)
            else:
                write_conv_layer(f, entry)

        f.write("#endif // RESNET50_IMAGENET_PARAMETERS_H\n")
    print(f"  Written: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract INT8-quantized ResNet-50 weights for Gemmini")
    parser.add_argument("--calibrate-dir", type=str, default=None,
                        help="Directory of JPEG images for calibration-based "
                             "activation ranges (highly recommended)")
    parser.add_argument("--num-calibrate", type=int, default=100,
                        help="Number of calibration images (default 100)")
    parser.add_argument("--pixel-minus-128", action="store_true",
                        help="Fold ImageNet normalization into first conv layer "
                             "so that raw pixel-128 INT8 input works (no "
                             "need to repreprocess images)")
    args = parser.parse_args()

    print(f"Loading {MODEL_NAME} from HuggingFace...")
    model = ResNetForImageClassification.from_pretrained(MODEL_NAME)
    model.eval()
    state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    num_classes = state_dict["classifier.1.weight"].shape[0]
    print(f"Model has {num_classes} output classes")
    assert num_classes == NUM_CLASSES, f"Expected {NUM_CLASSES} classes, got {num_classes}"

    # Input scale: depends on preprocessing mode
    if args.pixel_minus_128:
        # pixel-128 mode: input INT8 = pixel_uint8 - 128, range [-128, 127]
        # Normalization (mean/std) will be folded into conv_1 weights.
        x_scale_0 = 128.0 / 127.0
        print(f"Input preprocessing: pixel-128 (normalization folded into conv_1)")
        print(f"  x_scale_0 = 128/127 = {x_scale_0:.6f}")
    else:
        # Standard: x_float = (pixel/255 - mean)/std, quantized to INT8 with FMAX
        x_scale_0 = FMAX / 127.0
        print(f"Input preprocessing: (pixel/255-mean)/std/FMAX*127  ->  x_scale_0 = {x_scale_0:.6f}")

    # -----------------------------------------------------------------------
    # Pre-pass: compute y_range for every conv layer.
    # Prefer calibration (running actual images) over BN 3-sigma estimate.
    # BN estimates can be off by 2-5x for some layers, causing catastrophic
    # clipping that drops INT8 accuracy by ~20%.
    # -----------------------------------------------------------------------
    layer_y_range = {}

    if args.calibrate_dir:
        print("\nCalibration pass: measuring actual activation ranges...")
        calibrated = run_calibration(model, args.calibrate_dir, args.num_calibrate)
        # Use calibrated values; fall back to BN estimate for any missing layers
        for gemmini_name, hf_prefix, layer_type in LAYER_MAPPING:
            if layer_type == "fc":
                continue
            if gemmini_name in calibrated:
                layer_y_range[gemmini_name] = max(calibrated[gemmini_name], 1.0)
            else:
                _, _, bn_gamma, bn_beta = extract_and_fold(state_dict, hf_prefix)
                has_relu = LAYER_ACTIVATION.get(gemmini_name, False)
                layer_y_range[gemmini_name] = estimate_activation_range(bn_gamma, bn_beta, has_relu)
                print(f"  WARNING: {gemmini_name} not calibrated, using BN estimate")
    else:
        print("\nPre-pass: estimating activation ranges from BatchNorm statistics...")
        print("  WARNING: BN estimates can be inaccurate. Use --calibrate-dir for better accuracy.")
        for gemmini_name, hf_prefix, layer_type in LAYER_MAPPING:
            if layer_type == "fc":
                continue
            _, _, bn_gamma, bn_beta = extract_and_fold(state_dict, hf_prefix)
            has_relu = LAYER_ACTIVATION.get(gemmini_name, False)
            layer_y_range[gemmini_name] = estimate_activation_range(bn_gamma, bn_beta, has_relu)

    # -----------------------------------------------------------------------
    # Main quantization loop
    # -----------------------------------------------------------------------
    print(f"\nExtracting and quantizing {len(LAYER_MAPPING)} layers...")
    layers_data = []
    x_scale = x_scale_0

    # Projection shortcuts (conv_5, conv_15, conv_28, conv_47) share the same
    # block input as the first main-path conv of their stage, so they must use
    # the x_scale that was current BEFORE that first conv was processed.
    # SAVE_XSCALE_BEFORE maps "first-main-conv" -> "shortcut-name".
    SAVE_XSCALE_BEFORE = {
        "conv_2":  "conv_5",
        "conv_12": "conv_15",
        "conv_25": "conv_28",
        "conv_44": "conv_47",
    }
    shortcut_x_scale = {}  # shortcut_name -> correct x_scale

    for gemmini_name, hf_prefix, layer_type in LAYER_MAPPING:
        # Save x_scale for the corresponding projection shortcut before this
        # layer updates x_scale via the main bottleneck path.
        if gemmini_name in SAVE_XSCALE_BEFORE:
            shortcut_x_scale[SAVE_XSCALE_BEFORE[gemmini_name]] = x_scale
        # Projection shortcuts use the block's input x_scale (saved above).
        # All other layers use the current sequential x_scale.
        effective_x_scale = shortcut_x_scale.pop(gemmini_name, x_scale)

        if layer_type == "fc":
            fc_w_float = state_dict["classifier.1.weight"].float().numpy()  # [1000, 2048]
            fc_b_float = state_dict["classifier.1.bias"].float().numpy()    # [1000]

            # Gemmini layout: fc_54_w[K][J] = [2048][1000] (transposed from PyTorch)
            # tiled_matmul_nn_auto(I=4, J=1000, K=2048, A=average[4][2048], B=fc_54_w[2048][1000])
            w_gemmini = fc_w_float.T  # [2048, 1000]
            w_int, w_scale = quantize_weight_int8(w_gemmini)

            b_int_1d = quantize_bias_int32(fc_b_float, w_scale * x_scale)
            # Bias: replicated across batch dim -> [BATCH_SIZE][1000]
            b_int_2d = np.tile(b_int_1d.reshape(1, -1), (BATCH_SIZE, 1))

            fc_y_range = max(
                float(np.max(np.abs(fc_b_float))
                      + np.std(fc_w_float) * np.sqrt(float(fc_w_float.shape[1])) * x_scale * 3),
                5.0
            )
            _os, output_scale_str = compute_output_scale(w_scale, x_scale, fc_y_range)
            layer_y_range[gemmini_name] = fc_y_range

            print(f"  {gemmini_name:12s}  w_scale={w_scale:.6f}  x_scale={x_scale:.6f}  "
                  f"y_range={fc_y_range:.2f}  output_scale={_os:.4e}")

            layers_data.append({
                "name": gemmini_name,
                "layer_type": "fc",
                "weight": w_int,
                "bias": b_int_2d,
                "output_scale_str": output_scale_str,
            })

        else:
            if gemmini_name == "conv_1" and args.pixel_minus_128:
                # For pixel-128 input mode: fold ImageNet normalization into
                # the first conv layer's BN-folded weights.
                #
                # Model expects: x_norm = (pixel/255 - mean)/std
                # We have:       x_int8 = pixel - 128
                # So: x_norm = (x_int8 + 128)/(255*std) - mean/std
                #            = x_int8/(255*std) + (128/255 - mean)/std
                #
                # Folding into weights (before reshape to Gemmini layout):
                #   W_new[o,c,h,w] = W_BNfolded[o,c,h,w] / (255 * std[c])
                #   b_new[o] = b_BNfolded[o] + sum_c (128/255 - mean[c])/std[c] * sum_hw W_BNfolded[o,c,h,w]
                conv_w, gamma, beta, bn_m, bn_v = get_conv_bn_params(state_dict, hf_prefix)
                w_bn, b_bn = fold_bn(conv_w, gamma, beta, bn_m, bn_v)
                # w_bn shape: [out_ch, 3, kH, kW]
                mean_arr = np.array(IMAGENET_MEAN, dtype=np.float64)
                std_arr  = np.array(IMAGENET_STD, dtype=np.float64)
                norm_offset = (128.0 / 255.0 - mean_arr) / std_arr  # [3]
                # Add normalization offset to bias
                for c in range(3):
                    b_bn += w_bn[:, c, :, :].sum(axis=(1, 2)) * norm_offset[c]
                # Scale weights by 1/(255*std[c]) per input channel
                for c in range(3):
                    w_bn[:, c, :, :] /= (255.0 * std_arr[c])
                w_float = reshape_conv_weight(w_bn)
                b_float = b_bn
                bn_gamma = gamma
                bn_beta = beta
                print(f"  [conv_1] Folded ImageNet normalization into weights (pixel-128 mode)")
            else:
                w_float, b_float, bn_gamma, bn_beta = extract_and_fold(state_dict, hf_prefix)
            has_relu = LAYER_ACTIVATION.get(gemmini_name, False)

            w_int, w_scale = quantize_weight_int8(w_float)

            b_int = quantize_bias_int32(b_float, w_scale * effective_x_scale)

            y_range = layer_y_range[gemmini_name]  # already computed in pre-pass
            _os, output_scale_str = compute_output_scale(w_scale, effective_x_scale, y_range)

            # res_scale: rescales the skip tensor to match this layer's output scale.
            # res_scale = y_range[skip_src] / y_range[this layer]
            # Used in: tiled_resadd_auto(..., res_scale, MVIN_SCALE_IDENTITY, ..., skip_out, this_out, ...)
            res_scale = 1.0
            if gemmini_name in RESIDUAL_SKIP:
                skip_src = RESIDUAL_SKIP[gemmini_name]
                if skip_src in layer_y_range:
                    res_scale = layer_y_range[skip_src] / y_range
                else:
                    print(f"  WARNING: skip source {skip_src} not found for {gemmini_name}")

            print(f"  {gemmini_name:12s}  w_scale={w_scale:.6f}  x_scale={effective_x_scale:.6f}  "
                  f"y_range={y_range:.2f}  output_scale={_os:.4e}  "
                  f"res_scale={res_scale:.4f}  {'RELU' if has_relu else 'LINEAR'}")

            layers_data.append({
                "name": gemmini_name,
                "layer_type": "conv",
                "weight": w_int,
                "bias": b_int,
                "output_scale_str": output_scale_str,
                "res_scale": res_scale,
            })

            # Propagate x_scale: next layer's input scale = this layer's output scale.
            # BUT: shortcut/projection layers (conv_5, conv_15, conv_28, conv_47)
            # feed into resadd, NOT into the next sequential layer. The resadd
            # output is quantized at the main path's scale, so we must NOT
            # overwrite x_scale after a shortcut layer.
            is_shortcut = gemmini_name in set(SAVE_XSCALE_BEFORE.values())
            if not is_shortcut:
                x_scale = y_range / 127.0

    # -----------------------------------------------------------------------
    # Write output header
    # -----------------------------------------------------------------------
    output_dir  = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.normpath(os.path.join(output_dir, "..", "resnet50_params.h"))
    print(f"\nWriting {output_path} ...")
    write_header(output_path, layers_data)

    total_params = sum(e["weight"].size + e["bias"].size for e in layers_data)
    print(f"Total parameters: {total_params:,}")
    print("Done.")


if __name__ == "__main__":
    main()
