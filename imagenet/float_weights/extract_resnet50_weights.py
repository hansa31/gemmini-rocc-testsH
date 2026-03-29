#!/usr/bin/env python3
"""
Extract FP32 weights from HuggingFace ResNet-50 (microsoft/resnet-50)
and generate resnet50_params_float.h matching the structure of resnet50_params.h.

BatchNorm is folded into conv weights and biases for every convolutional layer.
The FC classifier layer has no BatchNorm and is extracted directly.

Usage:
    pip install torch transformers
    python extract_resnet50_weights.py

Output:
    ../resnet50_params_float.h
"""

import os
import sys
import numpy as np

import torch
from transformers import ResNetForImageClassification


# ---------------------------------------------------------------------------
# HuggingFace-to-Gemmini layer mapping
# ---------------------------------------------------------------------------
# ResNet-50 architecture (bottleneck blocks):
#   Stage 0 (layer1): 3 blocks × (1x1, 3x3, 1x1)  — 64→64→256 channels
#   Stage 1 (layer2): 4 blocks × (1x1, 3x3, 1x1)  — 128→128→512 channels
#   Stage 2 (layer3): 6 blocks × (1x1, 3x3, 1x1)  — 256→256→1024 channels
#   Stage 3 (layer4): 3 blocks × (1x1, 3x3, 1x1)  — 512→512→2048 channels
#
# Each block's first element has a shortcut (projection) conv; the rest use
# identity shortcuts (no extra conv needed).
#
# HuggingFace key pattern:
#   Conv+BN:  resnet.encoder.stages.{s}.layers.{l}.layer.{k}.convolution.weight
#             resnet.encoder.stages.{s}.layers.{l}.layer.{k}.normalization.{weight,bias,running_mean,running_var}
#   Shortcut: resnet.encoder.stages.{s}.layers.{l}.shortcut.convolution.weight + normalization
#   Init conv: resnet.embedder.embedder.convolution.weight + normalization
#   FC:        classifier.1.weight / classifier.1.bias

LAYER_MAPPING = [
    # (gemmini_name, hf_prefix, layer_type)
    # layer_type: 'conv' for all conv+BN layers, 'fc' for classifier

    # Initial 7×7 conv + BN
    ("conv_1",  "resnet.embedder.embedder",                         "conv"),

    # Stage 0 (layer1) — 3 bottleneck blocks
    # Block 0: uses a projection shortcut (conv_5)
    ("conv_2",  "resnet.encoder.stages.0.layers.0.layer.0",         "conv"),
    ("conv_3",  "resnet.encoder.stages.0.layers.0.layer.1",         "conv"),
    ("conv_4",  "resnet.encoder.stages.0.layers.0.layer.2",         "conv"),
    ("conv_5",  "resnet.encoder.stages.0.layers.0.shortcut",        "conv"),  # projection skip

    # Block 1
    ("conv_6",  "resnet.encoder.stages.0.layers.1.layer.0",         "conv"),
    ("conv_7",  "resnet.encoder.stages.0.layers.1.layer.1",         "conv"),
    ("conv_8",  "resnet.encoder.stages.0.layers.1.layer.2",         "conv"),

    # Block 2
    ("conv_9",  "resnet.encoder.stages.0.layers.2.layer.0",         "conv"),
    ("conv_10", "resnet.encoder.stages.0.layers.2.layer.1",         "conv"),
    ("conv_11", "resnet.encoder.stages.0.layers.2.layer.2",         "conv"),

    # Stage 1 (layer2) — 4 bottleneck blocks
    # Block 0: uses a projection shortcut (conv_15) with stride 2
    ("conv_12", "resnet.encoder.stages.1.layers.0.layer.0",         "conv"),
    ("conv_13", "resnet.encoder.stages.1.layers.0.layer.1",         "conv"),  # 3x3, stride 2
    ("conv_14", "resnet.encoder.stages.1.layers.0.layer.2",         "conv"),
    ("conv_15", "resnet.encoder.stages.1.layers.0.shortcut",        "conv"),  # projection skip

    # Block 1
    ("conv_16", "resnet.encoder.stages.1.layers.1.layer.0",         "conv"),
    ("conv_17", "resnet.encoder.stages.1.layers.1.layer.1",         "conv"),
    ("conv_18", "resnet.encoder.stages.1.layers.1.layer.2",         "conv"),

    # Block 2
    ("conv_19", "resnet.encoder.stages.1.layers.2.layer.0",         "conv"),
    ("conv_20", "resnet.encoder.stages.1.layers.2.layer.1",         "conv"),
    ("conv_21", "resnet.encoder.stages.1.layers.2.layer.2",         "conv"),

    # Block 3
    ("conv_22", "resnet.encoder.stages.1.layers.3.layer.0",         "conv"),
    ("conv_23", "resnet.encoder.stages.1.layers.3.layer.1",         "conv"),
    ("conv_24", "resnet.encoder.stages.1.layers.3.layer.2",         "conv"),

    # Stage 2 (layer3) — 6 bottleneck blocks
    # Block 0: projection shortcut (conv_28) with stride 2
    ("conv_25", "resnet.encoder.stages.2.layers.0.layer.0",         "conv"),
    ("conv_26", "resnet.encoder.stages.2.layers.0.layer.1",         "conv"),  # 3x3, stride 2
    ("conv_27", "resnet.encoder.stages.2.layers.0.layer.2",         "conv"),
    ("conv_28", "resnet.encoder.stages.2.layers.0.shortcut",        "conv"),  # projection skip

    # Block 1
    ("conv_29", "resnet.encoder.stages.2.layers.1.layer.0",         "conv"),
    ("conv_30", "resnet.encoder.stages.2.layers.1.layer.1",         "conv"),
    ("conv_31", "resnet.encoder.stages.2.layers.1.layer.2",         "conv"),

    # Block 2
    ("conv_32", "resnet.encoder.stages.2.layers.2.layer.0",         "conv"),
    ("conv_33", "resnet.encoder.stages.2.layers.2.layer.1",         "conv"),
    ("conv_34", "resnet.encoder.stages.2.layers.2.layer.2",         "conv"),

    # Block 3
    ("conv_35", "resnet.encoder.stages.2.layers.3.layer.0",         "conv"),
    ("conv_36", "resnet.encoder.stages.2.layers.3.layer.1",         "conv"),
    ("conv_37", "resnet.encoder.stages.2.layers.3.layer.2",         "conv"),

    # Block 4
    ("conv_38", "resnet.encoder.stages.2.layers.4.layer.0",         "conv"),
    ("conv_39", "resnet.encoder.stages.2.layers.4.layer.1",         "conv"),
    ("conv_40", "resnet.encoder.stages.2.layers.4.layer.2",         "conv"),

    # Block 5
    ("conv_41", "resnet.encoder.stages.2.layers.5.layer.0",         "conv"),
    ("conv_42", "resnet.encoder.stages.2.layers.5.layer.1",         "conv"),
    ("conv_43", "resnet.encoder.stages.2.layers.5.layer.2",         "conv"),

    # Stage 3 (layer4) — 3 bottleneck blocks
    # Block 0: projection shortcut (conv_47) with stride 2
    ("conv_44", "resnet.encoder.stages.3.layers.0.layer.0",         "conv"),
    ("conv_45", "resnet.encoder.stages.3.layers.0.layer.1",         "conv"),  # 3x3, stride 2
    ("conv_46", "resnet.encoder.stages.3.layers.0.layer.2",         "conv"),
    ("conv_47", "resnet.encoder.stages.3.layers.0.shortcut",        "conv"),  # projection skip

    # Block 1
    ("conv_48", "resnet.encoder.stages.3.layers.1.layer.0",         "conv"),
    ("conv_49", "resnet.encoder.stages.3.layers.1.layer.1",         "conv"),
    ("conv_50", "resnet.encoder.stages.3.layers.1.layer.2",         "conv"),

    # Block 2
    ("conv_51", "resnet.encoder.stages.3.layers.2.layer.0",         "conv"),
    ("conv_52", "resnet.encoder.stages.3.layers.2.layer.1",         "conv"),
    ("conv_53", "resnet.encoder.stages.3.layers.2.layer.2",         "conv"),

    # Global average pool → FC classifier
    ("fc_54",   "classifier",                                        "fc"),
]


# ---------------------------------------------------------------------------
# Layer parameter structs (mirrors resnet50_params.h)
# ---------------------------------------------------------------------------
# Parameters copied exactly from resnet50_params.h; output_scale and res_scale
# are set to 1.0f for float inference (no fixed-point scaling required).

# fmt: off
CONV_PARAMS = {
    "conv_1":  {"batch_size": 4, "in_row_dim": 224, "in_col_dim": 224, "kernel_size": 7,  "in_channels": 3,    "out_channels": 64,   "stride": 2, "padding": 3, "bias": 1, "depthwise": 0, "out_row_dim": 112, "out_col_dim": 112, "n_patches": 50176, "patch_size": 147,  "pool_size": 3, "pool_stride": 2, "pool_padding": 1, "out_dim_pooled": 56,  "I": 50176, "J": 64,   "K": 147,  "res_scale": "1.0f"},
    "conv_2":  {"batch_size": 4, "in_row_dim": 56,  "in_col_dim": 56,  "kernel_size": 1,  "in_channels": 64,   "out_channels": 64,   "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 56,  "out_col_dim": 56,  "n_patches": 12544, "patch_size": 64,   "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 56,  "I": 12544, "J": 64,   "K": 64,   "res_scale": "1.0f"},
    "conv_3":  {"batch_size": 4, "in_row_dim": 56,  "in_col_dim": 56,  "kernel_size": 3,  "in_channels": 64,   "out_channels": 64,   "stride": 1, "padding": 1, "bias": 1, "depthwise": 0, "out_row_dim": 56,  "out_col_dim": 56,  "n_patches": 12544, "patch_size": 576,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 56,  "I": 12544, "J": 64,   "K": 576,  "res_scale": "1.0f"},
    "conv_4":  {"batch_size": 4, "in_row_dim": 56,  "in_col_dim": 56,  "kernel_size": 1,  "in_channels": 64,   "out_channels": 256,  "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 56,  "out_col_dim": 56,  "n_patches": 12544, "patch_size": 64,   "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 56,  "I": 12544, "J": 256,  "K": 64,   "res_scale": "1.0f"},
    "conv_5":  {"batch_size": 4, "in_row_dim": 56,  "in_col_dim": 56,  "kernel_size": 1,  "in_channels": 64,   "out_channels": 256,  "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 56,  "out_col_dim": 56,  "n_patches": 12544, "patch_size": 64,   "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 56,  "I": 12544, "J": 256,  "K": 64,   "res_scale": "1.0f"},
    "conv_6":  {"batch_size": 4, "in_row_dim": 56,  "in_col_dim": 56,  "kernel_size": 1,  "in_channels": 256,  "out_channels": 64,   "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 56,  "out_col_dim": 56,  "n_patches": 12544, "patch_size": 256,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 56,  "I": 12544, "J": 64,   "K": 256,  "res_scale": "1.0f"},
    "conv_7":  {"batch_size": 4, "in_row_dim": 56,  "in_col_dim": 56,  "kernel_size": 3,  "in_channels": 64,   "out_channels": 64,   "stride": 1, "padding": 1, "bias": 1, "depthwise": 0, "out_row_dim": 56,  "out_col_dim": 56,  "n_patches": 12544, "patch_size": 576,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 56,  "I": 12544, "J": 64,   "K": 576,  "res_scale": "1.0f"},
    "conv_8":  {"batch_size": 4, "in_row_dim": 56,  "in_col_dim": 56,  "kernel_size": 1,  "in_channels": 64,   "out_channels": 256,  "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 56,  "out_col_dim": 56,  "n_patches": 12544, "patch_size": 64,   "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 56,  "I": 12544, "J": 256,  "K": 64,   "res_scale": "1.0f"},
    "conv_9":  {"batch_size": 4, "in_row_dim": 56,  "in_col_dim": 56,  "kernel_size": 1,  "in_channels": 256,  "out_channels": 64,   "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 56,  "out_col_dim": 56,  "n_patches": 12544, "patch_size": 256,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 56,  "I": 12544, "J": 64,   "K": 256,  "res_scale": "1.0f"},
    "conv_10": {"batch_size": 4, "in_row_dim": 56,  "in_col_dim": 56,  "kernel_size": 3,  "in_channels": 64,   "out_channels": 64,   "stride": 1, "padding": 1, "bias": 1, "depthwise": 0, "out_row_dim": 56,  "out_col_dim": 56,  "n_patches": 12544, "patch_size": 576,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 56,  "I": 12544, "J": 64,   "K": 576,  "res_scale": "1.0f"},
    "conv_11": {"batch_size": 4, "in_row_dim": 56,  "in_col_dim": 56,  "kernel_size": 1,  "in_channels": 64,   "out_channels": 256,  "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 56,  "out_col_dim": 56,  "n_patches": 12544, "patch_size": 64,   "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 56,  "I": 12544, "J": 256,  "K": 64,   "res_scale": "1.0f"},
    "conv_12": {"batch_size": 4, "in_row_dim": 56,  "in_col_dim": 56,  "kernel_size": 1,  "in_channels": 256,  "out_channels": 128,  "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 56,  "out_col_dim": 56,  "n_patches": 12544, "patch_size": 256,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 56,  "I": 12544, "J": 128,  "K": 256,  "res_scale": "1.0f"},
    "conv_13": {"batch_size": 4, "in_row_dim": 56,  "in_col_dim": 56,  "kernel_size": 3,  "in_channels": 128,  "out_channels": 128,  "stride": 2, "padding": 1, "bias": 1, "depthwise": 0, "out_row_dim": 28,  "out_col_dim": 28,  "n_patches": 3136,  "patch_size": 1152, "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 28,  "I": 3136,  "J": 128,  "K": 1152, "res_scale": "1.0f"},
    "conv_14": {"batch_size": 4, "in_row_dim": 28,  "in_col_dim": 28,  "kernel_size": 1,  "in_channels": 128,  "out_channels": 512,  "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 28,  "out_col_dim": 28,  "n_patches": 3136,  "patch_size": 128,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 28,  "I": 3136,  "J": 512,  "K": 128,  "res_scale": "1.0f"},
    "conv_15": {"batch_size": 4, "in_row_dim": 56,  "in_col_dim": 56,  "kernel_size": 1,  "in_channels": 256,  "out_channels": 512,  "stride": 2, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 28,  "out_col_dim": 28,  "n_patches": 3136,  "patch_size": 256,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 28,  "I": 3136,  "J": 512,  "K": 256,  "res_scale": "1.0f"},
    "conv_16": {"batch_size": 4, "in_row_dim": 28,  "in_col_dim": 28,  "kernel_size": 1,  "in_channels": 512,  "out_channels": 128,  "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 28,  "out_col_dim": 28,  "n_patches": 3136,  "patch_size": 512,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 28,  "I": 3136,  "J": 128,  "K": 512,  "res_scale": "1.0f"},
    "conv_17": {"batch_size": 4, "in_row_dim": 28,  "in_col_dim": 28,  "kernel_size": 3,  "in_channels": 128,  "out_channels": 128,  "stride": 1, "padding": 1, "bias": 1, "depthwise": 0, "out_row_dim": 28,  "out_col_dim": 28,  "n_patches": 3136,  "patch_size": 1152, "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 28,  "I": 3136,  "J": 128,  "K": 1152, "res_scale": "1.0f"},
    "conv_18": {"batch_size": 4, "in_row_dim": 28,  "in_col_dim": 28,  "kernel_size": 1,  "in_channels": 128,  "out_channels": 512,  "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 28,  "out_col_dim": 28,  "n_patches": 3136,  "patch_size": 128,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 28,  "I": 3136,  "J": 512,  "K": 128,  "res_scale": "1.0f"},
    "conv_19": {"batch_size": 4, "in_row_dim": 28,  "in_col_dim": 28,  "kernel_size": 1,  "in_channels": 512,  "out_channels": 128,  "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 28,  "out_col_dim": 28,  "n_patches": 3136,  "patch_size": 512,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 28,  "I": 3136,  "J": 128,  "K": 512,  "res_scale": "1.0f"},
    "conv_20": {"batch_size": 4, "in_row_dim": 28,  "in_col_dim": 28,  "kernel_size": 3,  "in_channels": 128,  "out_channels": 128,  "stride": 1, "padding": 1, "bias": 1, "depthwise": 0, "out_row_dim": 28,  "out_col_dim": 28,  "n_patches": 3136,  "patch_size": 1152, "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 28,  "I": 3136,  "J": 128,  "K": 1152, "res_scale": "1.0f"},
    "conv_21": {"batch_size": 4, "in_row_dim": 28,  "in_col_dim": 28,  "kernel_size": 1,  "in_channels": 128,  "out_channels": 512,  "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 28,  "out_col_dim": 28,  "n_patches": 3136,  "patch_size": 128,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 28,  "I": 3136,  "J": 512,  "K": 128,  "res_scale": "1.0f"},
    "conv_22": {"batch_size": 4, "in_row_dim": 28,  "in_col_dim": 28,  "kernel_size": 1,  "in_channels": 512,  "out_channels": 128,  "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 28,  "out_col_dim": 28,  "n_patches": 3136,  "patch_size": 512,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 28,  "I": 3136,  "J": 128,  "K": 512,  "res_scale": "1.0f"},
    "conv_23": {"batch_size": 4, "in_row_dim": 28,  "in_col_dim": 28,  "kernel_size": 3,  "in_channels": 128,  "out_channels": 128,  "stride": 1, "padding": 1, "bias": 1, "depthwise": 0, "out_row_dim": 28,  "out_col_dim": 28,  "n_patches": 3136,  "patch_size": 1152, "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 28,  "I": 3136,  "J": 128,  "K": 1152, "res_scale": "1.0f"},
    "conv_24": {"batch_size": 4, "in_row_dim": 28,  "in_col_dim": 28,  "kernel_size": 1,  "in_channels": 128,  "out_channels": 512,  "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 28,  "out_col_dim": 28,  "n_patches": 3136,  "patch_size": 128,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 28,  "I": 3136,  "J": 512,  "K": 128,  "res_scale": "1.0f"},
    "conv_25": {"batch_size": 4, "in_row_dim": 28,  "in_col_dim": 28,  "kernel_size": 1,  "in_channels": 512,  "out_channels": 256,  "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 28,  "out_col_dim": 28,  "n_patches": 3136,  "patch_size": 512,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 28,  "I": 3136,  "J": 256,  "K": 512,  "res_scale": "1.0f"},
    "conv_26": {"batch_size": 4, "in_row_dim": 28,  "in_col_dim": 28,  "kernel_size": 3,  "in_channels": 256,  "out_channels": 256,  "stride": 2, "padding": 1, "bias": 1, "depthwise": 0, "out_row_dim": 14,  "out_col_dim": 14,  "n_patches": 784,   "patch_size": 2304, "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 14,  "I": 784,   "J": 256,  "K": 2304, "res_scale": "1.0f"},
    "conv_27": {"batch_size": 4, "in_row_dim": 14,  "in_col_dim": 14,  "kernel_size": 1,  "in_channels": 256,  "out_channels": 1024, "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 14,  "out_col_dim": 14,  "n_patches": 784,   "patch_size": 256,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 14,  "I": 784,   "J": 1024, "K": 256,  "res_scale": "1.0f"},
    "conv_28": {"batch_size": 4, "in_row_dim": 28,  "in_col_dim": 28,  "kernel_size": 1,  "in_channels": 512,  "out_channels": 1024, "stride": 2, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 14,  "out_col_dim": 14,  "n_patches": 784,   "patch_size": 512,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 14,  "I": 784,   "J": 1024, "K": 512,  "res_scale": "1.0f"},
    "conv_29": {"batch_size": 4, "in_row_dim": 14,  "in_col_dim": 14,  "kernel_size": 1,  "in_channels": 1024, "out_channels": 256,  "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 14,  "out_col_dim": 14,  "n_patches": 784,   "patch_size": 1024, "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 14,  "I": 784,   "J": 256,  "K": 1024, "res_scale": "1.0f"},
    "conv_30": {"batch_size": 4, "in_row_dim": 14,  "in_col_dim": 14,  "kernel_size": 3,  "in_channels": 256,  "out_channels": 256,  "stride": 1, "padding": 1, "bias": 1, "depthwise": 0, "out_row_dim": 14,  "out_col_dim": 14,  "n_patches": 784,   "patch_size": 2304, "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 14,  "I": 784,   "J": 256,  "K": 2304, "res_scale": "1.0f"},
    "conv_31": {"batch_size": 4, "in_row_dim": 14,  "in_col_dim": 14,  "kernel_size": 1,  "in_channels": 256,  "out_channels": 1024, "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 14,  "out_col_dim": 14,  "n_patches": 784,   "patch_size": 256,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 14,  "I": 784,   "J": 1024, "K": 256,  "res_scale": "1.0f"},
    "conv_32": {"batch_size": 4, "in_row_dim": 14,  "in_col_dim": 14,  "kernel_size": 1,  "in_channels": 1024, "out_channels": 256,  "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 14,  "out_col_dim": 14,  "n_patches": 784,   "patch_size": 1024, "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 14,  "I": 784,   "J": 256,  "K": 1024, "res_scale": "1.0f"},
    "conv_33": {"batch_size": 4, "in_row_dim": 14,  "in_col_dim": 14,  "kernel_size": 3,  "in_channels": 256,  "out_channels": 256,  "stride": 1, "padding": 1, "bias": 1, "depthwise": 0, "out_row_dim": 14,  "out_col_dim": 14,  "n_patches": 784,   "patch_size": 2304, "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 14,  "I": 784,   "J": 256,  "K": 2304, "res_scale": "1.0f"},
    "conv_34": {"batch_size": 4, "in_row_dim": 14,  "in_col_dim": 14,  "kernel_size": 1,  "in_channels": 256,  "out_channels": 1024, "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 14,  "out_col_dim": 14,  "n_patches": 784,   "patch_size": 256,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 14,  "I": 784,   "J": 1024, "K": 256,  "res_scale": "1.0f"},
    "conv_35": {"batch_size": 4, "in_row_dim": 14,  "in_col_dim": 14,  "kernel_size": 1,  "in_channels": 1024, "out_channels": 256,  "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 14,  "out_col_dim": 14,  "n_patches": 784,   "patch_size": 1024, "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 14,  "I": 784,   "J": 256,  "K": 1024, "res_scale": "1.0f"},
    "conv_36": {"batch_size": 4, "in_row_dim": 14,  "in_col_dim": 14,  "kernel_size": 3,  "in_channels": 256,  "out_channels": 256,  "stride": 1, "padding": 1, "bias": 1, "depthwise": 0, "out_row_dim": 14,  "out_col_dim": 14,  "n_patches": 784,   "patch_size": 2304, "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 14,  "I": 784,   "J": 256,  "K": 2304, "res_scale": "1.0f"},
    "conv_37": {"batch_size": 4, "in_row_dim": 14,  "in_col_dim": 14,  "kernel_size": 1,  "in_channels": 256,  "out_channels": 1024, "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 14,  "out_col_dim": 14,  "n_patches": 784,   "patch_size": 256,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 14,  "I": 784,   "J": 1024, "K": 256,  "res_scale": "1.0f"},
    "conv_38": {"batch_size": 4, "in_row_dim": 14,  "in_col_dim": 14,  "kernel_size": 1,  "in_channels": 1024, "out_channels": 256,  "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 14,  "out_col_dim": 14,  "n_patches": 784,   "patch_size": 1024, "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 14,  "I": 784,   "J": 256,  "K": 1024, "res_scale": "1.0f"},
    "conv_39": {"batch_size": 4, "in_row_dim": 14,  "in_col_dim": 14,  "kernel_size": 3,  "in_channels": 256,  "out_channels": 256,  "stride": 1, "padding": 1, "bias": 1, "depthwise": 0, "out_row_dim": 14,  "out_col_dim": 14,  "n_patches": 784,   "patch_size": 2304, "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 14,  "I": 784,   "J": 256,  "K": 2304, "res_scale": "1.0f"},
    "conv_40": {"batch_size": 4, "in_row_dim": 14,  "in_col_dim": 14,  "kernel_size": 1,  "in_channels": 256,  "out_channels": 1024, "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 14,  "out_col_dim": 14,  "n_patches": 784,   "patch_size": 256,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 14,  "I": 784,   "J": 1024, "K": 256,  "res_scale": "1.0f"},
    "conv_41": {"batch_size": 4, "in_row_dim": 14,  "in_col_dim": 14,  "kernel_size": 1,  "in_channels": 1024, "out_channels": 256,  "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 14,  "out_col_dim": 14,  "n_patches": 784,   "patch_size": 1024, "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 14,  "I": 784,   "J": 256,  "K": 1024, "res_scale": "1.0f"},
    "conv_42": {"batch_size": 4, "in_row_dim": 14,  "in_col_dim": 14,  "kernel_size": 3,  "in_channels": 256,  "out_channels": 256,  "stride": 1, "padding": 1, "bias": 1, "depthwise": 0, "out_row_dim": 14,  "out_col_dim": 14,  "n_patches": 784,   "patch_size": 2304, "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 14,  "I": 784,   "J": 256,  "K": 2304, "res_scale": "1.0f"},
    "conv_43": {"batch_size": 4, "in_row_dim": 14,  "in_col_dim": 14,  "kernel_size": 1,  "in_channels": 256,  "out_channels": 1024, "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 14,  "out_col_dim": 14,  "n_patches": 784,   "patch_size": 256,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 14,  "I": 784,   "J": 1024, "K": 256,  "res_scale": "1.0f"},
    "conv_44": {"batch_size": 4, "in_row_dim": 14,  "in_col_dim": 14,  "kernel_size": 1,  "in_channels": 1024, "out_channels": 512,  "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 14,  "out_col_dim": 14,  "n_patches": 784,   "patch_size": 1024, "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 14,  "I": 784,   "J": 512,  "K": 1024, "res_scale": "1.0f"},
    "conv_45": {"batch_size": 4, "in_row_dim": 14,  "in_col_dim": 14,  "kernel_size": 3,  "in_channels": 512,  "out_channels": 512,  "stride": 2, "padding": 1, "bias": 1, "depthwise": 0, "out_row_dim": 7,   "out_col_dim": 7,   "n_patches": 196,   "patch_size": 4608, "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 7,   "I": 196,   "J": 512,  "K": 4608, "res_scale": "1.0f"},
    "conv_46": {"batch_size": 4, "in_row_dim": 7,   "in_col_dim": 7,   "kernel_size": 1,  "in_channels": 512,  "out_channels": 2048, "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 7,   "out_col_dim": 7,   "n_patches": 196,   "patch_size": 512,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 7,   "I": 196,   "J": 2048, "K": 512,  "res_scale": "1.0f"},
    "conv_47": {"batch_size": 4, "in_row_dim": 14,  "in_col_dim": 14,  "kernel_size": 1,  "in_channels": 1024, "out_channels": 2048, "stride": 2, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 7,   "out_col_dim": 7,   "n_patches": 196,   "patch_size": 1024, "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 7,   "I": 196,   "J": 2048, "K": 1024, "res_scale": "1.0f"},
    "conv_48": {"batch_size": 4, "in_row_dim": 7,   "in_col_dim": 7,   "kernel_size": 1,  "in_channels": 2048, "out_channels": 512,  "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 7,   "out_col_dim": 7,   "n_patches": 196,   "patch_size": 2048, "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 7,   "I": 196,   "J": 512,  "K": 2048, "res_scale": "1.0f"},
    "conv_49": {"batch_size": 4, "in_row_dim": 7,   "in_col_dim": 7,   "kernel_size": 3,  "in_channels": 512,  "out_channels": 512,  "stride": 1, "padding": 1, "bias": 1, "depthwise": 0, "out_row_dim": 7,   "out_col_dim": 7,   "n_patches": 196,   "patch_size": 4608, "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 7,   "I": 196,   "J": 512,  "K": 4608, "res_scale": "1.0f"},
    "conv_50": {"batch_size": 4, "in_row_dim": 7,   "in_col_dim": 7,   "kernel_size": 1,  "in_channels": 512,  "out_channels": 2048, "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 7,   "out_col_dim": 7,   "n_patches": 196,   "patch_size": 512,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 7,   "I": 196,   "J": 2048, "K": 512,  "res_scale": "1.0f"},
    "conv_51": {"batch_size": 4, "in_row_dim": 7,   "in_col_dim": 7,   "kernel_size": 1,  "in_channels": 2048, "out_channels": 512,  "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 7,   "out_col_dim": 7,   "n_patches": 196,   "patch_size": 2048, "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 7,   "I": 196,   "J": 512,  "K": 2048, "res_scale": "1.0f"},
    "conv_52": {"batch_size": 4, "in_row_dim": 7,   "in_col_dim": 7,   "kernel_size": 3,  "in_channels": 512,  "out_channels": 512,  "stride": 1, "padding": 1, "bias": 1, "depthwise": 0, "out_row_dim": 7,   "out_col_dim": 7,   "n_patches": 196,   "patch_size": 4608, "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 7,   "I": 196,   "J": 512,  "K": 4608, "res_scale": "1.0f"},
    "conv_53": {"batch_size": 4, "in_row_dim": 7,   "in_col_dim": 7,   "kernel_size": 1,  "in_channels": 512,  "out_channels": 2048, "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 7,   "out_col_dim": 7,   "n_patches": 196,   "patch_size": 512,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 7,   "I": 196,   "J": 2048, "K": 512,  "res_scale": "1.0f"},
}

FC_PARAMS = {
    # I=batch_size, J=out_features, K=in_features
    # Weight layout: [K][J] = [2048][1000]  (transposed from PyTorch [1000, 2048])
    # Bias layout:   [batch_size][J] = [4][1000]
    "fc_54": {"batch_size": 4, "in_features": 2048, "out_features": 1000, "bias": 1, "I": 4, "J": 1000, "K": 2048},
}
# fmt: on

# Buffer declarations — mirrors the static arrays in resnet50_params.h.
# Tuples: (name, dim1, dim2) for 2-D buffers; special 4-D string for the pooled buffer.
BUFFERS = [
    # conv_1: input, output (pre-pool), pooled output
    ("conv_1_in",          50176, 147),
    ("conv_1_out",         50176, 64),
    ("conv_1_out_pooled",  None,  None),  # special: [4][56][56][64]

    # Stage 0 ---------------------------------------------------------------
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

    # Stage 1 ---------------------------------------------------------------
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

    # Stage 2 ---------------------------------------------------------------
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

    # Stage 3 ---------------------------------------------------------------
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

    # FC --------------------------------------------------------------------
    ("fc_54_out",   4,     1000),
]

BATCH_SIZE = 4


# ---------------------------------------------------------------------------
# BatchNorm folding
# ---------------------------------------------------------------------------

def fold_bn(conv_weight, bn_weight, bn_bias, bn_mean, bn_var, eps=1e-5):
    """Fold a BatchNorm layer into the preceding convolution's weight and bias.

    Args:
        conv_weight : [out_ch, in_ch, kH, kW]  (PyTorch layout)
        bn_weight   : gamma [out_ch]
        bn_bias     : beta  [out_ch]
        bn_mean     : running_mean [out_ch]
        bn_var      : running_var  [out_ch]
        eps         : BN epsilon

    Returns:
        (w_folded, b_folded) as float32 numpy arrays.
    """
    inv_std = 1.0 / np.sqrt(bn_var + eps)
    scale = bn_weight * inv_std                          # gamma / sqrt(var+eps)
    shape = [conv_weight.shape[0]] + [1] * (conv_weight.ndim - 1)
    w_folded = conv_weight * scale.reshape(shape)
    b_folded = bn_bias - bn_weight * bn_mean * inv_std
    return w_folded, b_folded


# ---------------------------------------------------------------------------
# Weight reshaping (PyTorch layout → Gemmini layout)
# ---------------------------------------------------------------------------

def reshape_conv_weight(w):
    """Regular conv: PyTorch [out_ch, in_ch, kH, kW] → Gemmini [patch_size, out_ch].

    patch_size = in_ch * kH * kW.
    """
    out_ch     = w.shape[0]
    patch_size = int(np.prod(w.shape[1:]))
    return w.reshape(out_ch, patch_size).T   # → [patch_size, out_ch]


# ---------------------------------------------------------------------------
# C code formatting helpers
# ---------------------------------------------------------------------------

def fmt_float(v):
    """Format a single float value as a C literal."""
    return f"{v:.8g}"


def fmt_1d(arr):
    """Format 1-D numpy array: {v0,v1,...}"""
    return "{" + ",".join(fmt_float(v) for v in arr.flat) + "}"


def fmt_2d(arr):
    """Format 2-D numpy array: {{r0},{r1},...}"""
    return "{" + ",".join(fmt_1d(arr[i]) for i in range(arr.shape[0])) + "}"


# ---------------------------------------------------------------------------
# Layer extraction from HuggingFace state dict
# ---------------------------------------------------------------------------

def get_conv_bn_params(state_dict, prefix):
    """Extract weight + BN parameters from a ResNet ConvLayer at *prefix*."""
    conv_w  = state_dict[f"{prefix}.convolution.weight"].float().numpy()
    gamma   = state_dict[f"{prefix}.normalization.weight"].float().numpy()
    beta    = state_dict[f"{prefix}.normalization.bias"].float().numpy()
    mean    = state_dict[f"{prefix}.normalization.running_mean"].float().numpy()
    var     = state_dict[f"{prefix}.normalization.running_var"].float().numpy()
    return conv_w, gamma, beta, mean, var


def extract_layer(state_dict, gemmini_name, hf_prefix, layer_type):
    """Extract and transform weights/biases for a single Gemmini layer.

    Returns (weight_array, bias_array) as float32 numpy arrays in Gemmini layout.
    """
    if layer_type == "fc":
        # Classifier: nn.Sequential(Dropout, Linear)
        fc_w = state_dict["classifier.1.weight"].float().numpy()   # [1000, 2048]
        fc_b = state_dict["classifier.1.bias"].float().numpy()     # [1000]

        # Gemmini layout: fc_54_w[K][J] = [2048][1000]  (transpose of PyTorch)
        w_gemmini = fc_w.T                                          # [2048, 1000]
        # Bias replicated across batch dim: [batch_size][1000]
        b_gemmini = np.tile(fc_b.reshape(1, -1), (BATCH_SIZE, 1))  # [4, 1000]
        return w_gemmini, b_gemmini

    # All ResNet conv layers include a BatchNorm → fold it
    conv_w, gamma, beta, mean, var = get_conv_bn_params(state_dict, hf_prefix)
    w_folded, b_folded = fold_bn(conv_w, gamma, beta, mean, var)

    # Reshape to Gemmini [patch_size][out_ch] layout
    w_gemmini = reshape_conv_weight(w_folded)
    return w_gemmini, b_folded


# ---------------------------------------------------------------------------
# Shape validation
# ---------------------------------------------------------------------------

def validate_shapes(gemmini_name, weight, bias, layer_type):
    """Assert that extracted arrays match the expected Gemmini buffer shapes."""
    if layer_type == "fc":
        assert weight.shape == (2048, 1000), \
            f"{gemmini_name}: fc weight shape {weight.shape} != (2048, 1000)"
        assert bias.shape == (BATCH_SIZE, 1000), \
            f"{gemmini_name}: fc bias shape {bias.shape} != ({BATCH_SIZE}, 1000)"
        return

    p        = CONV_PARAMS[gemmini_name]
    patch_sz = p["patch_size"]
    out_ch   = p["out_channels"]

    assert weight.shape == (patch_sz, out_ch), \
        f"{gemmini_name}: weight shape {weight.shape} != ({patch_sz}, {out_ch})"
    assert bias.shape == (out_ch,), \
        f"{gemmini_name}: bias shape {bias.shape} != ({out_ch},)"


# ---------------------------------------------------------------------------
# Header file generation
# ---------------------------------------------------------------------------

def write_header(output_path, layers_data):
    with open(output_path, "w") as f:
        f.write("#ifndef RESNET50_FLOAT_PARAMETERS_H\n")
        f.write("#define RESNET50_FLOAT_PARAMETERS_H\n\n")
        f.write("#include <include/gemmini_params.h>\n")
        f.write("#include <stdbool.h>\n\n")

        for gemmini_name, weight, bias, layer_type in layers_data:
            write_layer(f, gemmini_name, weight, bias, layer_type)

        f.write("#endif // RESNET50_FLOAT_PARAMETERS_H\n")


def write_layer(f, name, weight, bias, layer_type):
    if layer_type == "fc":
        write_fc_layer(f, name, weight, bias)
    else:
        write_conv_layer(f, name, weight, bias)
    f.write("\n\n")


def write_conv_layer(f, name, weight, bias):
    p         = CONV_PARAMS[name]
    patch_sz  = p["patch_size"]
    out_ch    = p["out_channels"]

    # Weight: [patch_size][out_ch]
    f.write(f"static const elem_t {name}_w[{patch_sz}][{out_ch}] row_align(1) = ")
    f.write(fmt_2d(weight))
    f.write(";\n")

    # Bias: [out_ch]
    f.write(f"static const acc_t {name}_b[{out_ch}] row_align_acc(1) = ")
    f.write(fmt_1d(bias))
    f.write(";\n")

    # Mutable buffers for this layer
    for buf in BUFFERS:
        buf_name = buf[0]
        if buf_name == "conv_1_out_pooled":
            if name == "conv_1":
                f.write("static elem_t conv_1_out_pooled[4][56][56][64];\n")
            continue
        if buf_name.startswith(name + "_") and buf[1] is not None:
            f.write(f"static elem_t {buf_name}[{buf[1]}][{buf[2]}] row_align(1);\n")

    # Params struct
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
    f.write(f".output_scale=1.0f, ")
    f.write(f".I={p['I']}, .J={p['J']}, .K={p['K']}, ")
    f.write(f".res_scale={p['res_scale']}")
    f.write("};\n")


def write_fc_layer(f, name, weight, bias):
    p      = FC_PARAMS[name]
    in_f   = p["in_features"]
    out_f  = p["out_features"]
    bs     = p["batch_size"]

    # Weight: [in_features][out_features]  (K × J layout for tiled_matmul)
    f.write(f"static const elem_t {name}_w[{in_f}][{out_f}] row_align(1) = ")
    f.write(fmt_2d(weight))
    f.write(";\n")

    # Bias: [batch_size][out_features]
    f.write(f"static const acc_t {name}_b[{bs}][{out_f}] row_align_acc(1) = ")
    f.write(fmt_2d(bias))
    f.write(";\n")

    # Output buffer
    f.write(f"static elem_t {name}_out[{bs}][{out_f}] row_align(1);\n")

    # Params struct
    f.write(f"static const struct FcParams {name}_params = {{")
    f.write(f".batch_size={bs}, ")
    f.write(f".in_features={in_f}, .out_features={out_f}, ")
    f.write(f".bias={p['bias']}, ")
    f.write(f".output_scale=1.0f, ")
    f.write(f".I={p['I']}, .J={p['J']}, .K={p['K']}")
    f.write("};\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("Loading microsoft/resnet-50 from HuggingFace ...")
    model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
    model.eval()
    state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    num_classes = state_dict["classifier.1.weight"].shape[0]
    print(f"Classifier output classes: {num_classes}")
    if num_classes != 1000:
        print(f"WARNING: expected 1000 classes, got {num_classes}", file=sys.stderr)

    print("\nState dict keys:")
    for k in sorted(state_dict.keys()):
        print(f"  {k}: {list(state_dict[k].shape)}")

    mapping = LAYER_MAPPING
    layers_data = []

    print(f"\nExtracting {len(mapping)} layers ...")
    for gemmini_name, hf_prefix, layer_type in mapping:
        print(f"  {gemmini_name:10s} <- {hf_prefix}")
        weight, bias = extract_layer(state_dict, gemmini_name, hf_prefix, layer_type)
        validate_shapes(gemmini_name, weight, bias, layer_type)
        layers_data.append((gemmini_name, weight, bias, layer_type))

    output_dir  = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.normpath(os.path.join(output_dir, "..", "resnet50_params_float.h"))

    print(f"\nWriting {output_path} ...")
    write_header(output_path, layers_data)
    print("Done!")

    total_params = sum(w.size + b.size for _, w, b, _ in layers_data)
    print(f"\nTotal floating-point parameters written: {total_params:,}")


if __name__ == "__main__":
    main()
