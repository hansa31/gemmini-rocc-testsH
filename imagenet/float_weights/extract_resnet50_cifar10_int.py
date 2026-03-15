#!/usr/bin/env python3
"""
Extract INT8-quantized weights from HuggingFace ResNet-50 fine-tuned on
CIFAR-10 (edadaltocg/resnet50_cifar10) and generate:
  - ../resnet50_cifar10_params.h  (INT8 weights, INT32 biases, power-of-2 output_scale)

The CIFAR-10 ResNet-50 uses a modified stem:
  - First conv: 3x3, stride=1, padding=1 (vs 7x7, stride=2 for ImageNet)
  - No initial 3x3 max-pool
This keeps the spatial dimension at 32x32 through all of Stage 0.

Quantization approach:
  - Per-layer symmetric weight quantization: w_int8 = clip(round(w_float / scale), -128, 127)
  - INT32 biases: b_int32 = round(b_float / (w_scale * x_scale))
  - Power-of-2 output_scale: output_scale = w_scale * x_scale / y_scale
  - Activation range estimated from BatchNorm running statistics (gamma, beta)
  - Scale propagation tracks x_scale through the network layer-by-layer

Usage:
    pip install torch transformers numpy
    python extract_resnet50_cifar10_int.py
"""

import os
import math
import numpy as np
import torch
from transformers import ResNetForImageClassification

MODEL_NAME = "edadaltocg/resnet50_cifar10"
INPUT_DIM = 32
BATCH_SIZE = 4
NUM_CLASSES = 10

# ---------------------------------------------------------------------------
# HuggingFace layer mapping (same key structure as microsoft/resnet-50)
# ---------------------------------------------------------------------------
# (gemmini_name, hf_prefix, layer_type)
# layer_type: 'conv' for conv+BN, 'fc' for FC classifier
LAYER_MAPPING = [
    # Initial 3x3 conv + BN (CIFAR-10 modification: 3x3 stride 1, no maxpool)
    ("conv_1",  "resnet.embedder.embedder",                          "conv"),

    # Stage 0 (layer1) — 3 bottleneck blocks (64->64->256)
    # Block 0: projection shortcut conv_5
    ("conv_2",  "resnet.encoder.stages.0.layers.0.layer.0",          "conv"),
    ("conv_3",  "resnet.encoder.stages.0.layers.0.layer.1",          "conv"),  # 3x3
    ("conv_4",  "resnet.encoder.stages.0.layers.0.layer.2",          "conv"),
    ("conv_5",  "resnet.encoder.stages.0.layers.0.shortcut",         "conv"),  # projection
    # Block 1: identity shortcut
    ("conv_6",  "resnet.encoder.stages.0.layers.1.layer.0",          "conv"),
    ("conv_7",  "resnet.encoder.stages.0.layers.1.layer.1",          "conv"),  # 3x3
    ("conv_8",  "resnet.encoder.stages.0.layers.1.layer.2",          "conv"),
    # Block 2: identity shortcut
    ("conv_9",  "resnet.encoder.stages.0.layers.2.layer.0",          "conv"),
    ("conv_10", "resnet.encoder.stages.0.layers.2.layer.1",          "conv"),  # 3x3
    ("conv_11", "resnet.encoder.stages.0.layers.2.layer.2",          "conv"),

    # Stage 1 (layer2) — 4 bottleneck blocks (128->128->512)
    # Block 0: stride-2 3x3 conv (conv_13), projection shortcut conv_15
    ("conv_12", "resnet.encoder.stages.1.layers.0.layer.0",          "conv"),
    ("conv_13", "resnet.encoder.stages.1.layers.0.layer.1",          "conv"),  # 3x3, stride 2
    ("conv_14", "resnet.encoder.stages.1.layers.0.layer.2",          "conv"),
    ("conv_15", "resnet.encoder.stages.1.layers.0.shortcut",         "conv"),  # projection, stride 2
    # Block 1
    ("conv_16", "resnet.encoder.stages.1.layers.1.layer.0",          "conv"),
    ("conv_17", "resnet.encoder.stages.1.layers.1.layer.1",          "conv"),  # 3x3
    ("conv_18", "resnet.encoder.stages.1.layers.1.layer.2",          "conv"),
    # Block 2
    ("conv_19", "resnet.encoder.stages.1.layers.2.layer.0",          "conv"),
    ("conv_20", "resnet.encoder.stages.1.layers.2.layer.1",          "conv"),  # 3x3
    ("conv_21", "resnet.encoder.stages.1.layers.2.layer.2",          "conv"),
    # Block 3
    ("conv_22", "resnet.encoder.stages.1.layers.3.layer.0",          "conv"),
    ("conv_23", "resnet.encoder.stages.1.layers.3.layer.1",          "conv"),  # 3x3
    ("conv_24", "resnet.encoder.stages.1.layers.3.layer.2",          "conv"),

    # Stage 2 (layer3) — 6 bottleneck blocks (256->256->1024)
    # Block 0: stride-2 3x3 conv (conv_26), projection shortcut conv_28
    ("conv_25", "resnet.encoder.stages.2.layers.0.layer.0",          "conv"),
    ("conv_26", "resnet.encoder.stages.2.layers.0.layer.1",          "conv"),  # 3x3, stride 2
    ("conv_27", "resnet.encoder.stages.2.layers.0.layer.2",          "conv"),
    ("conv_28", "resnet.encoder.stages.2.layers.0.shortcut",         "conv"),  # projection, stride 2
    # Block 1
    ("conv_29", "resnet.encoder.stages.2.layers.1.layer.0",          "conv"),
    ("conv_30", "resnet.encoder.stages.2.layers.1.layer.1",          "conv"),  # 3x3
    ("conv_31", "resnet.encoder.stages.2.layers.1.layer.2",          "conv"),
    # Block 2
    ("conv_32", "resnet.encoder.stages.2.layers.2.layer.0",          "conv"),
    ("conv_33", "resnet.encoder.stages.2.layers.2.layer.1",          "conv"),  # 3x3
    ("conv_34", "resnet.encoder.stages.2.layers.2.layer.2",          "conv"),
    # Block 3
    ("conv_35", "resnet.encoder.stages.2.layers.3.layer.0",          "conv"),
    ("conv_36", "resnet.encoder.stages.2.layers.3.layer.1",          "conv"),  # 3x3
    ("conv_37", "resnet.encoder.stages.2.layers.3.layer.2",          "conv"),
    # Block 4
    ("conv_38", "resnet.encoder.stages.2.layers.4.layer.0",          "conv"),
    ("conv_39", "resnet.encoder.stages.2.layers.4.layer.1",          "conv"),  # 3x3
    ("conv_40", "resnet.encoder.stages.2.layers.4.layer.2",          "conv"),
    # Block 5
    ("conv_41", "resnet.encoder.stages.2.layers.5.layer.0",          "conv"),
    ("conv_42", "resnet.encoder.stages.2.layers.5.layer.1",          "conv"),  # 3x3
    ("conv_43", "resnet.encoder.stages.2.layers.5.layer.2",          "conv"),

    # Stage 3 (layer4) — 3 bottleneck blocks (512->512->2048)
    # Block 0: stride-2 3x3 conv (conv_45), projection shortcut conv_47
    ("conv_44", "resnet.encoder.stages.3.layers.0.layer.0",          "conv"),
    ("conv_45", "resnet.encoder.stages.3.layers.0.layer.1",          "conv"),  # 3x3, stride 2
    ("conv_46", "resnet.encoder.stages.3.layers.0.layer.2",          "conv"),
    ("conv_47", "resnet.encoder.stages.3.layers.0.shortcut",         "conv"),  # projection, stride 2
    # Block 1
    ("conv_48", "resnet.encoder.stages.3.layers.1.layer.0",          "conv"),
    ("conv_49", "resnet.encoder.stages.3.layers.1.layer.1",          "conv"),  # 3x3
    ("conv_50", "resnet.encoder.stages.3.layers.1.layer.2",          "conv"),
    # Block 2
    ("conv_51", "resnet.encoder.stages.3.layers.2.layer.0",          "conv"),
    ("conv_52", "resnet.encoder.stages.3.layers.2.layer.1",          "conv"),  # 3x3
    ("conv_53", "resnet.encoder.stages.3.layers.2.layer.2",          "conv"),

    # Global average pool -> FC classifier (2048->10)
    ("fc_54",   "classifier",                                         "fc"),
]

# ---------------------------------------------------------------------------
# CIFAR-10 spatial parameter table
# ---------------------------------------------------------------------------
# Each entry: (gemmini_name, kernel, in_ch, out_ch, stride, padding, activation)
# All layers are standard (no depthwise) — ResNet50 has no depthwise convolutions.
# Spatial dims derived from 32x32 input with 3x3/s1 first conv (no maxpool).
#   Stage 0: 32x32 (n_patches=4096)
#   Stage 1: 32x32->16x16 (n_patches=4096 for inputs to block0, 1024 for outputs)
#   Stage 2: 16x16->8x8   (n_patches=1024 for inputs to block0, 256 for outputs)
#   Stage 3: 8x8->4x4     (n_patches=256  for inputs to block0, 64  for outputs)

# fmt: off
CONV_PARAMS = {
    # ---- Stem ----
    #        bs  in_r in_c  k   ic   oc   s  p  dw out_r out_c n_p   ps     pool_s pool_st pool_p pooled  I     J    K    res
    "conv_1":  {"batch_size": 4, "in_row_dim": 32, "in_col_dim": 32, "kernel_size": 3,  "in_channels": 3,    "out_channels": 64,   "stride": 1, "padding": 1, "bias": 1, "depthwise": 0, "out_row_dim": 32,  "out_col_dim": 32,  "n_patches": 4096, "patch_size": 27,   "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 32,  "I": 4096, "J": 64,   "K": 27,   "res_scale": "(1.0 / (1 << 0))"},

    # ---- Stage 0 (spatial: 32x32, n_patches=4096) ----
    "conv_2":  {"batch_size": 4, "in_row_dim": 32,  "in_col_dim": 32,  "kernel_size": 1, "in_channels": 64,   "out_channels": 64,   "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 32,  "out_col_dim": 32,  "n_patches": 4096, "patch_size": 64,   "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 32,  "I": 4096, "J": 64,   "K": 64,   "res_scale": "(1.0 / (1 << 0))"},
    "conv_3":  {"batch_size": 4, "in_row_dim": 32,  "in_col_dim": 32,  "kernel_size": 3, "in_channels": 64,   "out_channels": 64,   "stride": 1, "padding": 1, "bias": 1, "depthwise": 0, "out_row_dim": 32,  "out_col_dim": 32,  "n_patches": 4096, "patch_size": 576,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 32,  "I": 4096, "J": 64,   "K": 576,  "res_scale": "(1.0 / (1 << 0))"},
    "conv_4":  {"batch_size": 4, "in_row_dim": 32,  "in_col_dim": 32,  "kernel_size": 1, "in_channels": 64,   "out_channels": 256,  "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 32,  "out_col_dim": 32,  "n_patches": 4096, "patch_size": 64,   "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 32,  "I": 4096, "J": 256,  "K": 64,   "res_scale": "(1.0 / (1 << 0))"},
    "conv_5":  {"batch_size": 4, "in_row_dim": 32,  "in_col_dim": 32,  "kernel_size": 1, "in_channels": 64,   "out_channels": 256,  "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 32,  "out_col_dim": 32,  "n_patches": 4096, "patch_size": 64,   "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 32,  "I": 4096, "J": 256,  "K": 64,   "res_scale": "(1.0 / (1 << 0))"},
    "conv_6":  {"batch_size": 4, "in_row_dim": 32,  "in_col_dim": 32,  "kernel_size": 1, "in_channels": 256,  "out_channels": 64,   "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 32,  "out_col_dim": 32,  "n_patches": 4096, "patch_size": 256,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 32,  "I": 4096, "J": 64,   "K": 256,  "res_scale": "(1.0 / (1 << 0))"},
    "conv_7":  {"batch_size": 4, "in_row_dim": 32,  "in_col_dim": 32,  "kernel_size": 3, "in_channels": 64,   "out_channels": 64,   "stride": 1, "padding": 1, "bias": 1, "depthwise": 0, "out_row_dim": 32,  "out_col_dim": 32,  "n_patches": 4096, "patch_size": 576,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 32,  "I": 4096, "J": 64,   "K": 576,  "res_scale": "(1.0 / (1 << 0))"},
    "conv_8":  {"batch_size": 4, "in_row_dim": 32,  "in_col_dim": 32,  "kernel_size": 1, "in_channels": 64,   "out_channels": 256,  "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 32,  "out_col_dim": 32,  "n_patches": 4096, "patch_size": 64,   "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 32,  "I": 4096, "J": 256,  "K": 64,   "res_scale": "(1.0 / (1 << 0))"},
    "conv_9":  {"batch_size": 4, "in_row_dim": 32,  "in_col_dim": 32,  "kernel_size": 1, "in_channels": 256,  "out_channels": 64,   "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 32,  "out_col_dim": 32,  "n_patches": 4096, "patch_size": 256,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 32,  "I": 4096, "J": 64,   "K": 256,  "res_scale": "(1.0 / (1 << 0))"},
    "conv_10": {"batch_size": 4, "in_row_dim": 32,  "in_col_dim": 32,  "kernel_size": 3, "in_channels": 64,   "out_channels": 64,   "stride": 1, "padding": 1, "bias": 1, "depthwise": 0, "out_row_dim": 32,  "out_col_dim": 32,  "n_patches": 4096, "patch_size": 576,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 32,  "I": 4096, "J": 64,   "K": 576,  "res_scale": "(1.0 / (1 << 0))"},
    "conv_11": {"batch_size": 4, "in_row_dim": 32,  "in_col_dim": 32,  "kernel_size": 1, "in_channels": 64,   "out_channels": 256,  "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 32,  "out_col_dim": 32,  "n_patches": 4096, "patch_size": 64,   "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 32,  "I": 4096, "J": 256,  "K": 64,   "res_scale": "(1.0 / (1 << 0))"},

    # ---- Stage 1: Block 0 (in: 32x32, out: 16x16) ----
    "conv_12": {"batch_size": 4, "in_row_dim": 32,  "in_col_dim": 32,  "kernel_size": 1, "in_channels": 256,  "out_channels": 128,  "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 32,  "out_col_dim": 32,  "n_patches": 4096, "patch_size": 256,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 32,  "I": 4096, "J": 128,  "K": 256,  "res_scale": "(1.0 / (1 << 0))"},
    "conv_13": {"batch_size": 4, "in_row_dim": 32,  "in_col_dim": 32,  "kernel_size": 3, "in_channels": 128,  "out_channels": 128,  "stride": 2, "padding": 1, "bias": 1, "depthwise": 0, "out_row_dim": 16,  "out_col_dim": 16,  "n_patches": 1024, "patch_size": 1152, "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 16,  "I": 1024, "J": 128,  "K": 1152, "res_scale": "(1.0 / (1 << 0))"},
    "conv_14": {"batch_size": 4, "in_row_dim": 16,  "in_col_dim": 16,  "kernel_size": 1, "in_channels": 128,  "out_channels": 512,  "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 16,  "out_col_dim": 16,  "n_patches": 1024, "patch_size": 128,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 16,  "I": 1024, "J": 512,  "K": 128,  "res_scale": "(1.0 / (1 << 0))"},
    "conv_15": {"batch_size": 4, "in_row_dim": 32,  "in_col_dim": 32,  "kernel_size": 1, "in_channels": 256,  "out_channels": 512,  "stride": 2, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 16,  "out_col_dim": 16,  "n_patches": 1024, "patch_size": 256,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 16,  "I": 1024, "J": 512,  "K": 256,  "res_scale": "(1.0 / (1 << 0))"},
    # ---- Stage 1: Blocks 1-3 (16x16, n_patches=1024) ----
    "conv_16": {"batch_size": 4, "in_row_dim": 16,  "in_col_dim": 16,  "kernel_size": 1, "in_channels": 512,  "out_channels": 128,  "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 16,  "out_col_dim": 16,  "n_patches": 1024, "patch_size": 512,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 16,  "I": 1024, "J": 128,  "K": 512,  "res_scale": "(1.0 / (1 << 0))"},
    "conv_17": {"batch_size": 4, "in_row_dim": 16,  "in_col_dim": 16,  "kernel_size": 3, "in_channels": 128,  "out_channels": 128,  "stride": 1, "padding": 1, "bias": 1, "depthwise": 0, "out_row_dim": 16,  "out_col_dim": 16,  "n_patches": 1024, "patch_size": 1152, "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 16,  "I": 1024, "J": 128,  "K": 1152, "res_scale": "(1.0 / (1 << 0))"},
    "conv_18": {"batch_size": 4, "in_row_dim": 16,  "in_col_dim": 16,  "kernel_size": 1, "in_channels": 128,  "out_channels": 512,  "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 16,  "out_col_dim": 16,  "n_patches": 1024, "patch_size": 128,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 16,  "I": 1024, "J": 512,  "K": 128,  "res_scale": "(1.0 / (1 << 0))"},
    "conv_19": {"batch_size": 4, "in_row_dim": 16,  "in_col_dim": 16,  "kernel_size": 1, "in_channels": 512,  "out_channels": 128,  "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 16,  "out_col_dim": 16,  "n_patches": 1024, "patch_size": 512,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 16,  "I": 1024, "J": 128,  "K": 512,  "res_scale": "(1.0 / (1 << 0))"},
    "conv_20": {"batch_size": 4, "in_row_dim": 16,  "in_col_dim": 16,  "kernel_size": 3, "in_channels": 128,  "out_channels": 128,  "stride": 1, "padding": 1, "bias": 1, "depthwise": 0, "out_row_dim": 16,  "out_col_dim": 16,  "n_patches": 1024, "patch_size": 1152, "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 16,  "I": 1024, "J": 128,  "K": 1152, "res_scale": "(1.0 / (1 << 0))"},
    "conv_21": {"batch_size": 4, "in_row_dim": 16,  "in_col_dim": 16,  "kernel_size": 1, "in_channels": 128,  "out_channels": 512,  "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 16,  "out_col_dim": 16,  "n_patches": 1024, "patch_size": 128,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 16,  "I": 1024, "J": 512,  "K": 128,  "res_scale": "(1.0 / (1 << 0))"},
    "conv_22": {"batch_size": 4, "in_row_dim": 16,  "in_col_dim": 16,  "kernel_size": 1, "in_channels": 512,  "out_channels": 128,  "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 16,  "out_col_dim": 16,  "n_patches": 1024, "patch_size": 512,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 16,  "I": 1024, "J": 128,  "K": 512,  "res_scale": "(1.0 / (1 << 0))"},
    "conv_23": {"batch_size": 4, "in_row_dim": 16,  "in_col_dim": 16,  "kernel_size": 3, "in_channels": 128,  "out_channels": 128,  "stride": 1, "padding": 1, "bias": 1, "depthwise": 0, "out_row_dim": 16,  "out_col_dim": 16,  "n_patches": 1024, "patch_size": 1152, "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 16,  "I": 1024, "J": 128,  "K": 1152, "res_scale": "(1.0 / (1 << 0))"},
    "conv_24": {"batch_size": 4, "in_row_dim": 16,  "in_col_dim": 16,  "kernel_size": 1, "in_channels": 128,  "out_channels": 512,  "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 16,  "out_col_dim": 16,  "n_patches": 1024, "patch_size": 128,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 16,  "I": 1024, "J": 512,  "K": 128,  "res_scale": "(1.0 / (1 << 0))"},

    # ---- Stage 2: Block 0 (in: 16x16, out: 8x8) ----
    "conv_25": {"batch_size": 4, "in_row_dim": 16,  "in_col_dim": 16,  "kernel_size": 1, "in_channels": 512,  "out_channels": 256,  "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 16,  "out_col_dim": 16,  "n_patches": 1024, "patch_size": 512,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 16,  "I": 1024, "J": 256,  "K": 512,  "res_scale": "(1.0 / (1 << 0))"},
    "conv_26": {"batch_size": 4, "in_row_dim": 16,  "in_col_dim": 16,  "kernel_size": 3, "in_channels": 256,  "out_channels": 256,  "stride": 2, "padding": 1, "bias": 1, "depthwise": 0, "out_row_dim": 8,   "out_col_dim": 8,   "n_patches": 256,  "patch_size": 2304, "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 8,   "I": 256,  "J": 256,  "K": 2304, "res_scale": "(1.0 / (1 << 0))"},
    "conv_27": {"batch_size": 4, "in_row_dim": 8,   "in_col_dim": 8,   "kernel_size": 1, "in_channels": 256,  "out_channels": 1024, "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 8,   "out_col_dim": 8,   "n_patches": 256,  "patch_size": 256,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 8,   "I": 256,  "J": 1024, "K": 256,  "res_scale": "(1.0 / (1 << 0))"},
    "conv_28": {"batch_size": 4, "in_row_dim": 16,  "in_col_dim": 16,  "kernel_size": 1, "in_channels": 512,  "out_channels": 1024, "stride": 2, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 8,   "out_col_dim": 8,   "n_patches": 256,  "patch_size": 512,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 8,   "I": 256,  "J": 1024, "K": 512,  "res_scale": "(1.0 / (1 << 0))"},
    # ---- Stage 2: Blocks 1-5 (8x8, n_patches=256) ----
    "conv_29": {"batch_size": 4, "in_row_dim": 8,   "in_col_dim": 8,   "kernel_size": 1, "in_channels": 1024, "out_channels": 256,  "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 8,   "out_col_dim": 8,   "n_patches": 256,  "patch_size": 1024, "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 8,   "I": 256,  "J": 256,  "K": 1024, "res_scale": "(1.0 / (1 << 0))"},
    "conv_30": {"batch_size": 4, "in_row_dim": 8,   "in_col_dim": 8,   "kernel_size": 3, "in_channels": 256,  "out_channels": 256,  "stride": 1, "padding": 1, "bias": 1, "depthwise": 0, "out_row_dim": 8,   "out_col_dim": 8,   "n_patches": 256,  "patch_size": 2304, "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 8,   "I": 256,  "J": 256,  "K": 2304, "res_scale": "(1.0 / (1 << 0))"},
    "conv_31": {"batch_size": 4, "in_row_dim": 8,   "in_col_dim": 8,   "kernel_size": 1, "in_channels": 256,  "out_channels": 1024, "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 8,   "out_col_dim": 8,   "n_patches": 256,  "patch_size": 256,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 8,   "I": 256,  "J": 1024, "K": 256,  "res_scale": "(1.0 / (1 << 0))"},
    "conv_32": {"batch_size": 4, "in_row_dim": 8,   "in_col_dim": 8,   "kernel_size": 1, "in_channels": 1024, "out_channels": 256,  "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 8,   "out_col_dim": 8,   "n_patches": 256,  "patch_size": 1024, "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 8,   "I": 256,  "J": 256,  "K": 1024, "res_scale": "(1.0 / (1 << 0))"},
    "conv_33": {"batch_size": 4, "in_row_dim": 8,   "in_col_dim": 8,   "kernel_size": 3, "in_channels": 256,  "out_channels": 256,  "stride": 1, "padding": 1, "bias": 1, "depthwise": 0, "out_row_dim": 8,   "out_col_dim": 8,   "n_patches": 256,  "patch_size": 2304, "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 8,   "I": 256,  "J": 256,  "K": 2304, "res_scale": "(1.0 / (1 << 0))"},
    "conv_34": {"batch_size": 4, "in_row_dim": 8,   "in_col_dim": 8,   "kernel_size": 1, "in_channels": 256,  "out_channels": 1024, "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 8,   "out_col_dim": 8,   "n_patches": 256,  "patch_size": 256,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 8,   "I": 256,  "J": 1024, "K": 256,  "res_scale": "(1.0 / (1 << 0))"},
    "conv_35": {"batch_size": 4, "in_row_dim": 8,   "in_col_dim": 8,   "kernel_size": 1, "in_channels": 1024, "out_channels": 256,  "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 8,   "out_col_dim": 8,   "n_patches": 256,  "patch_size": 1024, "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 8,   "I": 256,  "J": 256,  "K": 1024, "res_scale": "(1.0 / (1 << 0))"},
    "conv_36": {"batch_size": 4, "in_row_dim": 8,   "in_col_dim": 8,   "kernel_size": 3, "in_channels": 256,  "out_channels": 256,  "stride": 1, "padding": 1, "bias": 1, "depthwise": 0, "out_row_dim": 8,   "out_col_dim": 8,   "n_patches": 256,  "patch_size": 2304, "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 8,   "I": 256,  "J": 256,  "K": 2304, "res_scale": "(1.0 / (1 << 0))"},
    "conv_37": {"batch_size": 4, "in_row_dim": 8,   "in_col_dim": 8,   "kernel_size": 1, "in_channels": 256,  "out_channels": 1024, "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 8,   "out_col_dim": 8,   "n_patches": 256,  "patch_size": 256,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 8,   "I": 256,  "J": 1024, "K": 256,  "res_scale": "(1.0 / (1 << 0))"},
    "conv_38": {"batch_size": 4, "in_row_dim": 8,   "in_col_dim": 8,   "kernel_size": 1, "in_channels": 1024, "out_channels": 256,  "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 8,   "out_col_dim": 8,   "n_patches": 256,  "patch_size": 1024, "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 8,   "I": 256,  "J": 256,  "K": 1024, "res_scale": "(1.0 / (1 << 0))"},
    "conv_39": {"batch_size": 4, "in_row_dim": 8,   "in_col_dim": 8,   "kernel_size": 3, "in_channels": 256,  "out_channels": 256,  "stride": 1, "padding": 1, "bias": 1, "depthwise": 0, "out_row_dim": 8,   "out_col_dim": 8,   "n_patches": 256,  "patch_size": 2304, "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 8,   "I": 256,  "J": 256,  "K": 2304, "res_scale": "(1.0 / (1 << 0))"},
    "conv_40": {"batch_size": 4, "in_row_dim": 8,   "in_col_dim": 8,   "kernel_size": 1, "in_channels": 256,  "out_channels": 1024, "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 8,   "out_col_dim": 8,   "n_patches": 256,  "patch_size": 256,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 8,   "I": 256,  "J": 1024, "K": 256,  "res_scale": "(1.0 / (1 << 0))"},
    "conv_41": {"batch_size": 4, "in_row_dim": 8,   "in_col_dim": 8,   "kernel_size": 1, "in_channels": 1024, "out_channels": 256,  "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 8,   "out_col_dim": 8,   "n_patches": 256,  "patch_size": 1024, "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 8,   "I": 256,  "J": 256,  "K": 1024, "res_scale": "(1.0 / (1 << 0))"},
    "conv_42": {"batch_size": 4, "in_row_dim": 8,   "in_col_dim": 8,   "kernel_size": 3, "in_channels": 256,  "out_channels": 256,  "stride": 1, "padding": 1, "bias": 1, "depthwise": 0, "out_row_dim": 8,   "out_col_dim": 8,   "n_patches": 256,  "patch_size": 2304, "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 8,   "I": 256,  "J": 256,  "K": 2304, "res_scale": "(1.0 / (1 << 0))"},
    "conv_43": {"batch_size": 4, "in_row_dim": 8,   "in_col_dim": 8,   "kernel_size": 1, "in_channels": 256,  "out_channels": 1024, "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 8,   "out_col_dim": 8,   "n_patches": 256,  "patch_size": 256,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 8,   "I": 256,  "J": 1024, "K": 256,  "res_scale": "(1.0 / (1 << 0))"},

    # ---- Stage 3: Block 0 (in: 8x8, out: 4x4) ----
    "conv_44": {"batch_size": 4, "in_row_dim": 8,   "in_col_dim": 8,   "kernel_size": 1, "in_channels": 1024, "out_channels": 512,  "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 8,   "out_col_dim": 8,   "n_patches": 256,  "patch_size": 1024, "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 8,   "I": 256,  "J": 512,  "K": 1024, "res_scale": "(1.0 / (1 << 0))"},
    "conv_45": {"batch_size": 4, "in_row_dim": 8,   "in_col_dim": 8,   "kernel_size": 3, "in_channels": 512,  "out_channels": 512,  "stride": 2, "padding": 1, "bias": 1, "depthwise": 0, "out_row_dim": 4,   "out_col_dim": 4,   "n_patches": 64,   "patch_size": 4608, "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 4,   "I": 64,   "J": 512,  "K": 4608, "res_scale": "(1.0 / (1 << 0))"},
    "conv_46": {"batch_size": 4, "in_row_dim": 4,   "in_col_dim": 4,   "kernel_size": 1, "in_channels": 512,  "out_channels": 2048, "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 4,   "out_col_dim": 4,   "n_patches": 64,   "patch_size": 512,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 4,   "I": 64,   "J": 2048, "K": 512,  "res_scale": "(1.0 / (1 << 0))"},
    "conv_47": {"batch_size": 4, "in_row_dim": 8,   "in_col_dim": 8,   "kernel_size": 1, "in_channels": 1024, "out_channels": 2048, "stride": 2, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 4,   "out_col_dim": 4,   "n_patches": 64,   "patch_size": 1024, "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 4,   "I": 64,   "J": 2048, "K": 1024, "res_scale": "(1.0 / (1 << 0))"},
    # ---- Stage 3: Blocks 1-2 (4x4, n_patches=64) ----
    "conv_48": {"batch_size": 4, "in_row_dim": 4,   "in_col_dim": 4,   "kernel_size": 1, "in_channels": 2048, "out_channels": 512,  "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 4,   "out_col_dim": 4,   "n_patches": 64,   "patch_size": 2048, "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 4,   "I": 64,   "J": 512,  "K": 2048, "res_scale": "(1.0 / (1 << 0))"},
    "conv_49": {"batch_size": 4, "in_row_dim": 4,   "in_col_dim": 4,   "kernel_size": 3, "in_channels": 512,  "out_channels": 512,  "stride": 1, "padding": 1, "bias": 1, "depthwise": 0, "out_row_dim": 4,   "out_col_dim": 4,   "n_patches": 64,   "patch_size": 4608, "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 4,   "I": 64,   "J": 512,  "K": 4608, "res_scale": "(1.0 / (1 << 0))"},
    "conv_50": {"batch_size": 4, "in_row_dim": 4,   "in_col_dim": 4,   "kernel_size": 1, "in_channels": 512,  "out_channels": 2048, "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 4,   "out_col_dim": 4,   "n_patches": 64,   "patch_size": 512,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 4,   "I": 64,   "J": 2048, "K": 512,  "res_scale": "(1.0 / (1 << 0))"},
    "conv_51": {"batch_size": 4, "in_row_dim": 4,   "in_col_dim": 4,   "kernel_size": 1, "in_channels": 2048, "out_channels": 512,  "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 4,   "out_col_dim": 4,   "n_patches": 64,   "patch_size": 2048, "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 4,   "I": 64,   "J": 512,  "K": 2048, "res_scale": "(1.0 / (1 << 0))"},
    "conv_52": {"batch_size": 4, "in_row_dim": 4,   "in_col_dim": 4,   "kernel_size": 3, "in_channels": 512,  "out_channels": 512,  "stride": 1, "padding": 1, "bias": 1, "depthwise": 0, "out_row_dim": 4,   "out_col_dim": 4,   "n_patches": 64,   "patch_size": 4608, "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 4,   "I": 64,   "J": 512,  "K": 4608, "res_scale": "(1.0 / (1 << 0))"},
    "conv_53": {"batch_size": 4, "in_row_dim": 4,   "in_col_dim": 4,   "kernel_size": 1, "in_channels": 512,  "out_channels": 2048, "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 4,   "out_col_dim": 4,   "n_patches": 64,   "patch_size": 512,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 4,   "I": 64,   "J": 2048, "K": 512,  "res_scale": "(1.0 / (1 << 0))"},
}
# fmt: on

FC_PARAMS = {
    # fc_54_w layout: [in_features][out_features] = [2048][10]
    # fc_54_b layout: [out_features][batch_size]  = [10][4]
    # fc_54_out layout: [out_features][batch_size] = [10][4]
    "fc_54": {
        "batch_size": BATCH_SIZE,
        "in_features": 2048,
        "out_features": NUM_CLASSES,
        "bias": 1,
        "I": NUM_CLASSES,
        "J": BATCH_SIZE,
        "K": 2048,
    }
}

# Buffer declarations for resnet50_cifar10_params.h
# (name, dim1, dim2) for 2-D static arrays
BUFFERS = [
    # Stem
    ("conv_1_in",    4096,  27),    ("conv_1_out",   4096,  64),

    # Stage 0
    ("conv_2_in",    4096,  64),    ("conv_2_out",   4096,  64),
    ("conv_3_in",    4096,  576),   ("conv_3_out",   4096,  64),
    ("conv_4_in",    4096,  64),    ("conv_4_out",   4096,  256),
    ("conv_5_in",    4096,  64),    ("conv_5_out",   4096,  256),
    ("conv_6_in",    4096,  256),   ("conv_6_out",   4096,  64),
    ("conv_7_in",    4096,  576),   ("conv_7_out",   4096,  64),
    ("conv_8_in",    4096,  64),    ("conv_8_out",   4096,  256),
    ("conv_9_in",    4096,  256),   ("conv_9_out",   4096,  64),
    ("conv_10_in",   4096,  576),   ("conv_10_out",  4096,  64),
    ("conv_11_in",   4096,  64),    ("conv_11_out",  4096,  256),

    # Stage 1
    ("conv_12_in",   4096,  256),   ("conv_12_out",  4096,  128),
    ("conv_13_in",   1024,  1152),  ("conv_13_out",  1024,  128),
    ("conv_14_in",   1024,  128),   ("conv_14_out",  1024,  512),
    ("conv_15_in",   1024,  256),   ("conv_15_out",  1024,  512),
    ("conv_16_in",   1024,  512),   ("conv_16_out",  1024,  128),
    ("conv_17_in",   1024,  1152),  ("conv_17_out",  1024,  128),
    ("conv_18_in",   1024,  128),   ("conv_18_out",  1024,  512),
    ("conv_19_in",   1024,  512),   ("conv_19_out",  1024,  128),
    ("conv_20_in",   1024,  1152),  ("conv_20_out",  1024,  128),
    ("conv_21_in",   1024,  128),   ("conv_21_out",  1024,  512),
    ("conv_22_in",   1024,  512),   ("conv_22_out",  1024,  128),
    ("conv_23_in",   1024,  1152),  ("conv_23_out",  1024,  128),
    ("conv_24_in",   1024,  128),   ("conv_24_out",  1024,  512),

    # Stage 2
    ("conv_25_in",   1024,  512),   ("conv_25_out",  1024,  256),
    ("conv_26_in",   256,   2304),  ("conv_26_out",  256,   256),
    ("conv_27_in",   256,   256),   ("conv_27_out",  256,   1024),
    ("conv_28_in",   256,   512),   ("conv_28_out",  256,   1024),
    ("conv_29_in",   256,   1024),  ("conv_29_out",  256,   256),
    ("conv_30_in",   256,   2304),  ("conv_30_out",  256,   256),
    ("conv_31_in",   256,   256),   ("conv_31_out",  256,   1024),
    ("conv_32_in",   256,   1024),  ("conv_32_out",  256,   256),
    ("conv_33_in",   256,   2304),  ("conv_33_out",  256,   256),
    ("conv_34_in",   256,   256),   ("conv_34_out",  256,   1024),
    ("conv_35_in",   256,   1024),  ("conv_35_out",  256,   256),
    ("conv_36_in",   256,   2304),  ("conv_36_out",  256,   256),
    ("conv_37_in",   256,   256),   ("conv_37_out",  256,   1024),
    ("conv_38_in",   256,   1024),  ("conv_38_out",  256,   256),
    ("conv_39_in",   256,   2304),  ("conv_39_out",  256,   256),
    ("conv_40_in",   256,   256),   ("conv_40_out",  256,   1024),
    ("conv_41_in",   256,   1024),  ("conv_41_out",  256,   256),
    ("conv_42_in",   256,   2304),  ("conv_42_out",  256,   256),
    ("conv_43_in",   256,   256),   ("conv_43_out",  256,   1024),

    # Stage 3
    ("conv_44_in",   256,   1024),  ("conv_44_out",  256,   512),
    ("conv_45_in",   64,    4608),  ("conv_45_out",  64,    512),
    ("conv_46_in",   64,    512),   ("conv_46_out",  64,    2048),
    ("conv_47_in",   64,    1024),  ("conv_47_out",  64,    2048),
    ("conv_48_in",   64,    2048),  ("conv_48_out",  64,    512),
    ("conv_49_in",   64,    4608),  ("conv_49_out",  64,    512),
    ("conv_50_in",   64,    512),   ("conv_50_out",  64,    2048),
    ("conv_51_in",   64,    2048),  ("conv_51_out",  64,    512),
    ("conv_52_in",   64,    4608),  ("conv_52_out",  64,    512),
    ("conv_53_in",   64,    512),   ("conv_53_out",  64,    2048),

    # FC
    ("fc_54_out",    NUM_CLASSES, BATCH_SIZE),
]

# Activation type per layer (True = ReLU, False = linear/NO_ACTIVATION)
# ResNet-50 bottleneck: 1x1 ReLU, 3x3 ReLU, 1x1 Linear; all shortcuts are Linear
LAYER_ACTIVATION = {
    "conv_1":  True,
    # Stage 0
    "conv_2":  True,  "conv_3":  True,  "conv_4":  False, "conv_5":  False,
    "conv_6":  True,  "conv_7":  True,  "conv_8":  False,
    "conv_9":  True,  "conv_10": True,  "conv_11": False,
    # Stage 1
    "conv_12": True,  "conv_13": True,  "conv_14": False, "conv_15": False,
    "conv_16": True,  "conv_17": True,  "conv_18": False,
    "conv_19": True,  "conv_20": True,  "conv_21": False,
    "conv_22": True,  "conv_23": True,  "conv_24": False,
    # Stage 2
    "conv_25": True,  "conv_26": True,  "conv_27": False, "conv_28": False,
    "conv_29": True,  "conv_30": True,  "conv_31": False,
    "conv_32": True,  "conv_33": True,  "conv_34": False,
    "conv_35": True,  "conv_36": True,  "conv_37": False,
    "conv_38": True,  "conv_39": True,  "conv_40": False,
    "conv_41": True,  "conv_42": True,  "conv_43": False,
    # Stage 3
    "conv_44": True,  "conv_45": True,  "conv_46": False, "conv_47": False,
    "conv_48": True,  "conv_49": True,  "conv_50": False,
    "conv_51": True,  "conv_52": True,  "conv_53": False,
}


# ---------------------------------------------------------------------------
# BatchNorm folding
# ---------------------------------------------------------------------------

def fold_bn(conv_weight, bn_weight, bn_bias, bn_mean, bn_var, eps=1e-5):
    inv_std = 1.0 / np.sqrt(bn_var + eps)
    scale = bn_weight * inv_std
    shape = [conv_weight.shape[0]] + [1] * (conv_weight.ndim - 1)
    w_folded = conv_weight * scale.reshape(shape)
    b_folded = bn_bias - bn_weight * bn_mean * inv_std
    return w_folded, b_folded


# ---------------------------------------------------------------------------
# Weight reshaping: PyTorch [out_ch, in_ch, kH, kW] -> Gemmini [patch_size, out_ch]
# ---------------------------------------------------------------------------

def reshape_conv_weight(w):
    out_ch = w.shape[0]
    return w.reshape(out_ch, -1).T


# ---------------------------------------------------------------------------
# INT8 quantization
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
    b_int = np.clip(
        np.round(b_float / combined_scale), -(2**31), 2**31 - 1
    ).astype(np.int32)
    return b_int


def compute_output_scale_pow2(w_scale, x_scale, y_range):
    y_scale = y_range / 127.0
    raw = (w_scale * x_scale) / y_scale
    if raw <= 0 or not np.isfinite(raw):
        return 0, "(1.0 / (1 << 0))"
    log_val = -math.log2(raw)
    N = max(0, min(15, round(log_val)))
    return N, f"(1.0 / (1 << {N}))"


def estimate_activation_range(bn_gamma, bn_beta, has_relu):
    gamma_max = np.max(np.abs(bn_gamma))
    beta_max = np.max(np.abs(bn_beta))
    if has_relu:
        y_range = max(float(np.max(bn_beta + 3.0 * np.abs(bn_gamma))), 1.0)
    else:
        y_range = max(beta_max + 3.0 * gamma_max, 1.0)
    return y_range


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
# HuggingFace state dict helpers
# ---------------------------------------------------------------------------

def get_conv_bn_params(state_dict, prefix):
    conv_w = state_dict[f"{prefix}.convolution.weight"].numpy()
    bn_gamma = state_dict[f"{prefix}.normalization.weight"].numpy()
    bn_beta = state_dict[f"{prefix}.normalization.bias"].numpy()
    bn_mean = state_dict[f"{prefix}.normalization.running_mean"].numpy()
    bn_var = state_dict[f"{prefix}.normalization.running_var"].numpy()
    return conv_w, bn_gamma, bn_beta, bn_mean, bn_var


def extract_and_fold(state_dict, hf_prefix):
    conv_w, bn_gamma, bn_beta, bn_mean, bn_var = get_conv_bn_params(state_dict, hf_prefix)
    w_folded, b_folded = fold_bn(conv_w, bn_gamma, bn_beta, bn_mean, bn_var)
    return reshape_conv_weight(w_folded), b_folded, bn_gamma, bn_beta


# ---------------------------------------------------------------------------
# Header writing
# ---------------------------------------------------------------------------

def write_header(output_path, layers_data):
    with open(output_path, "w") as f:
        f.write("#ifndef RESNET50_CIFAR10_PARAMETERS_H\n")
        f.write("#define RESNET50_CIFAR10_PARAMETERS_H\n\n")
        f.write("#include <include/gemmini_params.h>\n")
        f.write("#include <stdbool.h>\n\n")

        for entry in layers_data:
            name = entry["name"]
            if entry["layer_type"] == "fc":
                write_fc_layer(f, entry)
            else:
                write_conv_layer(f, entry)
            f.write("\n\n")

        f.write("#endif // RESNET50_CIFAR10_PARAMETERS_H\n")


def write_conv_layer(f, entry):
    name = entry["name"]
    w_int = entry["weight"]
    b_int = entry["bias"]
    output_scale_str = entry["output_scale_str"]
    p = CONV_PARAMS[name]

    f.write(f"static const elem_t {name}_w[{p['patch_size']}][{p['out_channels']}] row_align(1) = ")
    f.write(fmt_int_2d(w_int))
    f.write(";\n")
    f.write(f"static const acc_t {name}_b[{p['out_channels']}] row_align_acc(1) = ")
    f.write(fmt_int_1d(b_int))
    f.write(";\n")
    # Write _in and _out buffers
    for buf_name, d1, d2 in BUFFERS:
        if buf_name == f"{name}_in" or buf_name == f"{name}_out":
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
    f.write(f".res_scale={p['res_scale']}")
    f.write("};\n")


def write_fc_layer(f, entry):
    name = entry["name"]
    w_int = entry["weight"]   # [2048][10]
    b_int = entry["bias"]     # [10][4]
    output_scale_str = entry["output_scale_str"]
    p = FC_PARAMS[name]
    out_f = p["out_features"]
    in_f = p["in_features"]

    f.write(f"static const elem_t {name}_w[{in_f}][{out_f}] row_align(1) = ")
    f.write(fmt_int_2d(w_int))
    f.write(";\n")
    f.write(f"static const acc_t {name}_b[{out_f}][{BATCH_SIZE}] row_align_acc(1) = ")
    f.write(fmt_int_2d(b_int))
    f.write(";\n")
    f.write(f"static elem_t {name}_out[{out_f}][{BATCH_SIZE}] row_align(1);\n")
    f.write(f"static const struct FcParams {name}_params = {{")
    f.write(f".batch_size={p['batch_size']}, ")
    f.write(f".in_features={in_f}, .out_features={out_f}, ")
    f.write(f".bias={p['bias']}, ")
    f.write(f".output_scale={output_scale_str}, ")
    f.write(f".I={p['I']}, .J={p['J']}, .K={p['K']}")
    f.write("};\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print(f"Loading {MODEL_NAME} from HuggingFace...")
    model = ResNetForImageClassification.from_pretrained(MODEL_NAME)
    model.eval()
    state_dict = {k: v.detach().cpu().float() for k, v in model.state_dict().items()}

    # Verify model key structure
    all_keys = list(state_dict.keys())
    has_embedder = any("resnet.embedder" in k for k in all_keys)
    has_encoder = any("resnet.encoder" in k for k in all_keys)
    print(f"  HF key structure: embedder={has_embedder}, encoder={has_encoder}")
    if not has_embedder or not has_encoder:
        print("WARNING: Unexpected key structure. Available top-level keys:")
        top_keys = sorted(set(k.split(".")[0] for k in all_keys))
        print(f"  {top_keys}")

    classifier_w = state_dict["classifier.1.weight"]
    actual_classes = classifier_w.shape[0]
    print(f"Model: {actual_classes} output classes (expected {NUM_CLASSES})")
    assert actual_classes == NUM_CLASSES, \
        f"Expected {NUM_CLASSES} classes, got {actual_classes}"

    print(f"\nSpatial dimensions for {INPUT_DIM}x{INPUT_DIM} input:")
    for name, p in CONV_PARAMS.items():
        print(f"  {name:10s}  {p['in_row_dim']:3d}x{p['in_col_dim']:<3d} -> "
              f"{p['out_row_dim']:3d}x{p['out_col_dim']:<3d}  "
              f"n_patches={p['n_patches']:5d}  {p['in_channels']}->{p['out_channels']}")

    # -----------------------------------------------------------------------
    # Extract & quantize all layers with scale propagation
    # -----------------------------------------------------------------------
    x_scale = 1.0  # input images in [-128, 127], 1 LSB = 1 pixel unit
    layers_data = []

    print(f"\nExtracting and quantizing {len(LAYER_MAPPING)} layers...")
    for gemmini_name, hf_prefix, layer_type in LAYER_MAPPING:
        if layer_type == "fc":
            fc_w_float = state_dict["classifier.1.weight"].numpy()   # [10, 2048]
            fc_b_float = state_dict["classifier.1.bias"].numpy()     # [10]

            # Gemmini FC layout: w[in_features][out_features] = w[2048][10]
            w_gemmini = fc_w_float.T  # [2048, 10]
            w_int, w_scale = quantize_weight_int8(w_gemmini)

            combined_scale = w_scale * x_scale
            b_int_1d = quantize_bias_int32(fc_b_float, combined_scale)
            # Bias replicated across batch: [out_features][batch_size] = [10][4]
            b_int_2d = np.tile(b_int_1d.reshape(-1, 1), (1, BATCH_SIZE))

            fc_y_range = max(
                float(np.max(np.abs(fc_b_float))
                      + np.max(np.abs(fc_w_float)) * np.sqrt(2048) * x_scale * 10),
                1.0
            )
            N, output_scale_str = compute_output_scale_pow2(w_scale, x_scale, fc_y_range)

            print(f"  {gemmini_name:12s}  w_scale={w_scale:.6f}  x_scale={x_scale:.6f}  "
                  f"y_range={fc_y_range:.2f}  output_scale=1/(1<<{N})")

            layers_data.append({
                "name": gemmini_name,
                "layer_type": "fc",
                "weight": w_int,
                "bias": b_int_2d,
                "output_scale_str": output_scale_str,
            })

        else:
            # Conv layer: extract and fold BN
            w_float, b_float, bn_gamma, bn_beta = extract_and_fold(state_dict, hf_prefix)

            has_relu = LAYER_ACTIVATION.get(gemmini_name, False)

            w_int, w_scale = quantize_weight_int8(w_float)
            combined_scale = w_scale * x_scale
            b_int = quantize_bias_int32(b_float, combined_scale)

            bn_gamma_np = bn_gamma if isinstance(bn_gamma, np.ndarray) else bn_gamma.numpy()
            bn_beta_np  = bn_beta  if isinstance(bn_beta,  np.ndarray) else bn_beta.numpy()
            y_range = estimate_activation_range(bn_gamma_np, bn_beta_np, has_relu)

            N, output_scale_str = compute_output_scale_pow2(w_scale, x_scale, y_range)

            print(f"  {gemmini_name:12s}  w_scale={w_scale:.6f}  x_scale={x_scale:.6f}  "
                  f"y_range={y_range:.2f}  output_scale=1/(1<<{N})  "
                  f"{'RELU' if has_relu else 'LINEAR'}")

            layers_data.append({
                "name": gemmini_name,
                "layer_type": "conv",
                "weight": w_int,
                "bias": b_int,
                "output_scale_str": output_scale_str,
            })

            # Scale propagation: next layer's x_scale = this layer's y_scale
            y_scale = y_range / 127.0
            x_scale = y_scale

    # -----------------------------------------------------------------------
    # Write params header
    # -----------------------------------------------------------------------
    output_dir = os.path.dirname(os.path.abspath(__file__))
    params_path = os.path.normpath(os.path.join(output_dir, "..", "resnet50_cifar10_params.h"))
    print(f"\nWriting {params_path}...")
    write_header(params_path, layers_data)
    print(f"  Done.")

    total_params = sum(
        e["weight"].size + e["bias"].size for e in layers_data
    )
    print(f"\nTotal parameters: {total_params:,}")
    print("Done.")


if __name__ == "__main__":
    main()
