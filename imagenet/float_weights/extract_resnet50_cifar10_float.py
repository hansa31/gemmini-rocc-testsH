#!/usr/bin/env python3
"""
Extract FP32 weights from HuggingFace ResNet-50 fine-tuned on CIFAR-10
(edadaltocg/resnet50_cifar10) and generate:
  - ../resnet50_cifar10_params_float.h  (FP32 weights, power-of-1 output_scale)

The CIFAR-10 ResNet-50 uses a modified stem:
  - First conv: 3x3, stride=1, padding=1 (vs 7x7, stride=2 for ImageNet)
  - No initial 3x3 max-pool
This keeps the spatial dimension at 32x32 through all of Stage 0.

BatchNorm is folded into the convolutional weights.

Usage:
    pip install torch transformers numpy
    python extract_resnet50_cifar10_float.py
"""

import os
import numpy as np
import torch
from huggingface_hub import hf_hub_download

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
    ("conv_2",  "resnet.encoder.stages.0.layers.0.layer.0",          "conv"),
    ("conv_3",  "resnet.encoder.stages.0.layers.0.layer.1",          "conv"),  # 3x3
    ("conv_4",  "resnet.encoder.stages.0.layers.0.layer.2",          "conv"),
    ("conv_5",  "resnet.encoder.stages.0.layers.0.shortcut",         "conv"),  # projection
    ("conv_6",  "resnet.encoder.stages.0.layers.1.layer.0",          "conv"),
    ("conv_7",  "resnet.encoder.stages.0.layers.1.layer.1",          "conv"),  # 3x3
    ("conv_8",  "resnet.encoder.stages.0.layers.1.layer.2",          "conv"),
    ("conv_9",  "resnet.encoder.stages.0.layers.2.layer.0",          "conv"),
    ("conv_10", "resnet.encoder.stages.0.layers.2.layer.1",          "conv"),  # 3x3
    ("conv_11", "resnet.encoder.stages.0.layers.2.layer.2",          "conv"),

    # Stage 1 (layer2) — 4 bottleneck blocks (128->128->512)
    ("conv_12", "resnet.encoder.stages.1.layers.0.layer.0",          "conv"),
    ("conv_13", "resnet.encoder.stages.1.layers.0.layer.1",          "conv"),  # 3x3, stride 2
    ("conv_14", "resnet.encoder.stages.1.layers.0.layer.2",          "conv"),
    ("conv_15", "resnet.encoder.stages.1.layers.0.shortcut",         "conv"),  # projection, stride 2
    ("conv_16", "resnet.encoder.stages.1.layers.1.layer.0",          "conv"),
    ("conv_17", "resnet.encoder.stages.1.layers.1.layer.1",          "conv"),  # 3x3
    ("conv_18", "resnet.encoder.stages.1.layers.1.layer.2",          "conv"),
    ("conv_19", "resnet.encoder.stages.1.layers.2.layer.0",          "conv"),
    ("conv_20", "resnet.encoder.stages.1.layers.2.layer.1",          "conv"),  # 3x3
    ("conv_21", "resnet.encoder.stages.1.layers.2.layer.2",          "conv"),
    ("conv_22", "resnet.encoder.stages.1.layers.3.layer.0",          "conv"),
    ("conv_23", "resnet.encoder.stages.1.layers.3.layer.1",          "conv"),  # 3x3
    ("conv_24", "resnet.encoder.stages.1.layers.3.layer.2",          "conv"),

    # Stage 2 (layer3) — 6 bottleneck blocks (256->256->1024)
    ("conv_25", "resnet.encoder.stages.2.layers.0.layer.0",          "conv"),
    ("conv_26", "resnet.encoder.stages.2.layers.0.layer.1",          "conv"),  # 3x3, stride 2
    ("conv_27", "resnet.encoder.stages.2.layers.0.layer.2",          "conv"),
    ("conv_28", "resnet.encoder.stages.2.layers.0.shortcut",         "conv"),  # projection, stride 2
    ("conv_29", "resnet.encoder.stages.2.layers.1.layer.0",          "conv"),
    ("conv_30", "resnet.encoder.stages.2.layers.1.layer.1",          "conv"),  # 3x3
    ("conv_31", "resnet.encoder.stages.2.layers.1.layer.2",          "conv"),
    ("conv_32", "resnet.encoder.stages.2.layers.2.layer.0",          "conv"),
    ("conv_33", "resnet.encoder.stages.2.layers.2.layer.1",          "conv"),  # 3x3
    ("conv_34", "resnet.encoder.stages.2.layers.2.layer.2",          "conv"),
    ("conv_35", "resnet.encoder.stages.2.layers.3.layer.0",          "conv"),
    ("conv_36", "resnet.encoder.stages.2.layers.3.layer.1",          "conv"),  # 3x3
    ("conv_37", "resnet.encoder.stages.2.layers.3.layer.2",          "conv"),
    ("conv_38", "resnet.encoder.stages.2.layers.4.layer.0",          "conv"),
    ("conv_39", "resnet.encoder.stages.2.layers.4.layer.1",          "conv"),  # 3x3
    ("conv_40", "resnet.encoder.stages.2.layers.4.layer.2",          "conv"),
    ("conv_41", "resnet.encoder.stages.2.layers.5.layer.0",          "conv"),
    ("conv_42", "resnet.encoder.stages.2.layers.5.layer.1",          "conv"),  # 3x3
    ("conv_43", "resnet.encoder.stages.2.layers.5.layer.2",          "conv"),

    # Stage 3 (layer4) — 3 bottleneck blocks (512->512->2048)
    ("conv_44", "resnet.encoder.stages.3.layers.0.layer.0",          "conv"),
    ("conv_45", "resnet.encoder.stages.3.layers.0.layer.1",          "conv"),  # 3x3, stride 2
    ("conv_46", "resnet.encoder.stages.3.layers.0.layer.2",          "conv"),
    ("conv_47", "resnet.encoder.stages.3.layers.0.shortcut",         "conv"),  # projection, stride 2
    ("conv_48", "resnet.encoder.stages.3.layers.1.layer.0",          "conv"),
    ("conv_49", "resnet.encoder.stages.3.layers.1.layer.1",          "conv"),  # 3x3
    ("conv_50", "resnet.encoder.stages.3.layers.1.layer.2",          "conv"),
    ("conv_51", "resnet.encoder.stages.3.layers.2.layer.0",          "conv"),
    ("conv_52", "resnet.encoder.stages.3.layers.2.layer.1",          "conv"),  # 3x3
    ("conv_53", "resnet.encoder.stages.3.layers.2.layer.2",          "conv"),

    # Global average pool -> FC classifier (2048->10)
    ("fc_54",   "classifier",                                         "fc"),
]

# ---------------------------------------------------------------------------
# CIFAR-10 spatial parameter table
# ---------------------------------------------------------------------------
# fmt: off
CONV_PARAMS = {
    # ---- Stem ----
    "conv_1":  {"batch_size": 4, "in_row_dim": 32,  "in_col_dim": 32,  "kernel_size": 3, "in_channels": 3,    "out_channels": 64,   "stride": 1, "padding": 1, "bias": 1, "depthwise": 0, "out_row_dim": 32,  "out_col_dim": 32,  "n_patches": 4096, "patch_size": 27,   "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 32,  "I": 4096, "J": 64,   "K": 27,   "res_scale": "1.0f"},

    # ---- Stage 0 (spatial: 32x32) ----
    "conv_2":  {"batch_size": 4, "in_row_dim": 32,  "in_col_dim": 32,  "kernel_size": 1, "in_channels": 64,   "out_channels": 64,   "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 32,  "out_col_dim": 32,  "n_patches": 4096, "patch_size": 64,   "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 32,  "I": 4096, "J": 64,   "K": 64,   "res_scale": "1.0f"},
    "conv_3":  {"batch_size": 4, "in_row_dim": 32,  "in_col_dim": 32,  "kernel_size": 3, "in_channels": 64,   "out_channels": 64,   "stride": 1, "padding": 1, "bias": 1, "depthwise": 0, "out_row_dim": 32,  "out_col_dim": 32,  "n_patches": 4096, "patch_size": 576,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 32,  "I": 4096, "J": 64,   "K": 576,  "res_scale": "1.0f"},
    "conv_4":  {"batch_size": 4, "in_row_dim": 32,  "in_col_dim": 32,  "kernel_size": 1, "in_channels": 64,   "out_channels": 256,  "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 32,  "out_col_dim": 32,  "n_patches": 4096, "patch_size": 64,   "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 32,  "I": 4096, "J": 256,  "K": 64,   "res_scale": "1.0f"},
    "conv_5":  {"batch_size": 4, "in_row_dim": 32,  "in_col_dim": 32,  "kernel_size": 1, "in_channels": 64,   "out_channels": 256,  "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 32,  "out_col_dim": 32,  "n_patches": 4096, "patch_size": 64,   "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 32,  "I": 4096, "J": 256,  "K": 64,   "res_scale": "1.0f"},
    "conv_6":  {"batch_size": 4, "in_row_dim": 32,  "in_col_dim": 32,  "kernel_size": 1, "in_channels": 256,  "out_channels": 64,   "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 32,  "out_col_dim": 32,  "n_patches": 4096, "patch_size": 256,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 32,  "I": 4096, "J": 64,   "K": 256,  "res_scale": "1.0f"},
    "conv_7":  {"batch_size": 4, "in_row_dim": 32,  "in_col_dim": 32,  "kernel_size": 3, "in_channels": 64,   "out_channels": 64,   "stride": 1, "padding": 1, "bias": 1, "depthwise": 0, "out_row_dim": 32,  "out_col_dim": 32,  "n_patches": 4096, "patch_size": 576,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 32,  "I": 4096, "J": 64,   "K": 576,  "res_scale": "1.0f"},
    "conv_8":  {"batch_size": 4, "in_row_dim": 32,  "in_col_dim": 32,  "kernel_size": 1, "in_channels": 64,   "out_channels": 256,  "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 32,  "out_col_dim": 32,  "n_patches": 4096, "patch_size": 64,   "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 32,  "I": 4096, "J": 256,  "K": 64,   "res_scale": "1.0f"},
    "conv_9":  {"batch_size": 4, "in_row_dim": 32,  "in_col_dim": 32,  "kernel_size": 1, "in_channels": 256,  "out_channels": 64,   "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 32,  "out_col_dim": 32,  "n_patches": 4096, "patch_size": 256,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 32,  "I": 4096, "J": 64,   "K": 256,  "res_scale": "1.0f"},
    "conv_10": {"batch_size": 4, "in_row_dim": 32,  "in_col_dim": 32,  "kernel_size": 3, "in_channels": 64,   "out_channels": 64,   "stride": 1, "padding": 1, "bias": 1, "depthwise": 0, "out_row_dim": 32,  "out_col_dim": 32,  "n_patches": 4096, "patch_size": 576,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 32,  "I": 4096, "J": 64,   "K": 576,  "res_scale": "1.0f"},
    "conv_11": {"batch_size": 4, "in_row_dim": 32,  "in_col_dim": 32,  "kernel_size": 1, "in_channels": 64,   "out_channels": 256,  "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 32,  "out_col_dim": 32,  "n_patches": 4096, "patch_size": 64,   "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 32,  "I": 4096, "J": 256,  "K": 64,   "res_scale": "1.0f"},

    # ---- Stage 1: Block 0 (in: 32x32, out: 16x16) ----
    "conv_12": {"batch_size": 4, "in_row_dim": 32,  "in_col_dim": 32,  "kernel_size": 1, "in_channels": 256,  "out_channels": 128,  "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 32,  "out_col_dim": 32,  "n_patches": 4096, "patch_size": 256,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 32,  "I": 4096, "J": 128,  "K": 256,  "res_scale": "1.0f"},
    "conv_13": {"batch_size": 4, "in_row_dim": 32,  "in_col_dim": 32,  "kernel_size": 3, "in_channels": 128,  "out_channels": 128,  "stride": 2, "padding": 1, "bias": 1, "depthwise": 0, "out_row_dim": 16,  "out_col_dim": 16,  "n_patches": 1024, "patch_size": 1152, "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 16,  "I": 1024, "J": 128,  "K": 1152, "res_scale": "1.0f"},
    "conv_14": {"batch_size": 4, "in_row_dim": 16,  "in_col_dim": 16,  "kernel_size": 1, "in_channels": 128,  "out_channels": 512,  "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 16,  "out_col_dim": 16,  "n_patches": 1024, "patch_size": 128,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 16,  "I": 1024, "J": 512,  "K": 128,  "res_scale": "1.0f"},
    "conv_15": {"batch_size": 4, "in_row_dim": 32,  "in_col_dim": 32,  "kernel_size": 1, "in_channels": 256,  "out_channels": 512,  "stride": 2, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 16,  "out_col_dim": 16,  "n_patches": 1024, "patch_size": 256,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 16,  "I": 1024, "J": 512,  "K": 256,  "res_scale": "1.0f"},
    # ---- Stage 1: Blocks 1-3 (16x16) ----
    "conv_16": {"batch_size": 4, "in_row_dim": 16,  "in_col_dim": 16,  "kernel_size": 1, "in_channels": 512,  "out_channels": 128,  "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 16,  "out_col_dim": 16,  "n_patches": 1024, "patch_size": 512,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 16,  "I": 1024, "J": 128,  "K": 512,  "res_scale": "1.0f"},
    "conv_17": {"batch_size": 4, "in_row_dim": 16,  "in_col_dim": 16,  "kernel_size": 3, "in_channels": 128,  "out_channels": 128,  "stride": 1, "padding": 1, "bias": 1, "depthwise": 0, "out_row_dim": 16,  "out_col_dim": 16,  "n_patches": 1024, "patch_size": 1152, "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 16,  "I": 1024, "J": 128,  "K": 1152, "res_scale": "1.0f"},
    "conv_18": {"batch_size": 4, "in_row_dim": 16,  "in_col_dim": 16,  "kernel_size": 1, "in_channels": 128,  "out_channels": 512,  "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 16,  "out_col_dim": 16,  "n_patches": 1024, "patch_size": 128,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 16,  "I": 1024, "J": 512,  "K": 128,  "res_scale": "1.0f"},
    "conv_19": {"batch_size": 4, "in_row_dim": 16,  "in_col_dim": 16,  "kernel_size": 1, "in_channels": 512,  "out_channels": 128,  "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 16,  "out_col_dim": 16,  "n_patches": 1024, "patch_size": 512,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 16,  "I": 1024, "J": 128,  "K": 512,  "res_scale": "1.0f"},
    "conv_20": {"batch_size": 4, "in_row_dim": 16,  "in_col_dim": 16,  "kernel_size": 3, "in_channels": 128,  "out_channels": 128,  "stride": 1, "padding": 1, "bias": 1, "depthwise": 0, "out_row_dim": 16,  "out_col_dim": 16,  "n_patches": 1024, "patch_size": 1152, "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 16,  "I": 1024, "J": 128,  "K": 1152, "res_scale": "1.0f"},
    "conv_21": {"batch_size": 4, "in_row_dim": 16,  "in_col_dim": 16,  "kernel_size": 1, "in_channels": 128,  "out_channels": 512,  "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 16,  "out_col_dim": 16,  "n_patches": 1024, "patch_size": 128,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 16,  "I": 1024, "J": 512,  "K": 128,  "res_scale": "1.0f"},
    "conv_22": {"batch_size": 4, "in_row_dim": 16,  "in_col_dim": 16,  "kernel_size": 1, "in_channels": 512,  "out_channels": 128,  "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 16,  "out_col_dim": 16,  "n_patches": 1024, "patch_size": 512,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 16,  "I": 1024, "J": 128,  "K": 512,  "res_scale": "1.0f"},
    "conv_23": {"batch_size": 4, "in_row_dim": 16,  "in_col_dim": 16,  "kernel_size": 3, "in_channels": 128,  "out_channels": 128,  "stride": 1, "padding": 1, "bias": 1, "depthwise": 0, "out_row_dim": 16,  "out_col_dim": 16,  "n_patches": 1024, "patch_size": 1152, "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 16,  "I": 1024, "J": 128,  "K": 1152, "res_scale": "1.0f"},
    "conv_24": {"batch_size": 4, "in_row_dim": 16,  "in_col_dim": 16,  "kernel_size": 1, "in_channels": 128,  "out_channels": 512,  "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 16,  "out_col_dim": 16,  "n_patches": 1024, "patch_size": 128,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 16,  "I": 1024, "J": 512,  "K": 128,  "res_scale": "1.0f"},

    # ---- Stage 2: Block 0 (in: 16x16, out: 8x8) ----
    "conv_25": {"batch_size": 4, "in_row_dim": 16,  "in_col_dim": 16,  "kernel_size": 1, "in_channels": 512,  "out_channels": 256,  "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 16,  "out_col_dim": 16,  "n_patches": 1024, "patch_size": 512,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 16,  "I": 1024, "J": 256,  "K": 512,  "res_scale": "1.0f"},
    "conv_26": {"batch_size": 4, "in_row_dim": 16,  "in_col_dim": 16,  "kernel_size": 3, "in_channels": 256,  "out_channels": 256,  "stride": 2, "padding": 1, "bias": 1, "depthwise": 0, "out_row_dim": 8,   "out_col_dim": 8,   "n_patches": 256,  "patch_size": 2304, "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 8,   "I": 256,  "J": 256,  "K": 2304, "res_scale": "1.0f"},
    "conv_27": {"batch_size": 4, "in_row_dim": 8,   "in_col_dim": 8,   "kernel_size": 1, "in_channels": 256,  "out_channels": 1024, "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 8,   "out_col_dim": 8,   "n_patches": 256,  "patch_size": 256,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 8,   "I": 256,  "J": 1024, "K": 256,  "res_scale": "1.0f"},
    "conv_28": {"batch_size": 4, "in_row_dim": 16,  "in_col_dim": 16,  "kernel_size": 1, "in_channels": 512,  "out_channels": 1024, "stride": 2, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 8,   "out_col_dim": 8,   "n_patches": 256,  "patch_size": 512,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 8,   "I": 256,  "J": 1024, "K": 512,  "res_scale": "1.0f"},
    # ---- Stage 2: Blocks 1-5 (8x8) ----
    "conv_29": {"batch_size": 4, "in_row_dim": 8,   "in_col_dim": 8,   "kernel_size": 1, "in_channels": 1024, "out_channels": 256,  "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 8,   "out_col_dim": 8,   "n_patches": 256,  "patch_size": 1024, "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 8,   "I": 256,  "J": 256,  "K": 1024, "res_scale": "1.0f"},
    "conv_30": {"batch_size": 4, "in_row_dim": 8,   "in_col_dim": 8,   "kernel_size": 3, "in_channels": 256,  "out_channels": 256,  "stride": 1, "padding": 1, "bias": 1, "depthwise": 0, "out_row_dim": 8,   "out_col_dim": 8,   "n_patches": 256,  "patch_size": 2304, "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 8,   "I": 256,  "J": 256,  "K": 2304, "res_scale": "1.0f"},
    "conv_31": {"batch_size": 4, "in_row_dim": 8,   "in_col_dim": 8,   "kernel_size": 1, "in_channels": 256,  "out_channels": 1024, "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 8,   "out_col_dim": 8,   "n_patches": 256,  "patch_size": 256,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 8,   "I": 256,  "J": 1024, "K": 256,  "res_scale": "1.0f"},
    "conv_32": {"batch_size": 4, "in_row_dim": 8,   "in_col_dim": 8,   "kernel_size": 1, "in_channels": 1024, "out_channels": 256,  "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 8,   "out_col_dim": 8,   "n_patches": 256,  "patch_size": 1024, "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 8,   "I": 256,  "J": 256,  "K": 1024, "res_scale": "1.0f"},
    "conv_33": {"batch_size": 4, "in_row_dim": 8,   "in_col_dim": 8,   "kernel_size": 3, "in_channels": 256,  "out_channels": 256,  "stride": 1, "padding": 1, "bias": 1, "depthwise": 0, "out_row_dim": 8,   "out_col_dim": 8,   "n_patches": 256,  "patch_size": 2304, "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 8,   "I": 256,  "J": 256,  "K": 2304, "res_scale": "1.0f"},
    "conv_34": {"batch_size": 4, "in_row_dim": 8,   "in_col_dim": 8,   "kernel_size": 1, "in_channels": 256,  "out_channels": 1024, "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 8,   "out_col_dim": 8,   "n_patches": 256,  "patch_size": 256,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 8,   "I": 256,  "J": 1024, "K": 256,  "res_scale": "1.0f"},
    "conv_35": {"batch_size": 4, "in_row_dim": 8,   "in_col_dim": 8,   "kernel_size": 1, "in_channels": 1024, "out_channels": 256,  "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 8,   "out_col_dim": 8,   "n_patches": 256,  "patch_size": 1024, "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 8,   "I": 256,  "J": 256,  "K": 1024, "res_scale": "1.0f"},
    "conv_36": {"batch_size": 4, "in_row_dim": 8,   "in_col_dim": 8,   "kernel_size": 3, "in_channels": 256,  "out_channels": 256,  "stride": 1, "padding": 1, "bias": 1, "depthwise": 0, "out_row_dim": 8,   "out_col_dim": 8,   "n_patches": 256,  "patch_size": 2304, "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 8,   "I": 256,  "J": 256,  "K": 2304, "res_scale": "1.0f"},
    "conv_37": {"batch_size": 4, "in_row_dim": 8,   "in_col_dim": 8,   "kernel_size": 1, "in_channels": 256,  "out_channels": 1024, "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 8,   "out_col_dim": 8,   "n_patches": 256,  "patch_size": 256,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 8,   "I": 256,  "J": 1024, "K": 256,  "res_scale": "1.0f"},
    "conv_38": {"batch_size": 4, "in_row_dim": 8,   "in_col_dim": 8,   "kernel_size": 1, "in_channels": 1024, "out_channels": 256,  "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 8,   "out_col_dim": 8,   "n_patches": 256,  "patch_size": 1024, "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 8,   "I": 256,  "J": 256,  "K": 1024, "res_scale": "1.0f"},
    "conv_39": {"batch_size": 4, "in_row_dim": 8,   "in_col_dim": 8,   "kernel_size": 3, "in_channels": 256,  "out_channels": 256,  "stride": 1, "padding": 1, "bias": 1, "depthwise": 0, "out_row_dim": 8,   "out_col_dim": 8,   "n_patches": 256,  "patch_size": 2304, "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 8,   "I": 256,  "J": 256,  "K": 2304, "res_scale": "1.0f"},
    "conv_40": {"batch_size": 4, "in_row_dim": 8,   "in_col_dim": 8,   "kernel_size": 1, "in_channels": 256,  "out_channels": 1024, "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 8,   "out_col_dim": 8,   "n_patches": 256,  "patch_size": 256,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 8,   "I": 256,  "J": 1024, "K": 256,  "res_scale": "1.0f"},
    "conv_41": {"batch_size": 4, "in_row_dim": 8,   "in_col_dim": 8,   "kernel_size": 1, "in_channels": 1024, "out_channels": 256,  "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 8,   "out_col_dim": 8,   "n_patches": 256,  "patch_size": 1024, "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 8,   "I": 256,  "J": 256,  "K": 1024, "res_scale": "1.0f"},
    "conv_42": {"batch_size": 4, "in_row_dim": 8,   "in_col_dim": 8,   "kernel_size": 3, "in_channels": 256,  "out_channels": 256,  "stride": 1, "padding": 1, "bias": 1, "depthwise": 0, "out_row_dim": 8,   "out_col_dim": 8,   "n_patches": 256,  "patch_size": 2304, "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 8,   "I": 256,  "J": 256,  "K": 2304, "res_scale": "1.0f"},
    "conv_43": {"batch_size": 4, "in_row_dim": 8,   "in_col_dim": 8,   "kernel_size": 1, "in_channels": 256,  "out_channels": 1024, "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 8,   "out_col_dim": 8,   "n_patches": 256,  "patch_size": 256,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 8,   "I": 256,  "J": 1024, "K": 256,  "res_scale": "1.0f"},

    # ---- Stage 3: Block 0 (in: 8x8, out: 4x4) ----
    "conv_44": {"batch_size": 4, "in_row_dim": 8,   "in_col_dim": 8,   "kernel_size": 1, "in_channels": 1024, "out_channels": 512,  "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 8,   "out_col_dim": 8,   "n_patches": 256,  "patch_size": 1024, "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 8,   "I": 256,  "J": 512,  "K": 1024, "res_scale": "1.0f"},
    "conv_45": {"batch_size": 4, "in_row_dim": 8,   "in_col_dim": 8,   "kernel_size": 3, "in_channels": 512,  "out_channels": 512,  "stride": 2, "padding": 1, "bias": 1, "depthwise": 0, "out_row_dim": 4,   "out_col_dim": 4,   "n_patches": 64,   "patch_size": 4608, "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 4,   "I": 64,   "J": 512,  "K": 4608, "res_scale": "1.0f"},
    "conv_46": {"batch_size": 4, "in_row_dim": 4,   "in_col_dim": 4,   "kernel_size": 1, "in_channels": 512,  "out_channels": 2048, "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 4,   "out_col_dim": 4,   "n_patches": 64,   "patch_size": 512,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 4,   "I": 64,   "J": 2048, "K": 512,  "res_scale": "1.0f"},
    "conv_47": {"batch_size": 4, "in_row_dim": 8,   "in_col_dim": 8,   "kernel_size": 1, "in_channels": 1024, "out_channels": 2048, "stride": 2, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 4,   "out_col_dim": 4,   "n_patches": 64,   "patch_size": 1024, "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 4,   "I": 64,   "J": 2048, "K": 1024, "res_scale": "1.0f"},
    # ---- Stage 3: Blocks 1-2 (4x4) ----
    "conv_48": {"batch_size": 4, "in_row_dim": 4,   "in_col_dim": 4,   "kernel_size": 1, "in_channels": 2048, "out_channels": 512,  "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 4,   "out_col_dim": 4,   "n_patches": 64,   "patch_size": 2048, "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 4,   "I": 64,   "J": 512,  "K": 2048, "res_scale": "1.0f"},
    "conv_49": {"batch_size": 4, "in_row_dim": 4,   "in_col_dim": 4,   "kernel_size": 3, "in_channels": 512,  "out_channels": 512,  "stride": 1, "padding": 1, "bias": 1, "depthwise": 0, "out_row_dim": 4,   "out_col_dim": 4,   "n_patches": 64,   "patch_size": 4608, "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 4,   "I": 64,   "J": 512,  "K": 4608, "res_scale": "1.0f"},
    "conv_50": {"batch_size": 4, "in_row_dim": 4,   "in_col_dim": 4,   "kernel_size": 1, "in_channels": 512,  "out_channels": 2048, "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 4,   "out_col_dim": 4,   "n_patches": 64,   "patch_size": 512,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 4,   "I": 64,   "J": 2048, "K": 512,  "res_scale": "1.0f"},
    "conv_51": {"batch_size": 4, "in_row_dim": 4,   "in_col_dim": 4,   "kernel_size": 1, "in_channels": 2048, "out_channels": 512,  "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 4,   "out_col_dim": 4,   "n_patches": 64,   "patch_size": 2048, "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 4,   "I": 64,   "J": 512,  "K": 2048, "res_scale": "1.0f"},
    "conv_52": {"batch_size": 4, "in_row_dim": 4,   "in_col_dim": 4,   "kernel_size": 3, "in_channels": 512,  "out_channels": 512,  "stride": 1, "padding": 1, "bias": 1, "depthwise": 0, "out_row_dim": 4,   "out_col_dim": 4,   "n_patches": 64,   "patch_size": 4608, "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 4,   "I": 64,   "J": 512,  "K": 4608, "res_scale": "1.0f"},
    "conv_53": {"batch_size": 4, "in_row_dim": 4,   "in_col_dim": 4,   "kernel_size": 1, "in_channels": 512,  "out_channels": 2048, "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 4,   "out_col_dim": 4,   "n_patches": 64,   "patch_size": 512,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 4,   "I": 64,   "J": 2048, "K": 512,  "res_scale": "1.0f"},
}
# fmt: on

FC_PARAMS = {
    # For float: fc_54_w[out_features][in_features] = [10][2048]   (A[I][K])
    #            average[in_features][batch_size]   = [2048][4]    (B[K][J])
    #            fc_54_out[out_features][batch_size] = [10][4]      (C[I][J])
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

# Buffer declarations (same as INT version — _in and _out per layer)
BUFFERS = [
    ("conv_1_in",   4096, 27),    ("conv_1_out",   4096, 64),
    ("conv_2_in",   4096, 64),    ("conv_2_out",   4096, 64),
    ("conv_3_in",   4096, 576),   ("conv_3_out",   4096, 64),
    ("conv_4_in",   4096, 64),    ("conv_4_out",   4096, 256),
    ("conv_5_in",   4096, 64),    ("conv_5_out",   4096, 256),
    ("conv_6_in",   4096, 256),   ("conv_6_out",   4096, 64),
    ("conv_7_in",   4096, 576),   ("conv_7_out",   4096, 64),
    ("conv_8_in",   4096, 64),    ("conv_8_out",   4096, 256),
    ("conv_9_in",   4096, 256),   ("conv_9_out",   4096, 64),
    ("conv_10_in",  4096, 576),   ("conv_10_out",  4096, 64),
    ("conv_11_in",  4096, 64),    ("conv_11_out",  4096, 256),
    ("conv_12_in",  4096, 256),   ("conv_12_out",  4096, 128),
    ("conv_13_in",  1024, 1152),  ("conv_13_out",  1024, 128),
    ("conv_14_in",  1024, 128),   ("conv_14_out",  1024, 512),
    ("conv_15_in",  1024, 256),   ("conv_15_out",  1024, 512),
    ("conv_16_in",  1024, 512),   ("conv_16_out",  1024, 128),
    ("conv_17_in",  1024, 1152),  ("conv_17_out",  1024, 128),
    ("conv_18_in",  1024, 128),   ("conv_18_out",  1024, 512),
    ("conv_19_in",  1024, 512),   ("conv_19_out",  1024, 128),
    ("conv_20_in",  1024, 1152),  ("conv_20_out",  1024, 128),
    ("conv_21_in",  1024, 128),   ("conv_21_out",  1024, 512),
    ("conv_22_in",  1024, 512),   ("conv_22_out",  1024, 128),
    ("conv_23_in",  1024, 1152),  ("conv_23_out",  1024, 128),
    ("conv_24_in",  1024, 128),   ("conv_24_out",  1024, 512),
    ("conv_25_in",  1024, 512),   ("conv_25_out",  1024, 256),
    ("conv_26_in",  256,  2304),  ("conv_26_out",  256,  256),
    ("conv_27_in",  256,  256),   ("conv_27_out",  256,  1024),
    ("conv_28_in",  256,  512),   ("conv_28_out",  256,  1024),
    ("conv_29_in",  256,  1024),  ("conv_29_out",  256,  256),
    ("conv_30_in",  256,  2304),  ("conv_30_out",  256,  256),
    ("conv_31_in",  256,  256),   ("conv_31_out",  256,  1024),
    ("conv_32_in",  256,  1024),  ("conv_32_out",  256,  256),
    ("conv_33_in",  256,  2304),  ("conv_33_out",  256,  256),
    ("conv_34_in",  256,  256),   ("conv_34_out",  256,  1024),
    ("conv_35_in",  256,  1024),  ("conv_35_out",  256,  256),
    ("conv_36_in",  256,  2304),  ("conv_36_out",  256,  256),
    ("conv_37_in",  256,  256),   ("conv_37_out",  256,  1024),
    ("conv_38_in",  256,  1024),  ("conv_38_out",  256,  256),
    ("conv_39_in",  256,  2304),  ("conv_39_out",  256,  256),
    ("conv_40_in",  256,  256),   ("conv_40_out",  256,  1024),
    ("conv_41_in",  256,  1024),  ("conv_41_out",  256,  256),
    ("conv_42_in",  256,  2304),  ("conv_42_out",  256,  256),
    ("conv_43_in",  256,  256),   ("conv_43_out",  256,  1024),
    ("conv_44_in",  256,  1024),  ("conv_44_out",  256,  512),
    ("conv_45_in",  64,   4608),  ("conv_45_out",  64,   512),
    ("conv_46_in",  64,   512),   ("conv_46_out",  64,   2048),
    ("conv_47_in",  64,   1024),  ("conv_47_out",  64,   2048),
    ("conv_48_in",  64,   2048),  ("conv_48_out",  64,   512),
    ("conv_49_in",  64,   4608),  ("conv_49_out",  64,   512),
    ("conv_50_in",  64,   512),   ("conv_50_out",  64,   2048),
    ("conv_51_in",  64,   2048),  ("conv_51_out",  64,   512),
    ("conv_52_in",  64,   4608),  ("conv_52_out",  64,   512),
    ("conv_53_in",  64,   512),   ("conv_53_out",  64,   2048),
    ("fc_54_out",   NUM_CLASSES, BATCH_SIZE),
]


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
# C formatting (float)
# ---------------------------------------------------------------------------

def fmt_float(v):
    return f"{float(v):.8g}"

def fmt_float_1d(arr):
    return "{" + ",".join(fmt_float(v) for v in arr.flat) + "}"

def fmt_float_2d(arr):
    rows = []
    for r in range(arr.shape[0]):
        rows.append("{" + ",".join(fmt_float(v) for v in arr[r]) + "}")
    return "{" + ",".join(rows) + "}"


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


# ---------------------------------------------------------------------------
# Header writing
# ---------------------------------------------------------------------------

def write_header(output_path, layers_data):
    with open(output_path, "w") as f:
        f.write("#ifndef RESNET50_CIFAR10_PARAMETERS_FLOAT_H\n")
        f.write("#define RESNET50_CIFAR10_PARAMETERS_FLOAT_H\n\n")
        f.write("#include <include/gemmini_params.h>\n")
        f.write("#include <stdbool.h>\n\n")

        for entry in layers_data:
            if entry["layer_type"] == "fc":
                write_fc_layer(f, entry)
            else:
                write_conv_layer(f, entry)
            f.write("\n\n")

        f.write("#endif // RESNET50_CIFAR10_PARAMETERS_FLOAT_H\n")


def write_conv_layer(f, entry):
    name = entry["name"]
    w = entry["weight"]  # [patch_size][out_channels], float
    b = entry["bias"]    # [out_channels], float
    p = CONV_PARAMS[name]

    f.write(f"static const elem_t {name}_w[{p['patch_size']}][{p['out_channels']}] row_align(1) = ")
    f.write(fmt_float_2d(w))
    f.write(";\n")
    f.write(f"static const acc_t {name}_b[{p['out_channels']}] row_align_acc(1) = ")
    f.write(fmt_float_1d(b))
    f.write(";\n")
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
    f.write(f".output_scale=1.0f, ")
    f.write(f".I={p['I']}, .J={p['J']}, .K={p['K']}, ")
    f.write(f".res_scale={p['res_scale']}")
    f.write("};\n")


def write_fc_layer(f, entry):
    name = entry["name"]
    w = entry["weight"]  # [out_features][in_features], float
    b = entry["bias"]    # [out_features][batch_size], float
    p = FC_PARAMS[name]
    out_f = p["out_features"]
    in_f = p["in_features"]

    f.write(f"static const elem_t {name}_w[{out_f}][{in_f}] row_align(1) = ")
    f.write(fmt_float_2d(w))
    f.write(";\n")
    f.write(f"static const acc_t {name}_b[{out_f}][{BATCH_SIZE}] row_align_acc(1) = ")
    f.write(fmt_float_2d(b))
    f.write(";\n")
    f.write(f"static elem_t {name}_out[{out_f}][{BATCH_SIZE}] row_align(1);\n")
    f.write(f"static const struct FcParams {name}_params = {{")
    f.write(f".batch_size={p['batch_size']}, ")
    f.write(f".in_features={in_f}, .out_features={out_f}, ")
    f.write(f".bias={p['bias']}, ")
    f.write(f".output_scale=1.0f, ")
    f.write(f".I={p['I']}, .J={p['J']}, .K={p['K']}")
    f.write("};\n")


# ---------------------------------------------------------------------------
# Key remapping: torchvision/timm -> HuggingFace format
# ---------------------------------------------------------------------------
# The edadaltocg/resnet50_cifar10 checkpoint uses torchvision-style keys
# (conv1, bn1, layer1.0.conv1, etc.) but the rest of this script expects
# HuggingFace-style keys (resnet.embedder.embedder.convolution.weight, etc.).

def _remap_torchvision_to_hf(sd):
    """Remap torchvision ResNet state_dict keys to HuggingFace format."""
    mapped = {}
    for k, v in sd.items():
        new_key = None
        if k.startswith('conv1.'):
            suffix = k[len('conv1.'):]
            new_key = f"resnet.embedder.embedder.convolution.{suffix}"
        elif k.startswith('bn1.'):
            suffix = k[len('bn1.'):]
            new_key = f"resnet.embedder.embedder.normalization.{suffix}"
        elif k.startswith('fc.'):
            suffix = k[len('fc.'):]
            new_key = f"classifier.1.{suffix}"
        elif k.startswith('layer'):
            parts = k.split('.')
            stage = int(parts[0][5:]) - 1   # layer1 -> stage 0
            block = int(parts[1])
            if parts[2] == 'downsample':
                sub_type = 'convolution' if parts[3] == '0' else 'normalization'
                suffix = '.'.join(parts[4:])
                new_key = (f"resnet.encoder.stages.{stage}.layers.{block}"
                           f".shortcut.{sub_type}.{suffix}")
            elif parts[2].startswith('conv'):
                conv_idx = int(parts[2][4:]) - 1   # conv1 -> 0
                suffix = '.'.join(parts[3:])
                new_key = (f"resnet.encoder.stages.{stage}.layers.{block}"
                           f".layer.{conv_idx}.convolution.{suffix}")
            elif parts[2].startswith('bn'):
                bn_idx = int(parts[2][2:]) - 1      # bn1 -> 0
                suffix = '.'.join(parts[3:])
                new_key = (f"resnet.encoder.stages.{stage}.layers.{block}"
                           f".layer.{bn_idx}.normalization.{suffix}")
        if new_key is not None:
            mapped[new_key] = v
    return mapped


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print(f"Loading {MODEL_NAME} from HuggingFace...")
    ckpt_path = hf_hub_download(repo_id=MODEL_NAME, filename="pytorch_model.bin")
    raw_sd = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    state_dict = {k: v.float() for k, v in _remap_torchvision_to_hf(raw_sd).items()}

    classifier_w = state_dict["classifier.1.weight"]
    actual_classes = classifier_w.shape[0]
    print(f"Model: {actual_classes} output classes (expected {NUM_CLASSES})")
    assert actual_classes == NUM_CLASSES, \
        f"Expected {NUM_CLASSES} classes, got {actual_classes}"

    layers_data = []

    print(f"\nExtracting and folding BN for {len(LAYER_MAPPING)} layers...")
    for gemmini_name, hf_prefix, layer_type in LAYER_MAPPING:
        if layer_type == "fc":
            # FC: PyTorch weight [out_classes, in_features] -> keep as [out_features][in_features]
            fc_w_float = state_dict["classifier.1.weight"].numpy()   # [10, 2048]
            fc_b_float = state_dict["classifier.1.bias"].numpy()     # [10]

            # Bias replicated across batch: [out_features][batch_size] = [10][4]
            b_2d = np.tile(fc_b_float.reshape(-1, 1), (1, BATCH_SIZE))  # [10, 4]

            print(f"  {gemmini_name:12s}  w={fc_w_float.shape}  b={fc_b_float.shape}")

            layers_data.append({
                "name": gemmini_name,
                "layer_type": "fc",
                "weight": fc_w_float,   # [10][2048]
                "bias":   b_2d,         # [10][4]
            })

        else:
            # Conv: fold BN
            conv_w = state_dict[f"{hf_prefix}.convolution.weight"].numpy()
            bn_gamma = state_dict[f"{hf_prefix}.normalization.weight"].numpy()
            bn_beta  = state_dict[f"{hf_prefix}.normalization.bias"].numpy()
            bn_mean  = state_dict[f"{hf_prefix}.normalization.running_mean"].numpy()
            bn_var   = state_dict[f"{hf_prefix}.normalization.running_var"].numpy()

            w_folded, b_folded = fold_bn(conv_w, bn_gamma, bn_beta, bn_mean, bn_var)
            w_gemmini = reshape_conv_weight(w_folded)  # [patch_size][out_channels]

            p = CONV_PARAMS[gemmini_name]
            assert w_gemmini.shape == (p["patch_size"], p["out_channels"]), \
                f"{gemmini_name}: weight shape {w_gemmini.shape} != ({p['patch_size']}, {p['out_channels']})"
            assert b_folded.shape == (p["out_channels"],), \
                f"{gemmini_name}: bias shape {b_folded.shape} != ({p['out_channels']},)"

            print(f"  {gemmini_name:12s}  w={w_gemmini.shape}  b={b_folded.shape}"
                  f"  w_max={np.max(np.abs(w_folded)):.4f}")

            layers_data.append({
                "name": gemmini_name,
                "layer_type": "conv",
                "weight": w_gemmini,
                "bias":   b_folded,
            })

    # Write params header
    output_dir = os.path.dirname(os.path.abspath(__file__))
    params_path = os.path.normpath(os.path.join(output_dir, "..", "resnet50_cifar10_params_float.h"))
    print(f"\nWriting {params_path} ...")
    write_header(params_path, layers_data)
    print("Done.")


if __name__ == "__main__":
    main()
