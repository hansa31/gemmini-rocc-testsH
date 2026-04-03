#!/usr/bin/env python3
"""
Extract FP32 weights from HuggingFace MobileNetV2 (google/mobilenet_v2_1.0_224)
and generate:
  - ../mobilenet_params_float.h  (FP32 weights with BN folded, float mode)
  - ../images.h                  (4 sample ImageNet validation images, 224x224)

The backbone (conv_1 through conv_52) uses the ImageNet pre-trained model's
weights with BatchNorm folded into conv weights.  The FC layer outputs 1000
classes (background class stripped from the 1001-class TF model).

Key design decisions:
  - BN folding uses eps=0.001 (actual MobileNetV2 BN epsilon).
  - Dead BN channels (running_var < BN_DEAD_VAR_THRESHOLD) have their folded
    weights and biases zeroed to prevent weight explosion from 1/sqrt(~0).
  - Weights are stored as float C literals; the C code converts them to FP16
    at runtime via float_to_elem_bits().
  - Reference images are stored as FP16 bit-pattern uint16_t values with
    proper [-1, 1] normalization (mean=0.5, std=0.5).

Usage:
    pip install torch transformers torchvision numpy
    python extract_mobilenet_weights.py
"""

import os
import struct
import numpy as np
import torch
from transformers import MobileNetV2ForImageClassification

MODEL_NAME = "google/mobilenet_v2_1.0_224"
INPUT_DIM = 224
BATCH_SIZE = 4
NUM_CLASSES = 1000  # After stripping TF background class

BN_EPS = 1e-3               # Actual MobileNetV2 BN epsilon (model.config uses 0.001)
BN_DEAD_VAR_THRESHOLD = 1e-3  # Zero channels with running_var below this


# ---------------------------------------------------------------------------
# HuggingFace-to-Gemmini layer mapping
# ---------------------------------------------------------------------------
# MobileNetV2 HuggingFace structure:
#   mobilenet_v2.conv_stem.first_conv  -> conv_1       (3->32,  3x3, stride 2)
#   mobilenet_v2.conv_stem.conv_3x3    -> conv_dw_2    (32 dw, 3x3, stride 1)
#   mobilenet_v2.conv_stem.reduce_1x1  -> conv_3       (32->16, 1x1)
#   mobilenet_v2.layer.0 .. .15        -> expand + dw + project (blocks 0-15)
#   mobilenet_v2.conv_1x1              -> conv_52      (320->1280, 1x1)
#   classifier                         -> fc_53        (1280->1000)

def build_layer_mapping():
    mapping = []
    # Initial conv stem — 3 sub-layers
    mapping.append(("conv_1",    "mobilenet_v2.conv_stem.first_conv", "conv"))
    mapping.append(("conv_dw_2", "mobilenet_v2.conv_stem.conv_3x3",   "dw"))
    mapping.append(("conv_3",    "mobilenet_v2.conv_stem.reduce_1x1", "conv"))
    # Inverted residual blocks: layer.0..15 -> conv_4..conv_51
    gemmini_idx = 4
    for hf_layer_idx in range(0, 16):
        prefix = f"mobilenet_v2.layer.{hf_layer_idx}"
        mapping.append((f"conv_{gemmini_idx}", f"{prefix}.expand_1x1", "conv"))
        gemmini_idx += 1
        mapping.append((f"conv_dw_{gemmini_idx}", f"{prefix}.conv_3x3", "dw"))
        gemmini_idx += 1
        mapping.append((f"conv_{gemmini_idx}", f"{prefix}.reduce_1x1", "conv"))
        gemmini_idx += 1
    mapping.append(("conv_52", "mobilenet_v2.conv_1x1", "conv"))
    mapping.append(("fc_53", "classifier", "fc"))
    return mapping


# ---------------------------------------------------------------------------
# Layer parameter structs (mirrors mobilenet_params.h exactly)
# ---------------------------------------------------------------------------
# fmt: off
CONV_PARAMS = {
    "conv_1":     {"batch_size": 4, "in_row_dim": 224, "in_col_dim": 224, "kernel_size": 3, "in_channels": 3,    "out_channels": 32,   "stride": 2, "padding": 1, "bias": 1, "depthwise": 0, "out_row_dim": 112, "out_col_dim": 112, "n_patches": 50176, "patch_size": 27,   "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 112, "I": 50176, "J": 32,   "K": 27,   "res_scale": "1.0f"},
    "conv_dw_2":  {"batch_size": 4, "in_row_dim": 112, "in_col_dim": 112, "kernel_size": 3, "in_channels": 32,   "out_channels": 32,   "stride": 1, "padding": 1, "bias": 1, "depthwise": 1, "out_row_dim": 112, "out_col_dim": 112, "n_patches": 50176, "patch_size": 9,    "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 112, "I": 50176, "J": 32,   "res_scale": "1.0f"},
    "conv_3":     {"batch_size": 4, "in_row_dim": 112, "in_col_dim": 112, "kernel_size": 1, "in_channels": 32,   "out_channels": 16,   "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 112, "out_col_dim": 112, "n_patches": 50176, "patch_size": 32,   "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 112, "I": 50176, "J": 16,   "K": 32,   "res_scale": "1.0f"},
    "conv_4":     {"batch_size": 4, "in_row_dim": 112, "in_col_dim": 112, "kernel_size": 1, "in_channels": 16,   "out_channels": 96,   "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 112, "out_col_dim": 112, "n_patches": 50176, "patch_size": 16,   "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 112, "I": 50176, "J": 96,   "K": 16,   "res_scale": "1.0f"},
    "conv_dw_5":  {"batch_size": 4, "in_row_dim": 112, "in_col_dim": 112, "kernel_size": 3, "in_channels": 96,   "out_channels": 96,   "stride": 2, "padding": 1, "bias": 1, "depthwise": 1, "out_row_dim": 56,  "out_col_dim": 56,  "n_patches": 12544, "patch_size": 9,    "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 56,  "I": 12544, "J": 96,   "res_scale": "1.0f"},
    "conv_6":     {"batch_size": 4, "in_row_dim": 56,  "in_col_dim": 56,  "kernel_size": 1, "in_channels": 96,   "out_channels": 24,   "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 56,  "out_col_dim": 56,  "n_patches": 12544, "patch_size": 96,   "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 56,  "I": 12544, "J": 24,   "K": 96,   "res_scale": "1.0f"},
    "conv_7":     {"batch_size": 4, "in_row_dim": 56,  "in_col_dim": 56,  "kernel_size": 1, "in_channels": 24,   "out_channels": 144,  "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 56,  "out_col_dim": 56,  "n_patches": 12544, "patch_size": 24,   "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 56,  "I": 12544, "J": 144,  "K": 24,   "res_scale": "1.0f"},
    "conv_dw_8":  {"batch_size": 4, "in_row_dim": 56,  "in_col_dim": 56,  "kernel_size": 3, "in_channels": 144,  "out_channels": 144,  "stride": 1, "padding": 1, "bias": 1, "depthwise": 1, "out_row_dim": 56,  "out_col_dim": 56,  "n_patches": 12544, "patch_size": 9,    "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 56,  "I": 12544, "J": 144,  "res_scale": "1.0f"},
    "conv_9":     {"batch_size": 4, "in_row_dim": 56,  "in_col_dim": 56,  "kernel_size": 1, "in_channels": 144,  "out_channels": 24,   "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 56,  "out_col_dim": 56,  "n_patches": 12544, "patch_size": 144,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 56,  "I": 12544, "J": 24,   "K": 144,  "res_scale": "1.0f"},
    "conv_10":    {"batch_size": 4, "in_row_dim": 56,  "in_col_dim": 56,  "kernel_size": 1, "in_channels": 24,   "out_channels": 144,  "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 56,  "out_col_dim": 56,  "n_patches": 12544, "patch_size": 24,   "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 56,  "I": 12544, "J": 144,  "K": 24,   "res_scale": "1.0f"},
    "conv_dw_11": {"batch_size": 4, "in_row_dim": 56,  "in_col_dim": 56,  "kernel_size": 3, "in_channels": 144,  "out_channels": 144,  "stride": 2, "padding": 1, "bias": 1, "depthwise": 1, "out_row_dim": 28,  "out_col_dim": 28,  "n_patches": 3136,  "patch_size": 9,    "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 28,  "I": 3136,  "J": 144,  "res_scale": "1.0f"},
    "conv_12":    {"batch_size": 4, "in_row_dim": 28,  "in_col_dim": 28,  "kernel_size": 1, "in_channels": 144,  "out_channels": 32,   "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 28,  "out_col_dim": 28,  "n_patches": 3136,  "patch_size": 144,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 28,  "I": 3136,  "J": 32,   "K": 144,  "res_scale": "1.0f"},
    "conv_13":    {"batch_size": 4, "in_row_dim": 28,  "in_col_dim": 28,  "kernel_size": 1, "in_channels": 32,   "out_channels": 192,  "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 28,  "out_col_dim": 28,  "n_patches": 3136,  "patch_size": 32,   "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 28,  "I": 3136,  "J": 192,  "K": 32,   "res_scale": "1.0f"},
    "conv_dw_14": {"batch_size": 4, "in_row_dim": 28,  "in_col_dim": 28,  "kernel_size": 3, "in_channels": 192,  "out_channels": 192,  "stride": 1, "padding": 1, "bias": 1, "depthwise": 1, "out_row_dim": 28,  "out_col_dim": 28,  "n_patches": 3136,  "patch_size": 9,    "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 28,  "I": 3136,  "J": 192,  "res_scale": "1.0f"},
    "conv_15":    {"batch_size": 4, "in_row_dim": 28,  "in_col_dim": 28,  "kernel_size": 1, "in_channels": 192,  "out_channels": 32,   "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 28,  "out_col_dim": 28,  "n_patches": 3136,  "patch_size": 192,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 28,  "I": 3136,  "J": 32,   "K": 192,  "res_scale": "1.0f"},
    "conv_16":    {"batch_size": 4, "in_row_dim": 28,  "in_col_dim": 28,  "kernel_size": 1, "in_channels": 32,   "out_channels": 192,  "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 28,  "out_col_dim": 28,  "n_patches": 3136,  "patch_size": 32,   "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 28,  "I": 3136,  "J": 192,  "K": 32,   "res_scale": "1.0f"},
    "conv_dw_17": {"batch_size": 4, "in_row_dim": 28,  "in_col_dim": 28,  "kernel_size": 3, "in_channels": 192,  "out_channels": 192,  "stride": 1, "padding": 1, "bias": 1, "depthwise": 1, "out_row_dim": 28,  "out_col_dim": 28,  "n_patches": 3136,  "patch_size": 9,    "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 28,  "I": 3136,  "J": 192,  "res_scale": "1.0f"},
    "conv_18":    {"batch_size": 4, "in_row_dim": 28,  "in_col_dim": 28,  "kernel_size": 1, "in_channels": 192,  "out_channels": 32,   "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 28,  "out_col_dim": 28,  "n_patches": 3136,  "patch_size": 192,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 28,  "I": 3136,  "J": 32,   "K": 192,  "res_scale": "1.0f"},
    "conv_19":    {"batch_size": 4, "in_row_dim": 28,  "in_col_dim": 28,  "kernel_size": 1, "in_channels": 32,   "out_channels": 192,  "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 28,  "out_col_dim": 28,  "n_patches": 3136,  "patch_size": 32,   "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 28,  "I": 3136,  "J": 192,  "K": 32,   "res_scale": "1.0f"},
    "conv_dw_20": {"batch_size": 4, "in_row_dim": 28,  "in_col_dim": 28,  "kernel_size": 3, "in_channels": 192,  "out_channels": 192,  "stride": 2, "padding": 1, "bias": 1, "depthwise": 1, "out_row_dim": 14,  "out_col_dim": 14,  "n_patches": 784,   "patch_size": 9,    "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 14,  "I": 784,   "J": 192,  "res_scale": "1.0f"},
    "conv_21":    {"batch_size": 4, "in_row_dim": 14,  "in_col_dim": 14,  "kernel_size": 1, "in_channels": 192,  "out_channels": 64,   "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 14,  "out_col_dim": 14,  "n_patches": 784,   "patch_size": 192,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 14,  "I": 784,   "J": 64,   "K": 192,  "res_scale": "1.0f"},
    "conv_22":    {"batch_size": 4, "in_row_dim": 14,  "in_col_dim": 14,  "kernel_size": 1, "in_channels": 64,   "out_channels": 384,  "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 14,  "out_col_dim": 14,  "n_patches": 784,   "patch_size": 64,   "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 14,  "I": 784,   "J": 384,  "K": 64,   "res_scale": "1.0f"},
    "conv_dw_23": {"batch_size": 4, "in_row_dim": 14,  "in_col_dim": 14,  "kernel_size": 3, "in_channels": 384,  "out_channels": 384,  "stride": 1, "padding": 1, "bias": 1, "depthwise": 1, "out_row_dim": 14,  "out_col_dim": 14,  "n_patches": 784,   "patch_size": 9,    "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 14,  "I": 784,   "J": 384,  "res_scale": "1.0f"},
    "conv_24":    {"batch_size": 4, "in_row_dim": 14,  "in_col_dim": 14,  "kernel_size": 1, "in_channels": 384,  "out_channels": 64,   "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 14,  "out_col_dim": 14,  "n_patches": 784,   "patch_size": 384,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 14,  "I": 784,   "J": 64,   "K": 384,  "res_scale": "1.0f"},
    "conv_25":    {"batch_size": 4, "in_row_dim": 14,  "in_col_dim": 14,  "kernel_size": 1, "in_channels": 64,   "out_channels": 384,  "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 14,  "out_col_dim": 14,  "n_patches": 784,   "patch_size": 64,   "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 14,  "I": 784,   "J": 384,  "K": 64,   "res_scale": "1.0f"},
    "conv_dw_26": {"batch_size": 4, "in_row_dim": 14,  "in_col_dim": 14,  "kernel_size": 3, "in_channels": 384,  "out_channels": 384,  "stride": 1, "padding": 1, "bias": 1, "depthwise": 1, "out_row_dim": 14,  "out_col_dim": 14,  "n_patches": 784,   "patch_size": 9,    "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 14,  "I": 784,   "J": 384,  "res_scale": "1.0f"},
    "conv_27":    {"batch_size": 4, "in_row_dim": 14,  "in_col_dim": 14,  "kernel_size": 1, "in_channels": 384,  "out_channels": 64,   "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 14,  "out_col_dim": 14,  "n_patches": 784,   "patch_size": 384,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 14,  "I": 784,   "J": 64,   "K": 384,  "res_scale": "1.0f"},
    "conv_28":    {"batch_size": 4, "in_row_dim": 14,  "in_col_dim": 14,  "kernel_size": 1, "in_channels": 64,   "out_channels": 384,  "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 14,  "out_col_dim": 14,  "n_patches": 784,   "patch_size": 64,   "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 14,  "I": 784,   "J": 384,  "K": 64,   "res_scale": "1.0f"},
    "conv_dw_29": {"batch_size": 4, "in_row_dim": 14,  "in_col_dim": 14,  "kernel_size": 3, "in_channels": 384,  "out_channels": 384,  "stride": 1, "padding": 1, "bias": 1, "depthwise": 1, "out_row_dim": 14,  "out_col_dim": 14,  "n_patches": 784,   "patch_size": 9,    "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 14,  "I": 784,   "J": 384,  "res_scale": "1.0f"},
    "conv_30":    {"batch_size": 4, "in_row_dim": 14,  "in_col_dim": 14,  "kernel_size": 1, "in_channels": 384,  "out_channels": 64,   "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 14,  "out_col_dim": 14,  "n_patches": 784,   "patch_size": 384,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 14,  "I": 784,   "J": 64,   "K": 384,  "res_scale": "1.0f"},
    "conv_31":    {"batch_size": 4, "in_row_dim": 14,  "in_col_dim": 14,  "kernel_size": 1, "in_channels": 64,   "out_channels": 384,  "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 14,  "out_col_dim": 14,  "n_patches": 784,   "patch_size": 64,   "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 14,  "I": 784,   "J": 384,  "K": 64,   "res_scale": "1.0f"},
    "conv_dw_32": {"batch_size": 4, "in_row_dim": 14,  "in_col_dim": 14,  "kernel_size": 3, "in_channels": 384,  "out_channels": 384,  "stride": 1, "padding": 1, "bias": 1, "depthwise": 1, "out_row_dim": 14,  "out_col_dim": 14,  "n_patches": 784,   "patch_size": 9,    "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 14,  "I": 784,   "J": 384,  "res_scale": "1.0f"},
    "conv_33":    {"batch_size": 4, "in_row_dim": 14,  "in_col_dim": 14,  "kernel_size": 1, "in_channels": 384,  "out_channels": 96,   "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 14,  "out_col_dim": 14,  "n_patches": 784,   "patch_size": 384,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 14,  "I": 784,   "J": 96,   "K": 384,  "res_scale": "1.0f"},
    "conv_34":    {"batch_size": 4, "in_row_dim": 14,  "in_col_dim": 14,  "kernel_size": 1, "in_channels": 96,   "out_channels": 576,  "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 14,  "out_col_dim": 14,  "n_patches": 784,   "patch_size": 96,   "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 14,  "I": 784,   "J": 576,  "K": 96,   "res_scale": "1.0f"},
    "conv_dw_35": {"batch_size": 4, "in_row_dim": 14,  "in_col_dim": 14,  "kernel_size": 3, "in_channels": 576,  "out_channels": 576,  "stride": 1, "padding": 1, "bias": 1, "depthwise": 1, "out_row_dim": 14,  "out_col_dim": 14,  "n_patches": 784,   "patch_size": 9,    "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 14,  "I": 784,   "J": 576,  "res_scale": "1.0f"},
    "conv_36":    {"batch_size": 4, "in_row_dim": 14,  "in_col_dim": 14,  "kernel_size": 1, "in_channels": 576,  "out_channels": 96,   "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 14,  "out_col_dim": 14,  "n_patches": 784,   "patch_size": 576,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 14,  "I": 784,   "J": 96,   "K": 576,  "res_scale": "1.0f"},
    "conv_37":    {"batch_size": 4, "in_row_dim": 14,  "in_col_dim": 14,  "kernel_size": 1, "in_channels": 96,   "out_channels": 576,  "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 14,  "out_col_dim": 14,  "n_patches": 784,   "patch_size": 96,   "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 14,  "I": 784,   "J": 576,  "K": 96,   "res_scale": "1.0f"},
    "conv_dw_38": {"batch_size": 4, "in_row_dim": 14,  "in_col_dim": 14,  "kernel_size": 3, "in_channels": 576,  "out_channels": 576,  "stride": 1, "padding": 1, "bias": 1, "depthwise": 1, "out_row_dim": 14,  "out_col_dim": 14,  "n_patches": 784,   "patch_size": 9,    "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 14,  "I": 784,   "J": 576,  "res_scale": "1.0f"},
    "conv_39":    {"batch_size": 4, "in_row_dim": 14,  "in_col_dim": 14,  "kernel_size": 1, "in_channels": 576,  "out_channels": 96,   "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 14,  "out_col_dim": 14,  "n_patches": 784,   "patch_size": 576,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 14,  "I": 784,   "J": 96,   "K": 576,  "res_scale": "1.0f"},
    "conv_40":    {"batch_size": 4, "in_row_dim": 14,  "in_col_dim": 14,  "kernel_size": 1, "in_channels": 96,   "out_channels": 576,  "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 14,  "out_col_dim": 14,  "n_patches": 784,   "patch_size": 96,   "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 14,  "I": 784,   "J": 576,  "K": 96,   "res_scale": "1.0f"},
    "conv_dw_41": {"batch_size": 4, "in_row_dim": 14,  "in_col_dim": 14,  "kernel_size": 3, "in_channels": 576,  "out_channels": 576,  "stride": 2, "padding": 1, "bias": 1, "depthwise": 1, "out_row_dim": 7,   "out_col_dim": 7,   "n_patches": 196,   "patch_size": 9,    "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 7,   "I": 196,   "J": 576,  "res_scale": "1.0f"},
    "conv_42":    {"batch_size": 4, "in_row_dim": 7,   "in_col_dim": 7,   "kernel_size": 1, "in_channels": 576,  "out_channels": 160,  "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 7,   "out_col_dim": 7,   "n_patches": 196,   "patch_size": 576,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 7,   "I": 196,   "J": 160,  "K": 576,  "res_scale": "1.0f"},
    "conv_43":    {"batch_size": 4, "in_row_dim": 7,   "in_col_dim": 7,   "kernel_size": 1, "in_channels": 160,  "out_channels": 960,  "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 7,   "out_col_dim": 7,   "n_patches": 196,   "patch_size": 160,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 7,   "I": 196,   "J": 960,  "K": 160,  "res_scale": "1.0f"},
    "conv_dw_44": {"batch_size": 4, "in_row_dim": 7,   "in_col_dim": 7,   "kernel_size": 3, "in_channels": 960,  "out_channels": 960,  "stride": 1, "padding": 1, "bias": 1, "depthwise": 1, "out_row_dim": 7,   "out_col_dim": 7,   "n_patches": 196,   "patch_size": 9,    "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 7,   "I": 196,   "J": 960,  "res_scale": "1.0f"},
    "conv_45":    {"batch_size": 4, "in_row_dim": 7,   "in_col_dim": 7,   "kernel_size": 1, "in_channels": 960,  "out_channels": 160,  "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 7,   "out_col_dim": 7,   "n_patches": 196,   "patch_size": 960,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 7,   "I": 196,   "J": 160,  "K": 960,  "res_scale": "1.0f"},
    "conv_46":    {"batch_size": 4, "in_row_dim": 7,   "in_col_dim": 7,   "kernel_size": 1, "in_channels": 160,  "out_channels": 960,  "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 7,   "out_col_dim": 7,   "n_patches": 196,   "patch_size": 160,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 7,   "I": 196,   "J": 960,  "K": 160,  "res_scale": "1.0f"},
    "conv_dw_47": {"batch_size": 4, "in_row_dim": 7,   "in_col_dim": 7,   "kernel_size": 3, "in_channels": 960,  "out_channels": 960,  "stride": 1, "padding": 1, "bias": 1, "depthwise": 1, "out_row_dim": 7,   "out_col_dim": 7,   "n_patches": 196,   "patch_size": 9,    "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 7,   "I": 196,   "J": 960,  "res_scale": "1.0f"},
    "conv_48":    {"batch_size": 4, "in_row_dim": 7,   "in_col_dim": 7,   "kernel_size": 1, "in_channels": 960,  "out_channels": 160,  "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 7,   "out_col_dim": 7,   "n_patches": 196,   "patch_size": 960,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 7,   "I": 196,   "J": 160,  "K": 960,  "res_scale": "1.0f"},
    "conv_49":    {"batch_size": 4, "in_row_dim": 7,   "in_col_dim": 7,   "kernel_size": 1, "in_channels": 160,  "out_channels": 960,  "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 7,   "out_col_dim": 7,   "n_patches": 196,   "patch_size": 160,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 7,   "I": 196,   "J": 960,  "K": 160,  "res_scale": "1.0f"},
    "conv_dw_50": {"batch_size": 4, "in_row_dim": 7,   "in_col_dim": 7,   "kernel_size": 3, "in_channels": 960,  "out_channels": 960,  "stride": 1, "padding": 1, "bias": 1, "depthwise": 1, "out_row_dim": 7,   "out_col_dim": 7,   "n_patches": 196,   "patch_size": 9,    "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 7,   "I": 196,   "J": 960,  "res_scale": "1.0f"},
    "conv_51":    {"batch_size": 4, "in_row_dim": 7,   "in_col_dim": 7,   "kernel_size": 1, "in_channels": 960,  "out_channels": 320,  "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 7,   "out_col_dim": 7,   "n_patches": 196,   "patch_size": 960,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 7,   "I": 196,   "J": 320,  "K": 960,  "res_scale": "1.0f"},
    "conv_52":    {"batch_size": 4, "in_row_dim": 7,   "in_col_dim": 7,   "kernel_size": 1, "in_channels": 320,  "out_channels": 1280, "stride": 1, "padding": 0, "bias": 1, "depthwise": 0, "out_row_dim": 7,   "out_col_dim": 7,   "n_patches": 196,   "patch_size": 320,  "pool_size": 1, "pool_stride": 1, "pool_padding": 0, "out_dim_pooled": 7,   "I": 196,   "J": 1280, "K": 320,  "res_scale": "1.0f"},
}
FC_PARAMS = {
    "fc_53": {"batch_size": 4, "in_features": 1280, "out_features": 1000, "bias": 1, "I": 1000, "J": 4, "K": 1280},
}
# fmt: on

# Buffer declarations
BUFFERS = [
    ("conv_1_in",  50176, 27), ("conv_1_out", 50176, 32),
    ("conv_dw_2_out",  50176, 32),
    ("conv_3_in",  50176, 32), ("conv_3_out", 50176, 16),
    ("conv_4_in",  50176, 16), ("conv_4_out", 50176, 96),
    ("conv_dw_5_out",  12544, 96),
    ("conv_6_in",  12544, 96), ("conv_6_out", 12544, 24),
    ("conv_7_in",  12544, 24), ("conv_7_out", 12544, 144),
    ("conv_dw_8_out",  12544, 144),
    ("conv_9_in",  12544, 144), ("conv_9_out", 12544, 24),
    ("conv_10_in",  12544, 24), ("conv_10_out", 12544, 144),
    ("conv_dw_11_out", 3136, 144),
    ("conv_12_in",  3136, 144), ("conv_12_out", 3136, 32),
    ("conv_13_in",  3136, 32), ("conv_13_out", 3136, 192),
    ("conv_dw_14_out", 3136, 192),
    ("conv_15_in",  3136, 192), ("conv_15_out", 3136, 32),
    ("conv_16_in",  3136, 32), ("conv_16_out", 3136, 192),
    ("conv_dw_17_out", 3136, 192),
    ("conv_18_in",  3136, 192), ("conv_18_out", 3136, 32),
    ("conv_19_in",  3136, 32), ("conv_19_out", 3136, 192),
    ("conv_dw_20_out", 784, 192),
    ("conv_21_in",  784, 192), ("conv_21_out", 784, 64),
    ("conv_22_in",  784, 64), ("conv_22_out", 784, 384),
    ("conv_dw_23_out", 784, 384),
    ("conv_24_in",  784, 384), ("conv_24_out", 784, 64),
    ("conv_25_in",  784, 64), ("conv_25_out", 784, 384),
    ("conv_dw_26_out", 784, 384),
    ("conv_27_in",  784, 384), ("conv_27_out", 784, 64),
    ("conv_28_in",  784, 64), ("conv_28_out", 784, 384),
    ("conv_dw_29_out", 784, 384),
    ("conv_30_in",  784, 384), ("conv_30_out", 784, 64),
    ("conv_31_in",  784, 64), ("conv_31_out", 784, 384),
    ("conv_dw_32_out", 784, 384),
    ("conv_33_in",  784, 384), ("conv_33_out", 784, 96),
    ("conv_34_in",  784, 96), ("conv_34_out", 784, 576),
    ("conv_dw_35_out", 784, 576),
    ("conv_36_in",  784, 576), ("conv_36_out", 784, 96),
    ("conv_37_in",  784, 96), ("conv_37_out", 784, 576),
    ("conv_dw_38_out", 784, 576),
    ("conv_39_in",  784, 576), ("conv_39_out", 784, 96),
    ("conv_40_in",  784, 96), ("conv_40_out", 784, 576),
    ("conv_dw_41_out", 196, 576),
    ("conv_42_in",  196, 576), ("conv_42_out", 196, 160),
    ("conv_43_in",  196, 160), ("conv_43_out", 196, 960),
    ("conv_dw_44_out", 196, 960),
    ("conv_45_in",  196, 960), ("conv_45_out", 196, 160),
    ("conv_46_in",  196, 160), ("conv_46_out", 196, 960),
    ("conv_dw_47_out", 196, 960),
    ("conv_48_in",  196, 960), ("conv_48_out", 196, 160),
    ("conv_49_in",  196, 160), ("conv_49_out", 196, 960),
    ("conv_dw_50_out", 196, 960),
    ("conv_51_in",  196, 960), ("conv_51_out", 196, 320),
    ("conv_52_in",  196, 320), ("conv_52_out", 196, 1280),
    ("fc_53_out", 1000, 4),
]


# ---------------------------------------------------------------------------
# BatchNorm folding with dead-channel zeroing
# ---------------------------------------------------------------------------

def fold_bn(conv_weight, bn_weight, bn_bias, bn_mean, bn_var,
            eps=BN_EPS, dead_threshold=BN_DEAD_VAR_THRESHOLD):
    """Fold BN into conv weights. Zero dead channels (var < threshold)."""
    inv_std = 1.0 / np.sqrt(bn_var + eps)
    scale = bn_weight * inv_std
    shape = [conv_weight.shape[0]] + [1] * (conv_weight.ndim - 1)
    w_folded = conv_weight * scale.reshape(shape)
    b_folded = bn_bias - bn_weight * bn_mean * inv_std

    dead_mask = bn_var < dead_threshold
    n_dead = int(np.sum(dead_mask))
    if n_dead > 0:
        w_folded[dead_mask] = 0.0
        b_folded[dead_mask] = 0.0

    return w_folded, b_folded, n_dead


# ---------------------------------------------------------------------------
# Weight reshaping
# ---------------------------------------------------------------------------

def reshape_conv_weight(w):
    out_ch = w.shape[0]
    patch_size = int(np.prod(w.shape[1:]))
    return w.reshape(out_ch, patch_size).T

def reshape_dw_weight(w):
    assert w.shape[1] == 1
    return w.squeeze(1)


# ---------------------------------------------------------------------------
# C code formatting (float)
# ---------------------------------------------------------------------------

def fmt_float(v):
    return f"{v:.8g}"

def fmt_1d(arr):
    return "{" + ",".join(fmt_float(v) for v in arr) + "}"

def fmt_2d(arr):
    return "{" + ",".join(fmt_1d(row) for row in arr) + "}"

def fmt_3d(arr):
    return "{" + ",".join(fmt_2d(plane) for plane in arr) + "}"


# ---------------------------------------------------------------------------
# FP16 conversion helpers (matching gemmini_float_convert.h)
# ---------------------------------------------------------------------------

def float_to_fp16_bits(val):
    """Convert a Python float to a FP16 bit pattern (uint16), matching
    the float_to_fp16_bits() in gemmini_float_convert.h (truncation, no rounding)."""
    fp32_bits = struct.unpack('<I', struct.pack('<f', float(val)))[0]
    sign = (fp32_bits >> 31) & 1
    exp32 = (fp32_bits >> 23) & 0xFF
    mant32 = fp32_bits & 0x7FFFFF

    if exp32 == 0xFF:
        if mant32 != 0:
            return (sign << 15) | 0x7C01
        return (sign << 15) | 0x7C00

    unbiased_exp = exp32 - 127
    fp16_exp = unbiased_exp + 15

    if exp32 == 0:
        return sign << 15

    if fp16_exp >= 31:
        return (sign << 15) | 0x7C00
    if fp16_exp <= 0:
        return sign << 15

    mant16 = mant32 >> 13
    return (sign << 15) | (fp16_exp << 10) | mant16


# ---------------------------------------------------------------------------
# Layer extraction
# ---------------------------------------------------------------------------

def get_conv_bn_params(state_dict, prefix):
    conv_w = state_dict[f"{prefix}.convolution.weight"].numpy()
    bn_gamma = state_dict[f"{prefix}.normalization.weight"].numpy()
    bn_beta = state_dict[f"{prefix}.normalization.bias"].numpy()
    bn_mean = state_dict[f"{prefix}.normalization.running_mean"].numpy()
    bn_var = state_dict[f"{prefix}.normalization.running_var"].numpy()
    return conv_w, bn_gamma, bn_beta, bn_mean, bn_var


def extract_layer(state_dict, gemmini_name, hf_prefix, layer_type, num_classes):
    if layer_type == "fc":
        fc_w = state_dict["classifier.weight"].numpy()
        fc_b = state_dict["classifier.bias"].numpy()

        if num_classes == 1001:
            fc_w = fc_w[1:, :]  # Strip TF background class
            fc_b = fc_b[1:]
        elif num_classes != 1000:
            raise ValueError(f"Unexpected num_classes={num_classes}")

        bias_2d = np.tile(fc_b.reshape(-1, 1), (1, BATCH_SIZE))
        return fc_w, bias_2d, 0

    conv_w, bn_gamma, bn_beta, bn_mean, bn_var = get_conv_bn_params(state_dict, hf_prefix)
    w_folded, b_folded, n_dead = fold_bn(conv_w, bn_gamma, bn_beta, bn_mean, bn_var)

    if n_dead > 0:
        print(f"    ** {gemmini_name}: zeroed {n_dead} dead BN channels "
              f"(var < {BN_DEAD_VAR_THRESHOLD})")

    if layer_type == "dw":
        return reshape_dw_weight(w_folded), b_folded, n_dead
    return reshape_conv_weight(w_folded), b_folded, n_dead


def validate_shapes(gemmini_name, weight, bias, layer_type):
    if layer_type == "fc":
        assert weight.shape == (NUM_CLASSES, 1280), \
            f"{gemmini_name}: weight {weight.shape} != ({NUM_CLASSES}, 1280)"
        assert bias.shape == (NUM_CLASSES, BATCH_SIZE), \
            f"{gemmini_name}: bias {bias.shape} != ({NUM_CLASSES}, {BATCH_SIZE})"
        return
    p = CONV_PARAMS[gemmini_name]
    out_ch = p["out_channels"]
    if layer_type == "dw":
        expected = (out_ch, 3, 3)
    else:
        expected = (p["patch_size"], out_ch)
    assert weight.shape == expected, \
        f"{gemmini_name}: weight {weight.shape} != {expected}"
    assert bias.shape == (out_ch,), \
        f"{gemmini_name}: bias {bias.shape} != ({out_ch},)"


# ---------------------------------------------------------------------------
# Header file generation (float weights stored as _w_float[], converted at runtime)
# ---------------------------------------------------------------------------

def write_header(output_path, layers_data):
    with open(output_path, "w") as f:
        f.write("#ifndef MOBILENET_FLOAT_PARAMETERS_H\n")
        f.write("#define MOBILENET_FLOAT_PARAMETERS_H\n\n")
        f.write('#include <include/gemmini_params.h>\n')
        f.write('#include "include/gemmini_float_convert.h"\n')
        f.write("#include <stdbool.h>\n\n")

        for gemmini_name, weight, bias, layer_type in layers_data:
            if layer_type == "fc":
                write_fc_layer(f, gemmini_name, weight, bias)
            elif layer_type == "dw":
                write_dw_layer(f, gemmini_name, weight, bias)
            else:
                write_conv_layer(f, gemmini_name, weight, bias)
            f.write("\n\n")

        write_convert_function(f, layers_data)
        f.write("\n#endif // MOBILENET_FLOAT_PARAMETERS_H\n")


def write_conv_layer(f, name, weight, bias):
    p = CONV_PARAMS[name]
    ps = p['patch_size']
    oc = p['out_channels']
    f.write(f"static const float {name}_w_float[{ps}][{oc}] = ")
    f.write(fmt_2d(weight))
    f.write(";\n")
    f.write(f"static elem_t {name}_w[{ps}][{oc}] row_align(1);\n")
    f.write(f"static const acc_t {name}_b[{oc}] row_align_acc(1) = ")
    f.write(fmt_1d(bias))
    f.write(";\n")
    for buf in BUFFERS:
        if buf[0].startswith(name + "_"):
            f.write(f"static elem_t {buf[0]}[{buf[1]}][{buf[2]}] row_align(1);\n")
    # ConvParams struct
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


def write_dw_layer(f, name, weight, bias):
    p = CONV_PARAMS[name]
    oc = p['out_channels']
    f.write(f"static const float {name}_w_float[{oc}][3][3] = ")
    f.write(fmt_3d(weight))
    f.write(";\n")
    f.write(f"static elem_t {name}_w[{oc}][3][3] row_align(1);\n")
    f.write(f"static const acc_t {name}_b[{oc}] row_align_acc(1) = ")
    f.write(fmt_1d(bias))
    f.write(";\n")
    for buf in BUFFERS:
        if buf[0] == f"{name}_out":
            f.write(f"static elem_t {buf[0]}[{buf[1]}][{buf[2]}] row_align(1);\n")
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
    f.write(f".res_scale={p['res_scale']}, ")
    f.write(f".I={p['I']}, .J={p['J']}")
    f.write("};\n")


def write_fc_layer(f, name, weight, bias):
    p = FC_PARAMS[name]
    out_f = p["out_features"]
    in_f = p["in_features"]
    f.write(f"static const float {name}_w_float[{out_f}][{in_f}] = ")
    f.write(fmt_2d(weight))
    f.write(";\n")
    f.write(f"static elem_t {name}_w[{out_f}][{in_f}] row_align(1);\n")
    f.write(f"static const acc_t {name}_b[{out_f}][{BATCH_SIZE}] row_align_acc(1) = ")
    f.write(fmt_2d(bias))
    f.write(";\n")
    f.write(f"static elem_t {name}_out[{out_f}][{BATCH_SIZE}] row_align(1);\n")
    f.write(f"static const struct FcParams {name}_params = {{")
    f.write(f".batch_size={p['batch_size']}, ")
    f.write(f".in_features={in_f}, .out_features={out_f}, ")
    f.write(f".bias={p['bias']}, ")
    f.write(f".output_scale=1.0f, ")
    f.write(f".I={p['I']}, .J={p['J']}, .K={p['K']}")
    f.write("};\n")


def write_convert_function(f, layers_data):
    """Write convert_all_weights() that converts float -> elem_t at runtime."""
    f.write("\n\nstatic void convert_all_weights() {\n")
    for gemmini_name, weight, bias, layer_type in layers_data:
        if layer_type == "fc":
            p = FC_PARAMS[gemmini_name]
            d0 = p["out_features"]
            d1 = p["in_features"]
            f.write(f"  for (int i = 0; i < {d0}; i++)\n")
            f.write(f"    for (int j = 0; j < {d1}; j++)\n")
            f.write(f"      {gemmini_name}_w[i][j] = float_to_elem_bits({gemmini_name}_w_float[i][j]);\n")
        elif layer_type == "dw":
            oc = weight.shape[0]
            f.write(f"  for (int i = 0; i < {oc}; i++)\n")
            f.write(f"    for (int j = 0; j < 3; j++)\n")
            f.write(f"      for (int k = 0; k < 3; k++)\n")
            f.write(f"        {gemmini_name}_w[i][j][k] = float_to_elem_bits({gemmini_name}_w_float[i][j][k]);\n")
        else:
            d0 = weight.shape[0]
            d1 = weight.shape[1]
            f.write(f"  for (int i = 0; i < {d0}; i++)\n")
            f.write(f"    for (int j = 0; j < {d1}; j++)\n")
            f.write(f"      {gemmini_name}_w[i][j] = float_to_elem_bits({gemmini_name}_w_float[i][j]);\n")
    f.write("}\n")


# ---------------------------------------------------------------------------
# ImageNet reference image generation (FP16 bit-pattern format)
# ---------------------------------------------------------------------------

def generate_imagenet_images(output_path, indices=None):
    """Generate images.h with 4 ImageNet validation images.

    Images are normalized to [-1, 1] (mean=0.5, std=0.5) and stored as
    FP16 bit-pattern uint16_t values.
    """
    try:
        from torchvision.datasets import ImageNet
        dataset = ImageNet(root="/tmp/imagenet_data", split="val")
    except Exception:
        print("WARNING: ImageNet dataset not available -- skipping image generation.")
        print("  You need to have ImageNet validation data accessible.")
        print("  Generating placeholder images.h with synthetic data instead.")
        return generate_synthetic_images(output_path, indices)

    if indices is None:
        indices = [0, 1, 2, 3]

    from PIL import Image as PILImage
    images_fp16 = []
    labels = []
    for idx in indices:
        img_pil, label = dataset[idx]
        img_pil = img_pil.convert("RGB").resize((INPUT_DIM, INPUT_DIM), PILImage.BILINEAR)
        img_np = np.array(img_pil, dtype=np.float32)  # [H, W, 3]
        img_norm = img_np / 127.5 - 1.0
        fp16_img = np.zeros_like(img_norm, dtype=np.uint16)
        for r in range(INPUT_DIM):
            for c in range(INPUT_DIM):
                for ch in range(3):
                    fp16_img[r, c, ch] = float_to_fp16_bits(img_norm[r, c, ch])
        images_fp16.append(fp16_img)
        labels.append(label)

    write_images_header(output_path, images_fp16, labels, indices)
    return labels


def generate_synthetic_images(output_path, indices=None):
    """Generate images.h with synthetic test patterns when ImageNet is unavailable.
    Uses fixed random seed for reproducibility. Labels set to known ImageNet classes."""
    if indices is None:
        indices = [0, 1, 2, 3]

    # Use known ImageNet val labels for indices 0-3: {65, 970, 230, 809}
    # (tench, alp, Shetland sheepdog, soup bowl) — standard torchvision order
    known_labels = {0: 65, 1: 970, 2: 230, 3: 809}

    rng = np.random.RandomState(42)
    images_fp16 = []
    labels = []
    for idx in indices:
        img_np = rng.randint(0, 256, size=(INPUT_DIM, INPUT_DIM, 3)).astype(np.float32)
        img_norm = img_np / 127.5 - 1.0
        fp16_img = np.zeros_like(img_norm, dtype=np.uint16)
        for r in range(INPUT_DIM):
            for c in range(INPUT_DIM):
                for ch in range(3):
                    fp16_img[r, c, ch] = float_to_fp16_bits(img_norm[r, c, ch])
        images_fp16.append(fp16_img)
        labels.append(known_labels.get(idx, 0))

    write_images_header(output_path, images_fp16, labels, indices)
    return labels


def write_images_header(output_path, images_fp16, labels, indices):
    with open(output_path, "w") as f:
        f.write("#ifndef IMAGES_H\n")
        f.write("#define IMAGES_H\n\n")
        f.write("#include <include/gemmini_params.h>\n\n")
        f.write(f"// ImageNet validation images at indices {indices}\n")
        f.write(f"// Labels: {labels}\n")
        f.write(f"// Normalization: (pixel/127.5 - 1.0) converted to FP16 bit patterns\n\n")
        f.write(f"static const elem_t images[{BATCH_SIZE}][{INPUT_DIM}][{INPUT_DIM}][3] row_align(1) = {{")
        for b in range(len(images_fp16)):
            if b > 0:
                f.write(",")
            f.write("{")
            for r in range(INPUT_DIM):
                if r > 0:
                    f.write(",")
                f.write("{")
                for c in range(INPUT_DIM):
                    if c > 0:
                        f.write(",")
                    vals = ",".join(str(int(images_fp16[b][r][c][ch])) for ch in range(3))
                    f.write("{" + vals + "}")
                f.write("}")
            f.write("}")
        f.write("};\n\n")
        f.write("#endif // IMAGES_H\n")

    print(f"  Written {output_path}")
    print(f"  Labels: {labels}")
    return labels


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print(f"Loading {MODEL_NAME} from HuggingFace...")
    model = MobileNetV2ForImageClassification.from_pretrained(MODEL_NAME)
    model.eval()
    state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    classifier_w = state_dict["classifier.weight"]
    num_classes = classifier_w.shape[0]
    print(f"Model has {num_classes} output classes")
    if num_classes == 1001:
        print("  -> Stripping background class (index 0) to get 1000 classes")

    print("\nState dict keys:")
    for k in sorted(state_dict.keys()):
        print(f"  {k}: {list(state_dict[k].shape)}")

    # Extract all layers
    mapping = build_layer_mapping()
    layers_data = []
    total_dead = 0

    print(f"\nExtracting {len(mapping)} layers...")
    for gemmini_name, hf_prefix, layer_type in mapping:
        print(f"  {gemmini_name:15s} <- {hf_prefix}")
        weight, bias, n_dead = extract_layer(state_dict, gemmini_name, hf_prefix, layer_type, num_classes)
        validate_shapes(gemmini_name, weight, bias, layer_type)
        layers_data.append((gemmini_name, weight, bias, layer_type))
        total_dead += n_dead

    if total_dead > 0:
        print(f"\n  Total dead BN channels zeroed: {total_dead}")

    # Write params header
    output_dir = os.path.dirname(os.path.abspath(__file__))
    params_path = os.path.join(output_dir, "..", "mobilenet_params_float.h")
    params_path = os.path.normpath(params_path)
    print(f"\nWriting {params_path}...")
    write_header(params_path, layers_data)

    # Generate reference images
    images_path = os.path.join(output_dir, "..", "images.h")
    images_path = os.path.normpath(images_path)
    print(f"\nGenerating {images_path}...")
    labels = generate_imagenet_images(images_path)

    # Summary
    total_params = sum(w.size + b.size for _, w, b, _ in layers_data)
    print(f"\nTotal parameters: {total_params:,}")
    print("Done!")


if __name__ == "__main__":
    main()
