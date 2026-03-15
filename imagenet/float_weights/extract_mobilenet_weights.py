#!/usr/bin/env python3
"""
Extract FP32 weights from HuggingFace MobileNetV2 (google/mobilenet_v2_1.0_224)
and generate mobilenet_params_float.h matching the structure of mobilenet_params.h.

BatchNorm is folded into conv weights and biases. The FC classifier layer has
no BatchNorm and is extracted directly.

Usage:
    pip install torch transformers
    python extract_mobilenet_weights.py

Output:
    ../mobilenet_params_float.h
"""

import os
import sys
import numpy as np

import torch
from transformers import MobileNetV2ForImageClassification


# ---------------------------------------------------------------------------
# HuggingFace-to-Gemmini layer mapping
# ---------------------------------------------------------------------------
# MobileNetV2ForImageClassification state dict key prefixes → Gemmini layer names.
#
# HuggingFace model structure:
#   mobilenet_v2.conv_stem.first_conv  → conv_1       (3→32,  3×3, stride 2)
#   mobilenet_v2.conv_stem.conv_3x3    → conv_dw_2    (32 dw, 3×3, stride 1)
#   mobilenet_v2.conv_stem.reduce_1x1  → conv_3       (32→16, 1×1)
#   mobilenet_v2.layer.0 .. .15        → expand + dw + project (blocks 0-15)
#   mobilenet_v2.conv_1x1              → conv_52      (320→1280, 1×1)
#   classifier                         → fc_53        (1280→1000)
#
# Each inverted residual (layer.i) contains:
#   .expand_1x1  (present in all 16 blocks; layer.0 is t=6, 16→96)
#   .conv_3x3    (depthwise)
#   .reduce_1x1  (projection, no activation)

def build_layer_mapping():
    """Return list of (gemmini_name, hf_prefix, layer_type) tuples.

    layer_type is one of: 'conv', 'dw', 'fc'
    """
    mapping = []

    # 1) Initial conv stem — 3 sub-layers inside mobilenet_v2.conv_stem
    mapping.append(("conv_1",    "mobilenet_v2.conv_stem.first_conv", "conv"))
    mapping.append(("conv_dw_2", "mobilenet_v2.conv_stem.conv_3x3",   "dw"))
    mapping.append(("conv_3",    "mobilenet_v2.conv_stem.reduce_1x1", "conv"))

    # 2) Inverted residual blocks
    # layer.0..15 → gemmini conv_4..conv_51 (each block: expand + dw + project)
    gemmini_idx = 4
    for hf_layer_idx in range(0, 16):
        prefix = f"mobilenet_v2.layer.{hf_layer_idx}"
        # expand 1x1
        mapping.append((f"conv_{gemmini_idx}", f"{prefix}.expand_1x1", "conv"))
        gemmini_idx += 1
        # depthwise 3x3
        mapping.append((f"conv_dw_{gemmini_idx}", f"{prefix}.conv_3x3", "dw"))
        gemmini_idx += 1
        # project 1x1
        mapping.append((f"conv_{gemmini_idx}", f"{prefix}.reduce_1x1", "conv"))
        gemmini_idx += 1

    # 3) Final 1x1 conv
    mapping.append(("conv_52", "mobilenet_v2.conv_1x1", "conv"))

    # 4) Classifier
    mapping.append(("fc_53", "classifier", "fc"))

    return mapping


# ---------------------------------------------------------------------------
# BatchNorm folding
# ---------------------------------------------------------------------------

def fold_bn(conv_weight, bn_weight, bn_bias, bn_mean, bn_var, eps=1e-5):
    """Fold BatchNorm into conv weight and bias.

    Args:
        conv_weight: [out_ch, in_ch_or_groups, kH, kW]
        bn_weight:   gamma [out_ch]
        bn_bias:     beta  [out_ch]
        bn_mean:     running_mean [out_ch]
        bn_var:      running_var  [out_ch]
        eps:         BN epsilon

    Returns:
        (folded_weight, folded_bias) as numpy arrays with same shape as inputs.
    """
    inv_std = 1.0 / np.sqrt(bn_var + eps)
    scale = bn_weight * inv_std  # gamma / sqrt(var + eps)

    # Reshape scale for broadcasting over conv weight dimensions
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
    out_ch = w.shape[0]
    patch_size = int(np.prod(w.shape[1:]))
    return w.reshape(out_ch, patch_size).T  # [patch_size, out_ch]


def reshape_dw_weight(w):
    """Depthwise conv: PyTorch [channels, 1, kH, kW] → Gemmini [channels, kH, kW]."""
    assert w.shape[1] == 1, f"Expected groups dim=1, got {w.shape[1]}"
    return w.squeeze(1)  # [channels, kH, kW]


# ---------------------------------------------------------------------------
# C code formatting
# ---------------------------------------------------------------------------

BATCH_SIZE = 4  # matches mobilenet_params.h


def fmt_float(v):
    """Format a float value as a C literal."""
    return f"{v:.8g}"


def fmt_1d(arr):
    """Format 1D array as C initializer: {v0,v1,...}."""
    return "{" + ",".join(fmt_float(v) for v in arr) + "}"


def fmt_2d(arr):
    """Format 2D array as C initializer: {{...},{...},...}."""
    return "{" + ",".join(fmt_1d(row) for row in arr) + "}"


def fmt_3d(arr):
    """Format 3D array as C initializer: {{{...}},...}."""
    return "{" + ",".join(fmt_2d(plane) for plane in arr) + "}"


# ---------------------------------------------------------------------------
# Layer parameter structs (mirrors mobilenet_params.h)
# ---------------------------------------------------------------------------
# These are copied exactly from the existing mobilenet_params.h, with
# output_scale and res_scale set to 1.0f for float mode.

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

# Buffer declarations — same shapes as mobilenet_params.h
# Format: (name, dim1, dim2) or (name, dim1, dim2, dim3) for DW
BUFFERS = [
    # conv_1
    ("conv_1_in",  50176, 27),
    ("conv_1_out", 50176, 32),
    # conv_dw_2
    ("conv_dw_2_out",  50176, 32),
    # conv_3
    ("conv_3_in",  50176, 32),
    ("conv_3_out", 50176, 16),
    # conv_4
    ("conv_4_in",  50176, 16),
    ("conv_4_out", 50176, 96),
    # conv_dw_5
    ("conv_dw_5_out",  12544, 96),
    # conv_6
    ("conv_6_in",  12544, 96),
    ("conv_6_out", 12544, 24),
    # conv_7
    ("conv_7_in",  12544, 24),
    ("conv_7_out", 12544, 144),
    # conv_dw_8
    ("conv_dw_8_out",  12544, 144),
    # conv_9
    ("conv_9_in",  12544, 144),
    ("conv_9_out", 12544, 24),
    # conv_10
    ("conv_10_in",  12544, 24),
    ("conv_10_out", 12544, 144),
    # conv_dw_11
    ("conv_dw_11_out", 3136, 144),
    # conv_12
    ("conv_12_in",  3136, 144),
    ("conv_12_out", 3136, 32),
    # conv_13
    ("conv_13_in",  3136, 32),
    ("conv_13_out", 3136, 192),
    # conv_dw_14
    ("conv_dw_14_out", 3136, 192),
    # conv_15
    ("conv_15_in",  3136, 192),
    ("conv_15_out", 3136, 32),
    # conv_16
    ("conv_16_in",  3136, 32),
    ("conv_16_out", 3136, 192),
    # conv_dw_17
    ("conv_dw_17_out", 3136, 192),
    # conv_18
    ("conv_18_in",  3136, 192),
    ("conv_18_out", 3136, 32),
    # conv_19
    ("conv_19_in",  3136, 32),
    ("conv_19_out", 3136, 192),
    # conv_dw_20
    ("conv_dw_20_out", 784, 192),
    # conv_21
    ("conv_21_in",  784, 192),
    ("conv_21_out", 784, 64),
    # conv_22
    ("conv_22_in",  784, 64),
    ("conv_22_out", 784, 384),
    # conv_dw_23
    ("conv_dw_23_out", 784, 384),
    # conv_24
    ("conv_24_in",  784, 384),
    ("conv_24_out", 784, 64),
    # conv_25
    ("conv_25_in",  784, 64),
    ("conv_25_out", 784, 384),
    # conv_dw_26
    ("conv_dw_26_out", 784, 384),
    # conv_27
    ("conv_27_in",  784, 384),
    ("conv_27_out", 784, 64),
    # conv_28
    ("conv_28_in",  784, 64),
    ("conv_28_out", 784, 384),
    # conv_dw_29
    ("conv_dw_29_out", 784, 384),
    # conv_30
    ("conv_30_in",  784, 384),
    ("conv_30_out", 784, 64),
    # conv_31
    ("conv_31_in",  784, 64),
    ("conv_31_out", 784, 384),
    # conv_dw_32
    ("conv_dw_32_out", 784, 384),
    # conv_33
    ("conv_33_in",  784, 384),
    ("conv_33_out", 784, 96),
    # conv_34
    ("conv_34_in",  784, 96),
    ("conv_34_out", 784, 576),
    # conv_dw_35
    ("conv_dw_35_out", 784, 576),
    # conv_36
    ("conv_36_in",  784, 576),
    ("conv_36_out", 784, 96),
    # conv_37
    ("conv_37_in",  784, 96),
    ("conv_37_out", 784, 576),
    # conv_dw_38
    ("conv_dw_38_out", 784, 576),
    # conv_39
    ("conv_39_in",  784, 576),
    ("conv_39_out", 784, 96),
    # conv_40
    ("conv_40_in",  784, 96),
    ("conv_40_out", 784, 576),
    # conv_dw_41
    ("conv_dw_41_out", 196, 576),
    # conv_42
    ("conv_42_in",  196, 576),
    ("conv_42_out", 196, 160),
    # conv_43
    ("conv_43_in",  196, 160),
    ("conv_43_out", 196, 960),
    # conv_dw_44
    ("conv_dw_44_out", 196, 960),
    # conv_45
    ("conv_45_in",  196, 960),
    ("conv_45_out", 196, 160),
    # conv_46
    ("conv_46_in",  196, 160),
    ("conv_46_out", 196, 960),
    # conv_dw_47
    ("conv_dw_47_out", 196, 960),
    # conv_48
    ("conv_48_in",  196, 960),
    ("conv_48_out", 196, 160),
    # conv_49
    ("conv_49_in",  196, 160),
    ("conv_49_out", 196, 960),
    # conv_dw_50
    ("conv_dw_50_out", 196, 960),
    # conv_51
    ("conv_51_in",  196, 960),
    ("conv_51_out", 196, 320),
    # conv_52
    ("conv_52_in",  196, 320),
    ("conv_52_out", 196, 1280),
    # fc_53
    ("fc_53_out", 1000, 4),
]


# ---------------------------------------------------------------------------
# Main extraction logic
# ---------------------------------------------------------------------------

def get_conv_bn_params(state_dict, prefix):
    """Extract conv weight and BN parameters from a HuggingFace MobileNetV2ConvLayer."""
    conv_w = state_dict[f"{prefix}.convolution.weight"].numpy()
    bn_gamma = state_dict[f"{prefix}.normalization.weight"].numpy()
    bn_beta = state_dict[f"{prefix}.normalization.bias"].numpy()
    bn_mean = state_dict[f"{prefix}.normalization.running_mean"].numpy()
    bn_var = state_dict[f"{prefix}.normalization.running_var"].numpy()
    return conv_w, bn_gamma, bn_beta, bn_mean, bn_var


def extract_layer(state_dict, gemmini_name, hf_prefix, layer_type, num_classes):
    """Extract and transform weights for a single Gemmini layer.

    Returns (weight_array, bias_array) as numpy arrays in Gemmini layout.
    """
    if layer_type == "fc":
        # FC layer — no BatchNorm, direct extraction
        # HuggingFace MobileNetV2 classifier is a bare nn.Linear
        fc_w = state_dict["classifier.weight"].numpy()  # [num_classes, 1280]
        fc_b = state_dict["classifier.bias"].numpy()    # [num_classes]

        if num_classes == 1001:
            # TF convention: index 0 is background class, skip it
            fc_w = fc_w[1:, :]  # [1000, 1280]
            fc_b = fc_b[1:]      # [1000]
        elif num_classes != 1000:
            raise ValueError(f"Unexpected num_classes={num_classes}, expected 1000 or 1001")

        # Gemmini layout: fc_53_w[1000][1280] — same as PyTorch
        # Bias: fc_53_b[1000][4] — replicate across batch dim
        bias_2d = np.tile(fc_b.reshape(-1, 1), (1, BATCH_SIZE))  # [1000, 4]
        return fc_w, bias_2d

    # Conv layer (regular or depthwise) — fold BatchNorm
    conv_w, bn_gamma, bn_beta, bn_mean, bn_var = get_conv_bn_params(state_dict, hf_prefix)
    w_folded, b_folded = fold_bn(conv_w, bn_gamma, bn_beta, bn_mean, bn_var)

    if layer_type == "dw":
        w_gemmini = reshape_dw_weight(w_folded)  # [channels, 3, 3]
    else:
        w_gemmini = reshape_conv_weight(w_folded)  # [patch_size, out_ch]

    return w_gemmini, b_folded


def validate_shapes(gemmini_name, weight, bias, layer_type):
    """Validate that extracted weight shapes match expected Gemmini shapes."""
    if layer_type == "fc":
        assert weight.shape == (1000, 1280), \
            f"{gemmini_name}: weight shape {weight.shape} != (1000, 1280)"
        assert bias.shape == (1000, BATCH_SIZE), \
            f"{gemmini_name}: bias shape {bias.shape} != (1000, {BATCH_SIZE})"
        return

    params = CONV_PARAMS[gemmini_name]
    out_ch = params["out_channels"]

    if layer_type == "dw":
        expected_w = (out_ch, 3, 3)
    else:
        patch_size = params["patch_size"]
        expected_w = (patch_size, out_ch)

    assert weight.shape == expected_w, \
        f"{gemmini_name}: weight shape {weight.shape} != {expected_w}"
    assert bias.shape == (out_ch,), \
        f"{gemmini_name}: bias shape {bias.shape} != ({out_ch},)"


# ---------------------------------------------------------------------------
# Header file generation
# ---------------------------------------------------------------------------

def write_header(output_path, layers_data):
    """Write the complete mobilenet_params_float.h file.

    layers_data: list of (gemmini_name, weight, bias, layer_type) tuples.
    """
    with open(output_path, "w") as f:
        f.write("#ifndef MOBILENET_FLOAT_PARAMETERS_H\n")
        f.write("#define MOBILENET_FLOAT_PARAMETERS_H\n\n")
        f.write("#include <include/gemmini_params.h>\n")
        f.write("#include <stdbool.h>\n\n")

        for gemmini_name, weight, bias, layer_type in layers_data:
            write_layer(f, gemmini_name, weight, bias, layer_type)

        f.write("#endif // MOBILENET_FLOAT_PARAMETERS_H\n")


def write_layer(f, name, weight, bias, layer_type):
    """Write weight, bias, buffer, and params declarations for one layer."""

    if layer_type == "fc":
        write_fc_layer(f, name, weight, bias)
    elif layer_type == "dw":
        write_dw_layer(f, name, weight, bias)
    else:
        write_conv_layer(f, name, weight, bias)

    f.write("\n\n")


def write_conv_layer(f, name, weight, bias):
    """Write a regular (non-depthwise) conv layer."""
    params = CONV_PARAMS[name]
    patch_size = params["patch_size"]
    out_ch = params["out_channels"]

    # Weight: [patch_size][out_ch]
    f.write(f"static const elem_t {name}_w[{patch_size}][{out_ch}] row_align(1) = ")
    f.write(fmt_2d(weight))
    f.write(";\n")

    # Bias: [out_ch]
    f.write(f"static const acc_t {name}_b[{out_ch}] row_align_acc(1) = ")
    f.write(fmt_1d(bias))
    f.write(";\n")

    # Buffers
    for buf in BUFFERS:
        if buf[0].startswith(name + "_"):
            f.write(f"static elem_t {buf[0]}[{buf[1]}][{buf[2]}] row_align(1);\n")

    # Params struct
    p = params
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
    """Write a depthwise conv layer."""
    params = CONV_PARAMS[name]
    out_ch = params["out_channels"]

    # Weight: [channels][3][3]
    f.write(f"static const elem_t {name}_w[{out_ch}][3][3] row_align(1) = ")
    f.write(fmt_3d(weight))
    f.write(";\n")

    # Bias: [out_ch]
    f.write(f"static const acc_t {name}_b[{out_ch}] row_align_acc(1) = ")
    f.write(fmt_1d(bias))
    f.write(";\n")

    # Buffer (DW layers only have _out buffer)
    for buf in BUFFERS:
        if buf[0] == f"{name}_out":
            f.write(f"static elem_t {buf[0]}[{buf[1]}][{buf[2]}] row_align(1);\n")

    # Params struct (no K field for depthwise)
    p = params
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
    """Write the FC classifier layer."""
    p = FC_PARAMS[name]
    out_f = p["out_features"]
    in_f = p["in_features"]

    # Weight: [out_features][in_features]
    f.write(f"static const elem_t {name}_w[{out_f}][{in_f}] row_align(1) = ")
    f.write(fmt_2d(weight))
    f.write(";\n")

    # Bias: [out_features][batch_size] — replicated across batch dim
    f.write(f"static const acc_t {name}_b[{out_f}][{BATCH_SIZE}] row_align_acc(1) = ")
    f.write(fmt_2d(bias))
    f.write(";\n")

    # Buffer
    f.write(f"static elem_t {name}_out[{out_f}][{BATCH_SIZE}] row_align(1);\n")

    # Params struct
    f.write(f"static const struct FcParams {name}_params = {{")
    f.write(f".batch_size={p['batch_size']}, ")
    f.write(f".in_features={in_f}, .out_features={out_f}, ")
    f.write(f".bias={p['bias']}, ")
    f.write(f".output_scale=1.0f, ")
    f.write(f".I={p['I']}, .J={p['J']}, .K={p['K']}")
    f.write("};\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("Loading google/mobilenet_v2_1.0_224 from HuggingFace...")
    model = MobileNetV2ForImageClassification.from_pretrained("google/mobilenet_v2_1.0_224")
    model.eval()
    state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    # Determine number of output classes
    classifier_weight = state_dict["classifier.weight"]
    num_classes = classifier_weight.shape[0]
    print(f"Model has {num_classes} output classes")
    if num_classes == 1001:
        print("  -> Stripping background class (index 0) to get 1000 classes")

    # Print all state dict keys for verification
    print("\nState dict keys:")
    for k in sorted(state_dict.keys()):
        print(f"  {k}: {list(state_dict[k].shape)}")

    # Build mapping and extract all layers
    mapping = build_layer_mapping()
    layers_data = []

    print(f"\nExtracting {len(mapping)} layers...")
    for gemmini_name, hf_prefix, layer_type in mapping:
        print(f"  {gemmini_name:15s} <- {hf_prefix}")
        weight, bias = extract_layer(state_dict, gemmini_name, hf_prefix, layer_type, num_classes)
        validate_shapes(gemmini_name, weight, bias, layer_type)
        layers_data.append((gemmini_name, weight, bias, layer_type))

    # Write header
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, "..", "mobilenet_params_float.h")
    output_path = os.path.normpath(output_path)

    print(f"\nWriting {output_path}...")
    write_header(output_path, layers_data)
    print("Done!")

    # Print summary statistics
    total_params = 0
    for name, w, b, lt in layers_data:
        total_params += w.size + b.size
    print(f"\nTotal parameters: {total_params:,}")


if __name__ == "__main__":
    main()
