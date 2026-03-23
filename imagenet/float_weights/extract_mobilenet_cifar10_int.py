#!/usr/bin/env python3
"""
Extract INT8-quantized weights from HuggingFace MobileNetV2 fine-tuned on
CIFAR-10 (jialicheng/cifar10_mobilenet-v2) and generate:
  - ../mobilenet_cifar10_params.h  (INT8 weights, INT32 biases, power-of-2 output_scale)
  - ../cifar10_images.h            (4 sample CIFAR-10 test images, resized to 224x224)

Quantization approach:
  - Per-layer symmetric weight quantization: w_int8 = clip(round(w_float / scale), -128, 127)
  - INT32 biases: b_int32 = round(b_float / (w_scale * x_scale))
  - Power-of-2 output_scale: output_scale = w_scale * x_scale / y_scale
  - Activation range estimated from BatchNorm running statistics (gamma, beta)
  - Scale propagation tracks x_scale through the network layer-by-layer

Usage:
    pip install torch transformers torchvision numpy
    python extract_mobilenet_cifar10_int.py
"""

import os
import math
import numpy as np
import torch
from transformers import MobileNetV2ForImageClassification

MODEL_NAME = "jialicheng/cifar10_mobilenet-v2"
INPUT_DIM = 224   # Model was fine-tuned from 224x224 ImageNet MobileNetV2
BATCH_SIZE = 4
NUM_CLASSES = 10

# ---------------------------------------------------------------------------
# MobileNetV2 architecture definition
# ---------------------------------------------------------------------------
# (name, kernel, in_ch, out_ch, stride, padding, depthwise, activation)
# fmt: off
LAYER_ARCH = [
    ("conv_1",     3,   3,   32, 2, 1, False, "relu"),
    ("conv_dw_2",  3,  32,   32, 1, 1,  True, "relu"),
    ("conv_3",     1,  32,   16, 1, 0, False, "none"),
    ("conv_4",     1,  16,   96, 1, 0, False, "relu"),
    ("conv_dw_5",  3,  96,   96, 2, 1,  True, "relu"),
    ("conv_6",     1,  96,   24, 1, 0, False, "none"),
    ("conv_7",     1,  24,  144, 1, 0, False, "relu"),
    ("conv_dw_8",  3, 144,  144, 1, 1,  True, "relu"),
    ("conv_9",     1, 144,   24, 1, 0, False, "none"),
    ("conv_10",    1,  24,  144, 1, 0, False, "relu"),
    ("conv_dw_11", 3, 144,  144, 2, 1,  True, "relu"),
    ("conv_12",    1, 144,   32, 1, 0, False, "none"),
    ("conv_13",    1,  32,  192, 1, 0, False, "relu"),
    ("conv_dw_14", 3, 192,  192, 1, 1,  True, "relu"),
    ("conv_15",    1, 192,   32, 1, 0, False, "none"),
    ("conv_16",    1,  32,  192, 1, 0, False, "relu"),
    ("conv_dw_17", 3, 192,  192, 1, 1,  True, "relu"),
    ("conv_18",    1, 192,   32, 1, 0, False, "none"),
    ("conv_19",    1,  32,  192, 1, 0, False, "relu"),
    ("conv_dw_20", 3, 192,  192, 2, 1,  True, "relu"),
    ("conv_21",    1, 192,   64, 1, 0, False, "none"),
    ("conv_22",    1,  64,  384, 1, 0, False, "relu"),
    ("conv_dw_23", 3, 384,  384, 1, 1,  True, "relu"),
    ("conv_24",    1, 384,   64, 1, 0, False, "none"),
    ("conv_25",    1,  64,  384, 1, 0, False, "relu"),
    ("conv_dw_26", 3, 384,  384, 1, 1,  True, "relu"),
    ("conv_27",    1, 384,   64, 1, 0, False, "none"),
    ("conv_28",    1,  64,  384, 1, 0, False, "relu"),
    ("conv_dw_29", 3, 384,  384, 1, 1,  True, "relu"),
    ("conv_30",    1, 384,   64, 1, 0, False, "none"),
    ("conv_31",    1,  64,  384, 1, 0, False, "relu"),
    ("conv_dw_32", 3, 384,  384, 1, 1,  True, "relu"),
    ("conv_33",    1, 384,   96, 1, 0, False, "none"),
    ("conv_34",    1,  96,  576, 1, 0, False, "relu"),
    ("conv_dw_35", 3, 576,  576, 1, 1,  True, "relu"),
    ("conv_36",    1, 576,   96, 1, 0, False, "none"),
    ("conv_37",    1,  96,  576, 1, 0, False, "relu"),
    ("conv_dw_38", 3, 576,  576, 1, 1,  True, "relu"),
    ("conv_39",    1, 576,   96, 1, 0, False, "none"),
    ("conv_40",    1,  96,  576, 1, 0, False, "relu"),
    ("conv_dw_41", 3, 576,  576, 2, 1,  True, "relu"),
    ("conv_42",    1, 576,  160, 1, 0, False, "none"),
    ("conv_43",    1, 160,  960, 1, 0, False, "relu"),
    ("conv_dw_44", 3, 960,  960, 1, 1,  True, "relu"),
    ("conv_45",    1, 960,  160, 1, 0, False, "none"),
    ("conv_46",    1, 160,  960, 1, 0, False, "relu"),
    ("conv_dw_47", 3, 960,  960, 1, 1,  True, "relu"),
    ("conv_48",    1, 960,  160, 1, 0, False, "none"),
    ("conv_49",    1, 160,  960, 1, 0, False, "relu"),
    ("conv_dw_50", 3, 960,  960, 1, 1,  True, "relu"),
    ("conv_51",    1, 960,  320, 1, 0, False, "none"),
    ("conv_52",    1, 320, 1280, 1, 0, False, "relu"),
]
# fmt: on

FC_PARAMS = {
    "fc_53": {
        "batch_size": BATCH_SIZE,
        "in_features": 1280,
        "out_features": NUM_CLASSES,
        "bias": 1,
        "I": NUM_CLASSES,
        "J": BATCH_SIZE,
        "K": 1280,
    }
}


# ---------------------------------------------------------------------------
# Compute CIFAR-10 spatial dimensions
# ---------------------------------------------------------------------------

def compute_conv_params():
    params = {}
    current_dim = INPUT_DIM
    for name, kernel, in_ch, out_ch, stride, padding, dw, _act in LAYER_ARCH:
        out_dim = (current_dim + 2 * padding - kernel) // stride + 1
        n_patches = BATCH_SIZE * out_dim * out_dim
        patch_size = kernel * kernel if dw else in_ch * kernel * kernel
        p = {
            "batch_size": BATCH_SIZE,
            "in_row_dim": current_dim, "in_col_dim": current_dim,
            "kernel_size": kernel,
            "in_channels": in_ch, "out_channels": out_ch,
            "stride": stride, "padding": padding,
            "bias": 1, "depthwise": 1 if dw else 0,
            "out_row_dim": out_dim, "out_col_dim": out_dim,
            "n_patches": n_patches, "patch_size": patch_size,
            "pool_size": 1, "pool_stride": 1, "pool_padding": 0,
            "out_dim_pooled": out_dim,
            "I": n_patches, "J": out_ch,
        }
        if not dw:
            p["K"] = patch_size
        params[name] = p
        current_dim = out_dim
    return params


def compute_buffers(conv_params):
    buffers = []
    for name, _k, _ic, _oc, _s, _p, dw, _a in LAYER_ARCH:
        p = conv_params[name]
        if not dw:
            buffers.append((f"{name}_in", p["n_patches"], p["patch_size"]))
        buffers.append((f"{name}_out", p["n_patches"], p["out_channels"]))
    buffers.append(("fc_53_out", NUM_CLASSES, BATCH_SIZE))
    return buffers


# ---------------------------------------------------------------------------
# Layer mapping
# ---------------------------------------------------------------------------

def build_layer_mapping():
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
# BatchNorm folding
# ---------------------------------------------------------------------------

# Channels where BN running_var is below this are treated as 'dead'
# (near-zero variance ⇒ 1/sqrt(var) blows up ⇒ numerically unstable folded
# weights that inflate w_scale and in turn inflate bias INT32 values,
# saturating virtually all INT8 outputs).
_BN_DEAD_VAR_THRESHOLD = 1e-3

# After dead-channel zeroing, if any surviving channel's max |weight| is
# more than this factor above the median of surviving channels, clip that
# channel's weights at the threshold.  Handles genuinely-active channels
# that have a pathologically-large BN γ from fine-tuning, without losing
# the channel's feature entirely.
_OUTLIER_WEIGHT_RATIO = 5.0


def fold_bn(conv_weight, bn_weight, bn_bias, bn_mean, bn_var, eps=1e-5):
    """Fold BN into conv weights, with dead-channel + outlier stabilisation.

    Two-stage numerical clean-up:
    1. Dead channels (running_var < _BN_DEAD_VAR_THRESHOLD): 1/sqrt(var)
       blows up, producing huge folded weights.  Zero the weight and set the
       folded bias to bn_bias (the channel's near-constant output ≈ β).
    2. Outlier channels (live but max |w| > _OUTLIER_WEIGHT_RATIO × median of
       live channels): caused by pathologically large BN γ from fine-tuning.
       Clip the weight channel-wise to the outlier threshold.  The bias is
       kept as-is (depends on BN β/γ/mean/var, not the conv weight).
    """
    inv_std = 1.0 / np.sqrt(bn_var + eps)
    scale = bn_weight * inv_std
    shape = [conv_weight.shape[0]] + [1] * (conv_weight.ndim - 1)
    w_folded = conv_weight * scale.reshape(shape)
    b_folded = bn_bias - bn_weight * bn_mean * inv_std

    # Stage 1 – dead channels
    dead = bn_var < _BN_DEAD_VAR_THRESHOLD
    if np.any(dead):
        w_folded[dead] = 0.0
        b_folded[dead] = bn_bias[dead]

    # Stage 2 – outlier clipping
    ch_max = np.max(np.abs(w_folded.reshape(w_folded.shape[0], -1)), axis=1)
    live = ch_max > 0
    if np.sum(live) > 1:
        median_max = np.median(ch_max[live])
        if median_max > 0:
            clip_val = _OUTLIER_WEIGHT_RATIO * median_max
            outlier_idx = np.where((ch_max > clip_val) & live)[0]
            for c in outlier_idx:
                w_folded[c] = np.clip(w_folded[c], -clip_val, clip_val)

    return w_folded, b_folded


# ---------------------------------------------------------------------------
# Weight reshaping
# ---------------------------------------------------------------------------

def reshape_conv_weight(w):
    out_ch = w.shape[0]
    if w.ndim == 4:
        # PyTorch: [out_ch, C, kH, kW] -> Gemmini NHWC: [out_ch, kH, kW, C]
        w = w.transpose(0, 2, 3, 1)
    return w.reshape(out_ch, -1).T

def reshape_dw_weight(w):
    assert w.shape[1] == 1
    return w.squeeze(1)


# ---------------------------------------------------------------------------
# INT8 quantization
# ---------------------------------------------------------------------------

def quantize_weight_int8(w_float):
    """Per-layer symmetric INT8 quantization."""
    w_abs_max = np.max(np.abs(w_float))
    if w_abs_max < 1e-10:
        return np.zeros_like(w_float, dtype=np.int8), 1e-10
    w_scale = w_abs_max / 127.0
    w_int = np.clip(np.round(w_float / w_scale), -128, 127).astype(np.int8)
    return w_int, w_scale


def quantize_bias_int32(b_float, combined_scale):
    """Quantize bias to INT32 using combined (w_scale * x_scale)."""
    if combined_scale < 1e-10:
        return np.zeros_like(b_float, dtype=np.int32)
    b_int = np.clip(
        np.round(b_float / combined_scale), -(2**31), 2**31 - 1
    ).astype(np.int32)
    return b_int


def compute_output_scale(w_scale, x_scale, y_range):
    """Compute output_scale as an exact float.

    output_scale = w_scale * x_scale / y_scale  (where y_scale = y_range/127)
    Using an exact float avoids the precision loss (and the ≤1.0 cap) of the
    old power-of-2 rounding, which caused saturation for layers with large
    BN-folded weights.
    """
    y_scale = y_range / 127.0
    raw = (w_scale * x_scale) / y_scale
    if raw <= 0 or not np.isfinite(raw):
        raw = 1.0
    return raw, f"{raw:.8e}f"


def estimate_activation_range(bn_gamma, bn_beta, has_relu):
    """Estimate output activation range from BN gamma/beta (3-sigma rule).

    For ReLU layers, MobileNetV2 uses ReLU6 which hard-clamps activations to
    [0, 6].  The raw BN 3-sigma estimate can exceed 6 (because it does not
    account for ReLU6 saturation), which propagates an inflated x_scale to
    downstream layers and causes output_scale > 1.  Cap at 6.0 for correctness.
    """
    gamma_max = np.max(np.abs(bn_gamma))
    beta_max = np.max(np.abs(bn_beta))
    if has_relu:
        # ReLU clips negatives; approximate positive range
        y_range = max(float(np.max(bn_beta + 3.0 * np.abs(bn_gamma))), 1.0)
        y_range = min(y_range, 6.0)   # ReLU6 hard upper bound
    else:
        y_range = max(beta_max + 3.0 * gamma_max, 1.0)
    return y_range


def quantize_depthwise_weight_int8(w_float, percentile=65.0):
    """Robust INT8 quantization for depthwise conv weights.

    Depthwise conv layers can contain dead BN channels (near-zero running
    variance) whose BN-folded weights are orders of magnitude larger than
    those of normal channels.  When we use the global max for w_scale these
    outliers inflate it so severely that ALL normal channels lose precision
    and output_scale >> 1 (causing INT8 saturation for most activations).

    Fix: derive w_scale from the 65th-percentile of per-channel max-abs
    values.  Dead channels (the top ~19-31% outliers) are allowed to
    saturate to +/-127 INT8; their downstream 1x1 projection weights are
    near-zero because the network learned to ignore these channels.
    """
    out_ch = w_float.shape[0]
    per_ch_max = np.max(np.abs(w_float.reshape(out_ch, -1)), axis=1)
    w_scale_base = max(float(np.percentile(per_ch_max, percentile)), 1e-10)
    w_scale = w_scale_base / 127.0
    w_int = np.clip(np.round(w_float / w_scale), -128, 127).astype(np.int8)
    return w_int, w_scale


# ---------------------------------------------------------------------------
# C formatting (int)
# ---------------------------------------------------------------------------

def fmt_int(v):
    return str(int(v))

def fmt_int_1d(arr):
    return "{" + ",".join(fmt_int(v) for v in arr) + "}"

def fmt_int_2d(arr):
    return "{" + ",".join(fmt_int_1d(row) for row in arr) + "}"

def fmt_int_3d(arr):
    return "{" + ",".join(fmt_int_2d(plane) for plane in arr) + "}"


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

def get_conv_bn_params(state_dict, prefix):
    conv_w = state_dict[f"{prefix}.convolution.weight"].numpy()
    bn_gamma = state_dict[f"{prefix}.normalization.weight"].numpy()
    bn_beta = state_dict[f"{prefix}.normalization.bias"].numpy()
    bn_mean = state_dict[f"{prefix}.normalization.running_mean"].numpy()
    bn_var = state_dict[f"{prefix}.normalization.running_var"].numpy()
    return conv_w, bn_gamma, bn_beta, bn_mean, bn_var


def extract_and_fold(state_dict, hf_prefix, layer_type):
    """Extract conv+BN weights, fold BN, reshape."""
    conv_w, bn_gamma, bn_beta, bn_mean, bn_var = get_conv_bn_params(
        state_dict, hf_prefix
    )
    w_folded, b_folded = fold_bn(conv_w, bn_gamma, bn_beta, bn_mean, bn_var)
    if layer_type == "dw":
        return reshape_dw_weight(w_folded), b_folded, bn_gamma, bn_beta
    return reshape_conv_weight(w_folded), b_folded, bn_gamma, bn_beta


# ---------------------------------------------------------------------------
# Header generation
# ---------------------------------------------------------------------------

def write_header(output_path, layers_data, conv_params, buffers):
    with open(output_path, "w") as f:
        f.write("#ifndef MOBILENET_CIFAR10_PARAMETERS_H\n")
        f.write("#define MOBILENET_CIFAR10_PARAMETERS_H\n\n")
        f.write("#include <include/gemmini_params.h>\n")
        f.write("#include <stdbool.h>\n\n")

        for entry in layers_data:
            name = entry["name"]
            lt = entry["layer_type"]
            if lt == "fc":
                write_fc_layer(f, entry)
            elif lt == "dw":
                write_dw_layer(f, entry, conv_params, buffers)
            else:
                write_conv_layer(f, entry, conv_params, buffers)
            f.write("\n\n")

        f.write("#endif // MOBILENET_CIFAR10_PARAMETERS_H\n")


def write_conv_layer(f, entry, conv_params, buffers):
    name = entry["name"]
    w_int = entry["weight"]
    b_int = entry["bias"]
    output_scale_str = entry["output_scale_str"]
    res_scale = entry.get("res_scale", 1.0)
    p = conv_params[name]

    f.write(f"static const elem_t {name}_w[{p['patch_size']}][{p['out_channels']}] row_align(1) = ")
    f.write(fmt_int_2d(w_int))
    f.write(";\n")
    f.write(f"static const acc_t {name}_b[{p['out_channels']}] row_align_acc(1) = ")
    f.write(fmt_int_1d(b_int))
    f.write(";\n")
    for buf in buffers:
        if buf[0].startswith(name + "_"):
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
    f.write(f".output_scale={output_scale_str}, ")
    f.write(f".I={p['I']}, .J={p['J']}, .K={p['K']}, ")
    f.write(f".res_scale={res_scale:.8e}f")
    f.write("};\n")


def write_dw_layer(f, entry, conv_params, buffers):
    name = entry["name"]
    w_int = entry["weight"]
    b_int = entry["bias"]
    output_scale_str = entry["output_scale_str"]
    res_scale = entry.get("res_scale", 1.0)
    p = conv_params[name]

    f.write(f"static const elem_t {name}_w[{p['out_channels']}][3][3] row_align(1) = ")
    f.write(fmt_int_3d(w_int))
    f.write(";\n")
    f.write(f"static const acc_t {name}_b[{p['out_channels']}] row_align_acc(1) = ")
    f.write(fmt_int_1d(b_int))
    f.write(";\n")
    for buf in buffers:
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
    f.write(f".output_scale={output_scale_str}, ")
    f.write(f".res_scale={res_scale:.8e}f, ")
    f.write(f".I={p['I']}, .J={p['J']}")
    f.write("};\n")


def write_fc_layer(f, entry):
    name = entry["name"]
    w_int = entry["weight"]
    b_int = entry["bias"]
    output_scale_str = entry["output_scale_str"]
    p = FC_PARAMS[name]
    out_f = p["out_features"]
    in_f = p["in_features"]

    f.write(f"static const elem_t {name}_w[{out_f}][{in_f}] row_align(1) = ")
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
# CIFAR-10 image generation
# ---------------------------------------------------------------------------

def generate_cifar10_images(output_path, indices=None, img_mean=None, img_std=None):
    """Generate cifar10_images.h with 4 sample CIFAR-10 test images.

    Images are normalised with the model's own mean/std and stored as INT8
    in the range [-127, 127] so that x_scale = max_abs_norm / 127.
    """
    if img_mean is None:
        img_mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    if img_std is None:
        img_std  = np.array([0.5, 0.5, 0.5], dtype=np.float32)

    if indices is None:
        indices = [0, 1, 2, 3]

    # Try torchvision first, then HuggingFace datasets as fallback
    def _make_getter():
        try:
            from torchvision.datasets import CIFAR10
            dataset = CIFAR10(root="/tmp/cifar10_data", train=False, download=True)
            def _get(idx):
                return dataset[idx]
            return _get
        except Exception:
            pass
        try:
            from datasets import load_dataset as _hf_load
            _hf_ds = _hf_load("uoft-cs/cifar10", split="test", trust_remote_code=False)
            def _get(idx):
                item = _hf_ds[idx]
                return item["img"], item["label"]
            return _get
        except Exception:
            pass
        return None

    _get_item = _make_getter()
    if _get_item is None:
        print("WARNING: Could not load CIFAR-10 (need torchvision or datasets).")
        print("  pip install torchvision   or   pip install datasets")
        return None

    from PIL import Image as PILImage
    images = []
    labels = []
    for idx in indices:
        img_pil, label = _get_item(idx)
        if INPUT_DIM != 32:
            img_pil = img_pil.resize((INPUT_DIM, INPUT_DIM), PILImage.BILINEAR)
        img_np = np.array(img_pil, dtype=np.float32)
        # Normalise to the same float range the model was trained on
        img_float = (img_np / 255.0 - img_mean) / img_std          # shape [H,W,3]
        # Compute x_scale from worst-case range across all channels
        f_min = float(np.min((0.0   / 255.0 - img_mean) / img_std))
        f_max = float(np.max((255.0 / 255.0 - img_mean) / img_std))
        x_sc  = max(abs(f_min), abs(f_max)) / 127.0
        img_int = np.clip(np.round(img_float / x_sc), -127, 127).astype(np.int8)
        images.append(img_int)
        labels.append(label)

    with open(output_path, "w") as f:
        f.write("#ifndef CIFAR10_IMAGES_224_H\n")
        f.write("#define CIFAR10_IMAGES_224_H\n\n")
        f.write("#include <include/gemmini_params.h>\n\n")
        f.write(f"// CIFAR-10 test images at indices {indices}\n")
        cifar10_classes = ["airplane", "automobile", "bird", "cat", "deer",
                           "dog", "frog", "horse", "ship", "truck"]
        label_names = [cifar10_classes[l] for l in labels]
        f.write(f"// Labels: {labels} ({label_names})\n")
        f.write(f"// Use in C:  int correct[] = {{{', '.join(str(l) for l in labels)}}};\n\n")
        f.write(f"static const elem_t images[{BATCH_SIZE}][{INPUT_DIM}][{INPUT_DIM}][3] row_align(1) = {{")
        for b in range(len(images)):
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
                    vals = ",".join(str(int(images[b][r][c][ch])) for ch in range(3))
                    f.write("{" + vals + "}")
                f.write("}")
            f.write("}")
        f.write("};\n\n")
        f.write("#endif // CIFAR10_IMAGES_224_H\n")

    print(f"  Written {output_path}")
    print(f"  Labels: {labels} ({label_names})")
    print(f"  int correct[] = {{{', '.join(str(l) for l in labels)}}};")
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
    actual_classes = classifier_w.shape[0]
    print(f"Model has {actual_classes} output classes")
    assert actual_classes == NUM_CLASSES, \
        f"Expected {NUM_CLASSES}, got {actual_classes}"

    # Compute CIFAR-10 spatial params
    conv_params = compute_conv_params()
    buffers = compute_buffers(conv_params)

    print(f"\nSpatial dimensions for {INPUT_DIM}x{INPUT_DIM} input:")
    for name, _k, _ic, _oc, _s, _p, _dw, _a in LAYER_ARCH:
        p = conv_params[name]
        print(f"  {name:15s}  {p['in_row_dim']:3d}x{p['in_col_dim']:<3d} -> "
              f"{p['out_row_dim']:3d}x{p['out_col_dim']:<3d}  "
              f"n_patches={p['n_patches']:5d}")

    # Build layer mapping
    mapping = build_layer_mapping()
    # Build name -> arch lookup
    arch_lookup = {name: (name, k, ic, oc, s, pad, dw, act)
                   for name, k, ic, oc, s, pad, dw, act in LAYER_ARCH}

    # -----------------------------------------------------------------------
    # Determine input normalization from the model's image processor
    # -----------------------------------------------------------------------
    try:
        from transformers import AutoImageProcessor
        processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
        img_mean = np.array(processor.image_mean, dtype=np.float32)  # e.g. [0.5,0.5,0.5]
        img_std  = np.array(processor.image_std,  dtype=np.float32)  # e.g. [0.5,0.5,0.5]
        print(f"Image processor normalization: mean={img_mean}, std={img_std}")
    except Exception:
        # Fall back to safe defaults (MobileNetV2 HuggingFace default)
        img_mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        img_std  = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        print("WARNING: Could not load image processor; using mean=0.5, std=0.5")

    # Each pixel p in [0,255] → normalised float f = (p/255 - mean) / std ≈ [-1, +1]
    # We store INT8 as x_int = clip(round(f * 127), -127, 127), so x_scale = 1/127.
    # Using the worst-case range across all three channels:
    f_min = float(np.min((0.0   / 255.0 - img_mean) / img_std))
    f_max = float(np.max((255.0 / 255.0 - img_mean) / img_std))
    x_scale = max(abs(f_min), abs(f_max)) / 127.0
    print(f"Input float range: [{f_min:.4f}, {f_max:.4f}]  →  x_scale={x_scale:.6f}")

    # -----------------------------------------------------------------------
    # Residual-connection skip-source map
    #   key   = gemmini reduce layer that receives the addition
    #   value = gemmini reduce layer whose output is the skip tensor
    # Blocks are identified by: stride==1 AND in_channels==out_channels for
    # the entire inverted-residual triple (expand→dw→reduce).
    # -----------------------------------------------------------------------
    RESIDUAL_SKIP = {
        "conv_9":  "conv_6",
        "conv_15": "conv_12",
        "conv_18": "conv_15",
        "conv_24": "conv_21",
        "conv_27": "conv_24",
        "conv_30": "conv_27",
        "conv_36": "conv_33",
        "conv_39": "conv_36",
        "conv_45": "conv_42",
        "conv_48": "conv_45",
    }

    # -----------------------------------------------------------------------
    # Extract & quantize all layers with scale propagation
    # -----------------------------------------------------------------------
    layers_data = []
    layer_y_range = {}   # gemmini_name → y_range, filled as we go

    print(f"\nExtracting and quantizing {len(mapping)} layers...")
    for gemmini_name, hf_prefix, layer_type in mapping:
        if layer_type == "fc":
            # --- FC layer ---
            fc_w_float = state_dict["classifier.weight"].numpy()
            fc_b_float = state_dict["classifier.bias"].numpy()

            w_int, w_scale = quantize_weight_int8(fc_w_float)
            combined_scale = w_scale * x_scale
            b_int_1d = quantize_bias_int32(fc_b_float, combined_scale)
            # Replicate bias across batch dimension
            b_int_2d = np.tile(b_int_1d.reshape(-1, 1), (1, BATCH_SIZE))

            # FC output range: rough estimate from weight statistics
            fc_y_range = max(
                float(np.max(np.abs(fc_b_float))
                      + np.max(np.abs(fc_w_float)) * np.sqrt(1280) * x_scale * 10),
                1.0
            )
            _os, output_scale_str = compute_output_scale(w_scale, x_scale, fc_y_range)
            layer_y_range[gemmini_name] = fc_y_range

            print(f"  {gemmini_name:15s}  w_scale={w_scale:.6f}  x_scale={x_scale:.6f}  "
                  f"y_range={fc_y_range:.2f}  output_scale={_os:.4e}")

            layers_data.append({
                "name": gemmini_name,
                "layer_type": "fc",
                "weight": w_int,
                "bias": b_int_2d,
                "output_scale_str": output_scale_str,
            })
        else:
            # --- Conv / DW layer ---
            w_float, b_float, bn_gamma, bn_beta = extract_and_fold(
                state_dict, hf_prefix, layer_type
            )

            # Determine activation type
            arch_info = arch_lookup.get(gemmini_name)
            has_relu = arch_info is not None and arch_info[7] == "relu"

            # Quantize weights: now that fold_bn zeroes dead-channel weights
            # the global max is stable for both DW and regular conv layers.
            w_int, w_scale = quantize_weight_int8(w_float)

            # Quantize bias: b_int = round(b_float / (w_scale * x_scale))
            combined_scale = w_scale * x_scale
            b_int = quantize_bias_int32(b_float, combined_scale)

            # Estimate output activation range from BN parameters
            y_range = estimate_activation_range(
                bn_gamma.numpy() if isinstance(bn_gamma, torch.Tensor) else bn_gamma,
                bn_beta.numpy() if isinstance(bn_beta, torch.Tensor) else bn_beta,
                has_relu
            )
            layer_y_range[gemmini_name] = y_range

            _os, output_scale_str = compute_output_scale(w_scale, x_scale, y_range)

            # --- Residual-connection res_scale ---
            # res_scale rescales the SKIP tensor so it is in the same units as
            # this layer's output before the addition:
            #   res_scale = y_scale_skip / y_scale_main
            #             = y_range_skip / y_range_main
            res_scale = 1.0
            if gemmini_name in RESIDUAL_SKIP:
                skip_src = RESIDUAL_SKIP[gemmini_name]
                if skip_src in layer_y_range:
                    res_scale = layer_y_range[skip_src] / y_range
                else:
                    print(f"  WARNING: skip source {skip_src} not yet processed for {gemmini_name}")

            print(f"  {gemmini_name:15s}  w_scale={w_scale:.6f}  x_scale={x_scale:.6f}  "
                  f"y_range={y_range:.2f}  output_scale={_os:.4e}  "
                  f"res_scale={res_scale:.4f}  "
                  f"{'RELU' if has_relu else 'LINEAR'}")

            layers_data.append({
                "name": gemmini_name,
                "layer_type": layer_type,
                "weight": w_int,
                "bias": b_int,
                "output_scale_str": output_scale_str,
                "res_scale": res_scale,
            })

            # Propagate: next layer's input scale = this layer's output scale
            y_scale = y_range / 127.0
            x_scale = y_scale

    # -----------------------------------------------------------------------
    # Write params header
    # -----------------------------------------------------------------------
    output_dir = os.path.dirname(os.path.abspath(__file__))
    params_path = os.path.join(output_dir, "..", "mobilenet_cifar10_params.h")
    params_path = os.path.normpath(params_path)
    print(f"\nWriting {params_path}...")
    write_header(params_path, layers_data, conv_params, buffers)

    # -----------------------------------------------------------------------
    # Generate CIFAR-10 images
    # -----------------------------------------------------------------------
    images_path = os.path.join(output_dir, "..", "cifar10_images_224.h")
    images_path = os.path.normpath(images_path)
    print(f"\nGenerating {images_path}...")
    labels = generate_cifar10_images(images_path, img_mean=img_mean, img_std=img_std)

    # Summary
    total_params = sum(e["weight"].size + e["bias"].size for e in layers_data)
    print(f"\nTotal parameters: {total_params:,}")
    print("Done!")


if __name__ == "__main__":
    main()
