#!/usr/bin/env python3
"""
Extract INT8-quantized weights from HuggingFace MobileNetV2 (google/mobilenet_v2_1.0_224)
and generate:
  - ../mobilenet_params.h  (INT8 weights, INT32 biases, exact float output_scale)

Architecture: MobileNetV2 (inverted residual blocks)
  - Stem: 3x3 conv stride 2 (3->32) + 3x3 dw (32) + 1x1 reduce (32->16)
  - 16 inverted residual blocks: expand_1x1 + depthwise_3x3 + reduce_1x1
  - Final 1x1 conv (320->1280) + global avg pool + FC (1280->1000)

  Residual connections (identity, NO projection shortcuts):
    conv_9 += conv_6,   conv_15 += conv_12,  conv_18 += conv_15
    conv_24 += conv_21, conv_27 += conv_24,  conv_30 += conv_27
    conv_36 += conv_33, conv_39 += conv_36,  conv_45 += conv_42
    conv_48 += conv_45

Key differences from ResNet-50:
  - BN eps = 0.001 (not 1e-5)
  - HuggingFace model has 1001 classes (background + 1000); must skip class 0
  - Depthwise conv layers: weight shape [ch][3][3]
  - 1x1 pointwise convs use tiled_matmul_nn_auto: weight shape [in_ch][out_ch]
  - FC layout: fc_53_w[1000][1280]  (NOT transposed like ResNet-50)
  - FC bias:  fc_53_b[1000][4]      (replicated across batch dim)
  - Global avg pool: average[1280][4] = [channels][batch]
  - No ReLU after resadd (all resadds have relu=false)
  - Reduce layers (projection) have NO_ACTIVATION
  - Expand and depthwise layers have RELU

Quantization:
  - Per-layer symmetric weight quantization: w_int8 = clip(round(w/scale), -128, 127)
  - INT32 biases: b_int32 = round(b_float / (w_scale * x_scale))
  - Exact float output_scale: output_scale = w_scale * x_scale / y_scale
  - Activation range from BatchNorm running statistics (3-sigma rule)
  - Scale propagation tracks x_scale sequentially through the network
  - res_scale = y_range[skip_src] / y_range[this_layer]

Usage:
    cd imagenet/float_weights
    conda run -n ImageNet python extract_mobilenet_imagenet_int8.py
    conda run -n ImageNet python extract_mobilenet_imagenet_int8.py --pixel-minus-128

Output:
    ../mobilenet_params.h
"""

import os
import argparse
import math
import numpy as np
import torch
from transformers import MobileNetV2ForImageClassification

MODEL_NAME = "google/mobilenet_v2_1.0_224"
INPUT_DIM = 224
BATCH_SIZE = 4
BN_EPS = 0.001  # MobileNetV2 uses eps=0.001, NOT 1e-5

# MobileNetV2 uses [-1, 1] normalization (mean=0.5, std=0.5),
# NOT the standard ImageNet mean/std used by ResNet.
# AutoImageProcessor: image_mean=[0.5,0.5,0.5], image_std=[0.5,0.5,0.5]
IMAGENET_MEAN = [0.5, 0.5, 0.5]  # RGB  (MobileNetV2 specific)
IMAGENET_STD  = [0.5, 0.5, 0.5]  # RGB  (MobileNetV2 specific)

# ---------------------------------------------------------------------------
# Architecture definition
# ---------------------------------------------------------------------------
# (gemmini_name, kernel, in_ch, out_ch, stride, padding, is_depthwise, activation)
# activation: "relu" for ReLU/ReLU6, "none" for linear (reduce/project layers)
LAYER_ARCH = [
    ("conv_1",     3,   3,   32, 2, 1, False, "relu"),   # first_conv
    ("conv_dw_2",  3,  32,   32, 1, 1,  True, "relu"),   # conv_stem.conv_3x3
    ("conv_3",     1,  32,   16, 1, 0, False, "none"),   # conv_stem.reduce_1x1
    # layer.0
    ("conv_4",     1,  16,   96, 1, 0, False, "relu"),
    ("conv_dw_5",  3,  96,   96, 2, 1,  True, "relu"),
    ("conv_6",     1,  96,   24, 1, 0, False, "none"),
    # layer.1
    ("conv_7",     1,  24,  144, 1, 0, False, "relu"),
    ("conv_dw_8",  3, 144,  144, 1, 1,  True, "relu"),
    ("conv_9",     1, 144,   24, 1, 0, False, "none"),   # resadd with conv_6
    # layer.2
    ("conv_10",    1,  24,  144, 1, 0, False, "relu"),
    ("conv_dw_11", 3, 144,  144, 2, 1,  True, "relu"),
    ("conv_12",    1, 144,   32, 1, 0, False, "none"),
    # layer.3
    ("conv_13",    1,  32,  192, 1, 0, False, "relu"),
    ("conv_dw_14", 3, 192,  192, 1, 1,  True, "relu"),
    ("conv_15",    1, 192,   32, 1, 0, False, "none"),   # resadd with conv_12
    # layer.4
    ("conv_16",    1,  32,  192, 1, 0, False, "relu"),
    ("conv_dw_17", 3, 192,  192, 1, 1,  True, "relu"),
    ("conv_18",    1, 192,   32, 1, 0, False, "none"),   # resadd with conv_15
    # layer.5
    ("conv_19",    1,  32,  192, 1, 0, False, "relu"),
    ("conv_dw_20", 3, 192,  192, 2, 1,  True, "relu"),
    ("conv_21",    1, 192,   64, 1, 0, False, "none"),
    # layer.6
    ("conv_22",    1,  64,  384, 1, 0, False, "relu"),
    ("conv_dw_23", 3, 384,  384, 1, 1,  True, "relu"),
    ("conv_24",    1, 384,   64, 1, 0, False, "none"),   # resadd with conv_21
    # layer.7
    ("conv_25",    1,  64,  384, 1, 0, False, "relu"),
    ("conv_dw_26", 3, 384,  384, 1, 1,  True, "relu"),
    ("conv_27",    1, 384,   64, 1, 0, False, "none"),   # resadd with conv_24
    # layer.8
    ("conv_28",    1,  64,  384, 1, 0, False, "relu"),
    ("conv_dw_29", 3, 384,  384, 1, 1,  True, "relu"),
    ("conv_30",    1, 384,   64, 1, 0, False, "none"),   # resadd with conv_27
    # layer.9
    ("conv_31",    1,  64,  384, 1, 0, False, "relu"),
    ("conv_dw_32", 3, 384,  384, 1, 1,  True, "relu"),
    ("conv_33",    1, 384,   96, 1, 0, False, "none"),
    # layer.10
    ("conv_34",    1,  96,  576, 1, 0, False, "relu"),
    ("conv_dw_35", 3, 576,  576, 1, 1,  True, "relu"),
    ("conv_36",    1, 576,   96, 1, 0, False, "none"),   # resadd with conv_33
    # layer.11
    ("conv_37",    1,  96,  576, 1, 0, False, "relu"),
    ("conv_dw_38", 3, 576,  576, 1, 1,  True, "relu"),
    ("conv_39",    1, 576,   96, 1, 0, False, "none"),   # resadd with conv_36
    # layer.12
    ("conv_40",    1,  96,  576, 1, 0, False, "relu"),
    ("conv_dw_41", 3, 576,  576, 2, 1,  True, "relu"),
    ("conv_42",    1, 576,  160, 1, 0, False, "none"),
    # layer.13
    ("conv_43",    1, 160,  960, 1, 0, False, "relu"),
    ("conv_dw_44", 3, 960,  960, 1, 1,  True, "relu"),
    ("conv_45",    1, 960,  160, 1, 0, False, "none"),   # resadd with conv_42
    # layer.14
    ("conv_46",    1, 160,  960, 1, 0, False, "relu"),
    ("conv_dw_47", 3, 960,  960, 1, 1,  True, "relu"),
    ("conv_48",    1, 960,  160, 1, 0, False, "none"),   # resadd with conv_45
    # layer.15
    ("conv_49",    1, 160,  960, 1, 0, False, "relu"),
    ("conv_dw_50", 3, 960,  960, 1, 1,  True, "relu"),
    ("conv_51",    1, 960,  320, 1, 0, False, "none"),
    # Final 1x1
    ("conv_52",    1, 320, 1280, 1, 0, False, "relu"),
]

# Residual skip connections: target -> source
# The target layer's output gets added to the source layer's output
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

# ---------------------------------------------------------------------------
# Layer mapping: Gemmini name -> HuggingFace prefix
# ---------------------------------------------------------------------------
def build_layer_mapping():
    """Return list of (gemmini_name, hf_prefix, layer_type) tuples."""
    mapping = []
    mapping.append(("conv_1",    "mobilenet_v2.conv_stem.first_conv", "conv"))
    mapping.append(("conv_dw_2", "mobilenet_v2.conv_stem.conv_3x3",   "dw"))
    mapping.append(("conv_3",    "mobilenet_v2.conv_stem.reduce_1x1", "conv"))

    gemmini_idx = 4
    for hf_layer_idx in range(16):
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
# Spatial dimension computation
# ---------------------------------------------------------------------------
def compute_spatial_dims():
    """Compute spatial output dims and Gemmini params for each conv layer."""
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


# ---------------------------------------------------------------------------
# BN folding
# ---------------------------------------------------------------------------
def fold_bn(conv_weight, bn_weight, bn_bias, bn_mean, bn_var, eps=BN_EPS):
    """Fold BatchNorm into conv weight and bias."""
    inv_std = 1.0 / np.sqrt(bn_var + eps)
    scale = bn_weight * inv_std
    shape = [conv_weight.shape[0]] + [1] * (conv_weight.ndim - 1)
    w_folded = conv_weight * scale.reshape(shape)
    b_folded = bn_bias - bn_weight * bn_mean * inv_std
    return w_folded, b_folded


def reshape_conv_weight(w):
    """Regular conv: [out_ch, in_ch, kH, kW] -> [patch_size, out_ch].
    Gemmini expects patch dimension in (kH, kW, in_ch) order.
    """
    out_ch = w.shape[0]
    if w.ndim == 4:
        w = w.transpose(0, 2, 3, 1)  # [out_ch, kH, kW, in_ch]
    return w.reshape(out_ch, -1).T


def reshape_dw_weight(w):
    """Depthwise conv: [channels, 1, kH, kW] -> [channels, kH, kW]."""
    assert w.shape[1] == 1, f"Expected groups dim=1, got {w.shape[1]}"
    return w.squeeze(1)


def get_conv_bn_params(state_dict, prefix):
    """Extract conv weight and BN parameters from a HuggingFace MobileNetV2ConvLayer."""
    conv_w = state_dict[f"{prefix}.convolution.weight"].numpy().astype(np.float64)
    bn_gamma = state_dict[f"{prefix}.normalization.weight"].numpy().astype(np.float64)
    bn_beta = state_dict[f"{prefix}.normalization.bias"].numpy().astype(np.float64)
    bn_mean = state_dict[f"{prefix}.normalization.running_mean"].numpy().astype(np.float64)
    bn_var = state_dict[f"{prefix}.normalization.running_var"].numpy().astype(np.float64)
    return conv_w, bn_gamma, bn_beta, bn_mean, bn_var


def extract_and_fold(state_dict, hf_prefix, layer_type):
    """Extract conv+BN params, fold BN, reshape for Gemmini.
    Returns: (w_reshaped, b_folded, bn_gamma, bn_beta)
    """
    conv_w, bn_gamma, bn_beta, bn_mean, bn_var = get_conv_bn_params(state_dict, hf_prefix)
    w_folded, b_folded = fold_bn(conv_w, bn_gamma, bn_beta, bn_mean, bn_var)
    if layer_type == "dw":
        return reshape_dw_weight(w_folded), b_folded, bn_gamma, bn_beta
    return reshape_conv_weight(w_folded), b_folded, bn_gamma, bn_beta


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------
def quantize_weight_int8(w_float):
    """Symmetric per-tensor INT8 quantization."""
    w_abs_max = np.max(np.abs(w_float))
    if w_abs_max < 1e-10:
        return np.zeros_like(w_float, dtype=np.int8), 1e-10
    w_scale = float(w_abs_max) / 127.0
    w_int = np.clip(np.round(w_float / w_scale), -128, 127).astype(np.int8)
    return w_int, w_scale


def quantize_weight_perchannel(w_flat):
    """Per-channel (per output channel) INT8 quantization for pointwise conv / FC.
    w_flat: [K, J] where J = out_channels.
    Returns (w_int8[K,J], ws_arr[J]).
    """
    J = w_flat.shape[1]
    ws = np.zeros(J, dtype=np.float64)
    for j in range(J):
        mx = float(np.max(np.abs(w_flat[:, j])))
        ws[j] = max(mx / 127.0, 1e-10)
    w_int = np.clip(np.round(w_flat / ws[np.newaxis, :]), -128, 127).astype(np.int8)
    return w_int, ws


def quantize_bias_int32(b_float, combined_scale):
    """Quantize bias to INT32 using combined_scale = w_scale * x_scale."""
    if combined_scale < 1e-10:
        return np.zeros_like(b_float, dtype=np.int32)
    return np.clip(
        np.round(b_float / combined_scale), -(2**31), 2**31 - 1
    ).astype(np.int32)


def compute_output_scale(w_scale, x_scale, y_range):
    """Compute exact float output_scale = w_scale * x_scale / y_scale."""
    y_scale = y_range / 127.0
    raw = (w_scale * x_scale) / y_scale
    if raw <= 0 or not np.isfinite(raw):
        raw = 1.0
    return float(raw), f"{raw:.8e}f"


def estimate_activation_range(bn_gamma, bn_beta, has_relu):
    """Estimate activation range from BN statistics.

    For ReLU6 layers (MobileNetV2 uses ReLU6 for all relu activations):
      - y_range = 6.0 (the ReLU6 clamp ensures output is in [0, 6])
      - In INT8, clipping to 127 with y_scale = 6/127 implements ReLU6

    For linear (no activation) layers (reduce/project):
      - Use per-channel max of |beta| + 6*|gamma| (6-sigma for wider coverage)
    """
    if has_relu:
        y_range = 6.0  # ReLU6 clamp
    else:
        y_range = max(float(np.max(np.abs(bn_beta) + 6.0 * np.abs(bn_gamma))), 1.0)
    return y_range


def run_calibration(model, image_dir, num_images=100, percentile=99.99):
    """Run calibration images through the float model to measure actual
    per-layer activation ranges using a percentile (not max-abs).

    Uses PyTorch forward hooks on each BN layer to capture post-BN output.
    For ReLU6 layers, applies min(relu(x), 6) before measuring.
    For linear layers, measures abs(x).

    Returns dict: gemmini_name -> calibrated y_range.
    """
    import cv2

    # Build lookup for which layers have relu vs linear
    arch_lookup = {n: a for n, k, ic, oc, s, p, dw, a in LAYER_ARCH}

    all_abs = {}

    def make_hook(gemmini_name, activation):
        def hook(module, inp, out):
            x = out.detach().float()
            if activation == "relu":
                x = torch.clamp(torch.relu(x), max=6.0)  # ReLU6
            vals = x.abs().flatten()
            if vals.numel() > 10000:
                idx = torch.randperm(vals.numel())[:10000]
                vals = vals[idx]
            if gemmini_name not in all_abs:
                all_abs[gemmini_name] = []
            all_abs[gemmini_name].append(vals.numpy())
        return hook

    # Register hooks on BN layers
    mapping = build_layer_mapping()
    hooks = []
    for gemmini_name, hf_prefix, layer_type in mapping:
        if layer_type == "fc":
            continue
        activation = arch_lookup.get(gemmini_name, "none")
        parts = hf_prefix.split(".")
        mod = model
        for p in parts:
            mod = getattr(mod, p)
        bn_mod = mod.normalization
        h = bn_mod.register_forward_hook(make_hook(gemmini_name, activation))
        hooks.append(h)

    files = sorted([
        f for f in os.listdir(image_dir)
        if f.lower().endswith((".jpeg", ".jpg", ".png"))
    ])[:num_images]
    print(f"  Running {len(files)} calibration images from {image_dir}...")

    mean_arr = np.array(IMAGENET_MEAN, dtype=np.float64)
    std_arr = np.array(IMAGENET_STD, dtype=np.float64)

    with torch.no_grad():
        for i, fname in enumerate(files):
            img_bgr = cv2.imread(os.path.join(image_dir, fname))
            if img_bgr is None:
                continue
            img = cv2.resize(img_bgr, (INPUT_DIM, INPUT_DIM))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_f = img_rgb.astype(np.float64) / 255.0
            for c in range(3):
                img_f[:, :, c] = (img_f[:, :, c] - mean_arr[c]) / std_arr[c]
            x = torch.tensor(img_f.transpose(2, 0, 1)[np.newaxis], dtype=torch.float32)
            model(x)
            if (i + 1) % 50 == 0:
                print(f"    [{i+1}/{len(files)}]")

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
# C formatting helpers
# ---------------------------------------------------------------------------
def fmt_int(v):    return str(int(v))
def fmt_int_1d(arr): return "{" + ",".join(fmt_int(v) for v in arr.flat if True) + "}" if arr.ndim == 1 else "{" + ",".join(fmt_int(v) for v in arr) + "}"
def fmt_int_2d(arr): return "{" + ",".join("{" + ",".join(fmt_int(v) for v in row) + "}" for row in arr) + "}"
def fmt_int_3d(arr): return "{" + ",".join("{" + ",".join("{" + ",".join(fmt_int(v) for v in plane) + "}" for plane in ch) + "}" for ch in arr) + "}"


def fmt_1d_flat(arr):
    """Format a 1D array as C initializer {v0,v1,...}."""
    return "{" + ",".join(str(int(v)) for v in arr.flat) + "}"


def fmt_1d_float(arr):
    """Format a 1D float array as C initializer {v0f,v1f,...}."""
    return "{" + ",".join(f"{float(v):.8e}f" for v in arr) + "}"


def fmt_2d(arr):
    """Format a 2D array as C initializer {{...},{...},...}."""
    rows = []
    for i in range(arr.shape[0]):
        rows.append("{" + ",".join(str(int(v)) for v in arr[i]) + "}")
    return "{" + ",".join(rows) + "}"


def fmt_3d(arr):
    """Format a 3D array as C initializer {{{...}},...}."""
    planes = []
    for i in range(arr.shape[0]):
        rows = []
        for j in range(arr.shape[1]):
            rows.append("{" + ",".join(str(int(v)) for v in arr[i, j]) + "}")
        planes.append("{" + ",".join(rows) + "}")
    return "{" + ",".join(planes) + "}"


# ---------------------------------------------------------------------------
# Header file writing
# ---------------------------------------------------------------------------
def write_header(output_path, layers_data, conv_params):
    """Write the complete mobilenet_params.h file."""
    with open(output_path, "w") as f:
        f.write("// Generated by extract_mobilenet_imagenet_int8.py\n")
        f.write("#ifndef MOBILENET_PARAMETERS_H\n")
        f.write("#define MOBILENET_PARAMETERS_H\n\n")
        f.write("#include <include/gemmini_params.h>\n\n")

        for entry in layers_data:
            name = entry["name"]
            lt = entry["layer_type"]

            if lt == "fc":
                write_fc_layer(f, entry, conv_params)
            elif lt == "dw":
                write_dw_layer(f, entry, conv_params)
            else:
                write_conv_layer(f, entry, conv_params)

        f.write("#endif // MOBILENET_PARAMETERS_H\n")

    print(f"Written: {output_path}")


def write_conv_layer(f, entry, conv_params):
    """Write a regular (non-depthwise) conv layer."""
    name = entry["name"]
    p = conv_params[name]
    w_int = entry["weight"]
    b_int = entry["bias"]
    output_scale_str = entry["output_scale_str"]
    res_scale = entry.get("res_scale", 1.0)

    # Weight: [patch_size][out_ch]
    f.write(f"static const elem_t {name}_w[{p['patch_size']}][{p['out_channels']}] row_align(1) = ")
    f.write(fmt_2d(w_int))
    f.write(";\n")

    # Bias: [out_ch]
    f.write(f"static const acc_t {name}_b[{p['out_channels']}] row_align_acc(1) = ")
    f.write(fmt_1d_flat(b_int))
    f.write(";\n")

    # Input buffer (for im2col or matmul input)
    f.write(f"static elem_t {name}_in[{p['I']}][{p['K']}] row_align(1);\n")

    # Output buffer
    f.write(f"static elem_t {name}_out[{p['I']}][{p['J']}] row_align(1);\n")

    # ConvParams struct
    res_scale_str = f"{res_scale:.8e}f"
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
    f.write(f".res_scale={res_scale_str}")
    f.write("};\n")

    # Per-channel output scale array (for non-DW layers using PC_MM in mobilenet_v1.c)
    if "output_scale_arr" in entry:
        os_arr = entry["output_scale_arr"]
        f.write(f"static const float {name}_os[{p['out_channels']}] = ")
        f.write(fmt_1d_float(os_arr))
        f.write(";\n")

    f.write("\n")


def write_dw_layer(f, entry, conv_params):
    """Write a depthwise conv layer."""
    name = entry["name"]
    p = conv_params[name]
    w_int = entry["weight"]
    b_int = entry["bias"]
    output_scale_str = entry["output_scale_str"]
    res_scale = entry.get("res_scale", 1.0)

    # Weight: [channels][3][3]
    f.write(f"static const elem_t {name}_w[{p['out_channels']}][3][3] row_align(1) = ")
    f.write(fmt_3d(w_int))
    f.write(";\n")

    # Bias: [out_ch]
    f.write(f"static const acc_t {name}_b[{p['out_channels']}] row_align_acc(1) = ")
    f.write(fmt_1d_flat(b_int))
    f.write(";\n")

    # Output buffer (DW layers only have _out buffer, no _in)
    f.write(f"static elem_t {name}_out[{p['I']}][{p['J']}] row_align(1);\n")

    # ConvParams struct (no K field for depthwise)
    res_scale_str = f"{res_scale:.8e}f"
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
    f.write(f".res_scale={res_scale_str}, ")
    f.write(f".I={p['I']}, .J={p['J']}")
    f.write("};\n\n")


def write_fc_layer(f, entry, conv_params):
    """Write the FC classifier layer."""
    name = entry["name"]
    w_int = entry["weight"]
    b_int_2d = entry["bias"]
    output_scale_str = entry["output_scale_str"]

    out_f = 1000
    in_f = 1280

    # Weight: fc_53_w[1000][1280]
    f.write(f"static const elem_t {name}_w[{out_f}][{in_f}] row_align(1) = ")
    f.write(fmt_2d(w_int))
    f.write(";\n")

    # Bias: fc_53_b[1000][4]
    f.write(f"static const acc_t {name}_b[{out_f}][{BATCH_SIZE}] row_align_acc(1) = ")
    f.write(fmt_2d(b_int_2d))
    f.write(";\n")

    # Output buffer
    f.write(f"static elem_t {name}_out[{out_f}][{BATCH_SIZE}] row_align(1);\n")

    # FcParams struct
    f.write(f"static const struct FcParams {name}_params = {{")
    f.write(f".batch_size={BATCH_SIZE}, ")
    f.write(f".in_features={in_f}, .out_features={out_f}, ")
    f.write(f".bias=1, ")
    f.write(f".output_scale={output_scale_str}, ")
    f.write(f".I={out_f}, .J={BATCH_SIZE}, .K={in_f}")
    f.write("};\n")

    # Per-channel (per class) output scale array
    if "output_scale_arr" in entry:
        os_arr = entry["output_scale_arr"]
        f.write(f"static const float {name}_os[{out_f}] = ")
        f.write(fmt_1d_float(os_arr))
        f.write(";\n")

    f.write("\n")


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Extract MobileNetV2 INT8 weights")
    parser.add_argument("--pixel-minus-128", action="store_true",
                        help="Fold ImageNet normalization into conv_1 weights "
                             "so that input is pixel-128 (int8).")
    parser.add_argument("--output", type=str, default=None,
                        help="Output header file path (default: ../mobilenet_params.h)")
    parser.add_argument("--calibrate-dir", type=str, default=None,
                        help="Directory of calibration JPEG images for activation range measurement")
    parser.add_argument("--num-calibrate", type=int, default=100,
                        help="Number of calibration images to use (default: 100)")
    args = parser.parse_args()

    print(f"Loading {MODEL_NAME} from HuggingFace...")
    model = MobileNetV2ForImageClassification.from_pretrained(MODEL_NAME)
    model.eval()
    state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    num_classes = state_dict["classifier.weight"].shape[0]
    print(f"Model has {num_classes} output classes")
    if num_classes == 1001:
        print("  -> Will strip background class (index 0) to get 1000 classes")

    # Compute spatial dimensions for all conv layers
    conv_params = compute_spatial_dims()

    # Build layer mapping
    mapping = build_layer_mapping()
    arch_lookup = {n: (n, k, ic, oc, s, p, dw, a)
                   for n, k, ic, oc, s, p, dw, a in LAYER_ARCH}

    # -----------------------------------------------------------------------
    # Pre-pass: compute activation ranges and x_scales for all layers
    # Prefer calibration (running actual images) over BN statistics.
    # BN estimates can be off for MobileNetV2 (ReLU6, narrow bottlenecks).
    # -----------------------------------------------------------------------
    print("\n--- Pre-pass: computing activation ranges ---")

    calibrated = {}
    if args.calibrate_dir:
        print("Calibration pass: measuring actual activation ranges...")
        calibrated = run_calibration(model, args.calibrate_dir, args.num_calibrate)
    else:
        print("WARNING: No calibration dir. Using BN estimates (may be inaccurate).")
        print("  Use --calibrate-dir /path/to/images for better accuracy.")

    layer_y_range = {}
    layer_x_scale = {}  # Store the x_scale used for each layer

    if args.pixel_minus_128:
        x_scale_0 = 128.0 / 127.0
        print(f"  pixel-minus-128 mode: x_scale_0 = {x_scale_0:.6f}")
    else:
        FMAX = max(
            IMAGENET_MEAN[i] / IMAGENET_STD[i]
            for i in range(3)
        ) + max(
            (1.0 - IMAGENET_MEAN[i]) / IMAGENET_STD[i]
            for i in range(3)
        )
        # Actually, FMAX should be the max absolute value of normalized input
        # which is max(|mean/std|, |(1-mean)/std|) across channels
        maxvals = []
        for i in range(3):
            maxvals.append(abs(-IMAGENET_MEAN[i] / IMAGENET_STD[i]))  # pixel=0
            maxvals.append(abs((1.0 - IMAGENET_MEAN[i]) / IMAGENET_STD[i]))  # pixel=255
        FMAX = max(maxvals)
        x_scale_0 = FMAX / 127.0
        print(f"  FMAX mode: FMAX={FMAX:.4f}, x_scale_0 = {x_scale_0:.6f}")

    x_scale = x_scale_0

    for gemmini_name, hf_prefix, layer_type in mapping:
        layer_x_scale[gemmini_name] = x_scale

        if layer_type == "fc":
            # FC layer: estimate output logit range
            # x_scale is per-INT8-unit; actual float input magnitudes are up to
            # x_scale * 127 = y_range of previous layer.  Use the float-level
            # magnitude so the 3-sigma CLT estimate is in the correct units.
            fc_w_float = state_dict["classifier.weight"].numpy().astype(np.float64)
            fc_b_float = state_dict["classifier.bias"].numpy().astype(np.float64)
            if num_classes == 1001:
                fc_w_float = fc_w_float[1:, :]
                fc_b_float = fc_b_float[1:]
            input_float_range = x_scale * 127.0   # = y_range of previous layer
            fc_y_range = max(
                float(np.max(np.abs(fc_b_float))
                      + np.std(fc_w_float) * np.sqrt(float(fc_w_float.shape[1])) * input_float_range * 3),
                5.0
            )
            layer_y_range[gemmini_name] = fc_y_range
        else:
            arch_info = arch_lookup[gemmini_name]
            has_relu = arch_info[7] == "relu"

            if gemmini_name in calibrated:
                y_range = max(calibrated[gemmini_name], 1.0)
            else:
                _, _, bn_gamma, bn_beta = extract_and_fold(state_dict, hf_prefix, layer_type)
                y_range = estimate_activation_range(bn_gamma, bn_beta, has_relu)
                if calibrated:
                    print(f"  WARNING: {gemmini_name} not calibrated, using BN estimate")
            layer_y_range[gemmini_name] = y_range

            # Propagate x_scale
            # After a resadd target layer, the resadd output is at the target's scale.
            # All MobileNetV2 residuals are identity (no projection shortcuts),
            # so we just propagate normally.
            x_scale = y_range / 127.0

    # -----------------------------------------------------------------------
    # Main extraction pass
    # -----------------------------------------------------------------------
    print("\n--- Main pass: quantizing weights ---")

    layers_data = []
    x_scale = x_scale_0

    for gemmini_name, hf_prefix, layer_type in mapping:
        y_range = layer_y_range[gemmini_name]

        if layer_type == "fc":
            # FC layer
            fc_w_float = state_dict["classifier.weight"].numpy().astype(np.float64)
            fc_b_float = state_dict["classifier.bias"].numpy().astype(np.float64)

            if num_classes == 1001:
                fc_w_float = fc_w_float[1:, :]  # [1000, 1280] — skip background
                fc_b_float = fc_b_float[1:]      # [1000]

            # fc_53_w[1000][1280] — per-row (per output class) quantization
            # w_flat for perchannel fn expects [K, J]: transpose to [1280, 1000]
            w_int_t, ws_arr = quantize_weight_perchannel(fc_w_float.T)
            w_int = w_int_t.T  # back to [1000, 1280] layout for header

            # Per-class bias: b_int[j] = round(b_f[j] / (ws[j] * x_scale))
            combined_scales = ws_arr * x_scale
            b_int_1d = np.where(
                combined_scales >= 1e-10,
                np.clip(np.round(fc_b_float / combined_scales), -(2**31), 2**31 - 1),
                np.zeros(len(fc_b_float))
            ).astype(np.int32)
            b_int_2d = np.tile(b_int_1d.reshape(-1, 1), (1, BATCH_SIZE))

            # Per-class output scales
            y_scale = y_range / 127.0
            os_arr = (ws_arr * x_scale) / y_scale
            os_arr = np.where(np.isfinite(os_arr) & (os_arr > 0), os_arr, 1.0)
            # output_scale_str: use mean for the FcParams struct (legacy; PC loop uses os array)
            _os = float(np.mean(os_arr))
            output_scale_str = f"{_os:.8e}f"

            print(f"  {gemmini_name:12s}  per-channel  x_scale={x_scale:.6f}  "
                  f"y_range={y_range:.2f}  os_mean={_os:.4e}")

            layers_data.append({
                "name": gemmini_name,
                "layer_type": "fc",
                "weight": w_int,
                "bias": b_int_2d,
                "output_scale_str": output_scale_str,
                "output_scale_arr": os_arr,
            })

        else:
            # Conv layer (regular or depthwise)
            if gemmini_name == "conv_1" and args.pixel_minus_128:
                # Fold ImageNet normalization into conv_1 weights
                conv_w, gamma, beta, bn_m, bn_v = get_conv_bn_params(state_dict, hf_prefix)
                w_bn, b_bn = fold_bn(conv_w, gamma, beta, bn_m, bn_v)
                # w_bn shape: [32, 3, 3, 3] (out_ch, in_ch=3, kH, kW)
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
                print(f"  [conv_1] Folded ImageNet normalization (pixel-128 mode)")
            else:
                w_float, b_float, bn_gamma, bn_beta = extract_and_fold(
                    state_dict, hf_prefix, layer_type
                )

            arch_info = arch_lookup[gemmini_name]
            has_relu = arch_info[7] == "relu"

            # For DW layers with ReLU: zero out dead channels (var ≈ 0).
            # These channels produce constant (bias-only) output that becomes 0
            # after ReLU6, so zeroing their weights is lossless. This prevents
            # dead channels from dominating the per-tensor w_scale.
            if layer_type == "dw" and has_relu:
                _, _, _, _, bn_var = get_conv_bn_params(state_dict, hf_prefix)
                dead_threshold = 0.01
                n_dead = 0
                for c in range(w_float.shape[0]):
                    if bn_var[c] < dead_threshold:
                        # Zero weights to remove outlier influence on per-tensor scale.
                        # Clamp bias to ReLU6 output so the constant contribution
                        # of the dead channel is preserved (not wiped to zero).
                        w_float[c] = 0.0
                        b_float[c] = float(np.clip(b_float[c], 0.0, 6.0))
                        n_dead += 1
                if n_dead > 0:
                    print(f"  [{gemmini_name}] Fixed {n_dead} dead channels (zero weights, clamp bias to ReLU6)")

            # res_scale: rescales the skip tensor to match this layer's output scale
            # Used in tiled_resadd_auto: skip_out * res_scale + this_out
            res_scale = 1.0
            if gemmini_name in RESIDUAL_SKIP:
                skip_src = RESIDUAL_SKIP[gemmini_name]
                if skip_src in layer_y_range:
                    res_scale = layer_y_range[skip_src] / y_range
                else:
                    print(f"  WARNING: skip source {skip_src} not found for {gemmini_name}")

            if layer_type == "dw" or gemmini_name == "conv_1":
                # Per-tensor quantization: DW layers and the first conv
                w_int, w_scale = quantize_weight_int8(w_float)
                b_int = quantize_bias_int32(b_float, w_scale * x_scale)

                _os, output_scale_str = compute_output_scale(w_scale, x_scale, y_range)
                output_scale_arr = None  # per-tensor: no per-channel array

                act_str = "RELU" if has_relu else "LINEAR"
                res_str = f" res_scale={res_scale:.4f}" if gemmini_name in RESIDUAL_SKIP else ""
                print(f"  {gemmini_name:12s}  w_scale={w_scale:.6f}  x_scale={x_scale:.6f}  "
                      f"y_range={y_range:.2f}  output_scale={_os:.4e}  {act_str}{res_str}")
            else:
                # Per-channel quantization for non-DW pointwise conv layers
                # w_float shape: [patch_size, out_ch] = [K, J]
                w_int, ws_arr = quantize_weight_perchannel(w_float)

                # Per-channel bias: b_int[j] = round(b_f[j] / (ws[j] * x_scale))
                combined_scales = ws_arr * x_scale
                b_int = np.where(
                    combined_scales >= 1e-10,
                    np.clip(np.round(b_float / combined_scales), -(2**31), 2**31 - 1),
                    np.zeros(len(b_float))
                ).astype(np.int32)

                # Per-channel output scales
                y_scale = y_range / 127.0
                os_arr = (ws_arr * x_scale) / y_scale
                output_scale_arr = np.where(np.isfinite(os_arr) & (os_arr > 0), os_arr, 1.0)

                # output_scale_str: use mean scale for ConvParams struct (legacy path)
                _os = float(np.mean(output_scale_arr))
                output_scale_str = f"{_os:.8e}f"

                act_str = "RELU" if has_relu else "LINEAR"
                res_str = f" res_scale={res_scale:.4f}" if gemmini_name in RESIDUAL_SKIP else ""
                print(f"  {gemmini_name:12s}  per-channel  x_scale={x_scale:.6f}  "
                      f"y_range={y_range:.2f}  os_mean={_os:.4e}  {act_str}{res_str}")

            entry_dict = {
                "name": gemmini_name,
                "layer_type": "dw" if arch_info[6] else "conv",
                "weight": w_int,
                "bias": b_int,
                "output_scale_str": output_scale_str,
                "res_scale": res_scale,
            }
            if output_scale_arr is not None:
                entry_dict["output_scale_arr"] = output_scale_arr
            layers_data.append(entry_dict)

            # Propagate x_scale: next layer's input scale = this layer's output scale
            x_scale = y_range / 127.0

    # -----------------------------------------------------------------------
    # Write output header
    # -----------------------------------------------------------------------
    output_dir = os.path.dirname(os.path.abspath(__file__))
    if args.output:
        output_path = args.output
    else:
        output_path = os.path.normpath(os.path.join(output_dir, "..", "mobilenet_params.h"))

    print(f"\nWriting {output_path}...")
    write_header(output_path, layers_data, conv_params)

    # Print summary
    total_params = 0
    for entry in layers_data:
        total_params += entry["weight"].size + entry["bias"].size
    print(f"Total parameters: {total_params:,}")
    print("Done!")


if __name__ == "__main__":
    main()
