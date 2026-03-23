#!/usr/bin/env python3
"""
Re-extract INT8-quantized weights from google/mobilenet_v2_1.0_224 and compare
with the known-good imagenet/mobilenet_params.h (85 % accuracy).

The comparison tests two quantization variants:
  A) pow2  — output_scale rounded to nearest power of 2   (matchs mobilenet_params.h style)
  B) exact — output_scale as an exact float               (matches post-fix CIFAR-10 style)

And two input scales for the first layer:
  1) x_scale_0 = 1.0         (raw pixel-128 treated as integer → "old-bug" style)
  2) x_scale_0 = 128/127     (symmetric INT8 for [-128, 127] range)

Usage:
    cd imagenet/mobilenet_verify
    conda run -n ImageNet python extract_mobilenet_imagenet_int8.py

Outputs:
  mobilenet_params_regen_pow2_x1.h   — pow2 output_scale, x_scale_0=1.0
  mobilenet_params_regen_pow2_x128.h — pow2 output_scale, x_scale_0=128/127
  mobilenet_params_regen_exact_x1.h  — exact output_scale, x_scale_0=1.0
  mobilenet_params_regen_exact_x128.h — exact output_scale, x_scale_0=128/127
  comparison_report.txt              — numerical comparison with mobilenet_params.h
"""

import os
import math
import sys
import re
import numpy as np
import torch
from transformers import MobileNetV2ForImageClassification

MODEL_NAME = "google/mobilenet_v2_1.0_224"
BATCH_SIZE = 4
NUM_CLASSES = 1000
INPUT_DIM = 224

# ---------------------------------------------------------------------------
# Architecture (same as extract_mobilenet_cifar10_int.py)
# ---------------------------------------------------------------------------
# (name, kernel, in_ch, out_ch, stride, padding, depthwise, activation)
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

FC_PARAMS_IMAGENET = {
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

# Residual skip connections (reduce layers that receive skip)
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
# Spatial dimension computation
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
    mapping.append(("conv_1",    "mobilenet_v2.conv_stem.first_conv", "conv"))
    mapping.append(("conv_dw_2", "mobilenet_v2.conv_stem.conv_3x3",   "dw"))
    mapping.append(("conv_3",    "mobilenet_v2.conv_stem.reduce_1x1", "conv"))
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
    return w.reshape(w.shape[0], -1).T  # [patch_size, out_ch]


def reshape_dw_weight(w):
    assert w.shape[1] == 1
    return w.squeeze(1)  # [channels, 3, 3]


def get_conv_bn_params(state_dict, prefix):
    conv_w = state_dict[f"{prefix}.convolution.weight"].numpy()
    bn_gamma = state_dict[f"{prefix}.normalization.weight"].numpy()
    bn_beta = state_dict[f"{prefix}.normalization.bias"].numpy()
    bn_mean = state_dict[f"{prefix}.normalization.running_mean"].numpy()
    bn_var = state_dict[f"{prefix}.normalization.running_var"].numpy()
    return conv_w, bn_gamma, bn_beta, bn_mean, bn_var


def extract_and_fold(state_dict, hf_prefix, layer_type):
    conv_w, bn_gamma, bn_beta, bn_mean, bn_var = get_conv_bn_params(
        state_dict, hf_prefix
    )
    w_folded, b_folded = fold_bn(conv_w, bn_gamma, bn_beta, bn_mean, bn_var)
    if layer_type == "dw":
        return reshape_dw_weight(w_folded), b_folded, bn_gamma, bn_beta
    return reshape_conv_weight(w_folded), b_folded, bn_gamma, bn_beta


# ---------------------------------------------------------------------------
# Quantization functions
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


def compute_output_scale_pow2(w_scale, x_scale, y_range):
    """Output scale rounded to nearest power of 2 (1/(1<<N) format)."""
    y_scale = y_range / 127.0
    raw = (w_scale * x_scale) / y_scale
    if raw <= 0 or not np.isfinite(raw):
        raw = 1.0
    N = max(0, min(30, round(-math.log2(raw))))
    return 1.0 / (1 << N), f"(1.0 / (1 << {N}))"


def compute_output_scale_exact(w_scale, x_scale, y_range):
    """Output scale as exact float."""
    y_scale = y_range / 127.0
    raw = (w_scale * x_scale) / y_scale
    if raw <= 0 or not np.isfinite(raw):
        raw = 1.0
    return raw, f"{raw:.8e}f"


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
def fmt_int(v):    return str(int(v))
def fmt_int_1d(arr): return "{" + ",".join(fmt_int(v) for v in arr) + "}"
def fmt_int_2d(arr): return "{" + ",".join(fmt_int_1d(row) for row in arr) + "}"
def fmt_int_3d(arr): return "{" + ",".join(fmt_int_2d(p) for p in arr) + "}"


# ---------------------------------------------------------------------------
# Header writing
# ---------------------------------------------------------------------------
def write_conv_layer(f, name, w_int, b_int, output_scale_str, res_scale_str, p):
    f.write(f"static const elem_t {name}_w[{p['patch_size']}][{p['out_channels']}] row_align(1) = ")
    f.write(fmt_int_2d(w_int)); f.write(";\n")
    f.write(f"static const acc_t {name}_b[{p['out_channels']}] row_align_acc(1) = ")
    f.write(fmt_int_1d(b_int)); f.write(";\n")
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
    f.write("};\n\n")


def write_dw_layer(f, name, w_int, b_int, output_scale_str, res_scale_str, p):
    f.write(f"static const elem_t {name}_w[{p['out_channels']}][3][3] row_align(1) = ")
    f.write(fmt_int_3d(w_int)); f.write(";\n")
    f.write(f"static const acc_t {name}_b[{p['out_channels']}] row_align_acc(1) = ")
    f.write(fmt_int_1d(b_int)); f.write(";\n")
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


def write_fc_layer(f, name, w_int, b_int_2d, output_scale_str, p):
    out_f, in_f = p["out_features"], p["in_features"]
    f.write(f"static const elem_t {name}_w[{out_f}][{in_f}] row_align(1) = ")
    f.write(fmt_int_2d(w_int)); f.write(";\n")
    f.write(f"static const acc_t {name}_b[{out_f}][{BATCH_SIZE}] row_align_acc(1) = ")
    f.write(fmt_int_2d(b_int_2d)); f.write(";\n")
    f.write(f"static const struct FcParams {name}_params = {{")
    f.write(f".batch_size={p['batch_size']}, ")
    f.write(f".in_features={in_f}, .out_features={out_f}, ")
    f.write(f".bias={p['bias']}, ")
    f.write(f".output_scale={output_scale_str}, ")
    f.write(f".I={p['I']}, .J={p['J']}, .K={p['K']}")
    f.write("};\n\n")


def write_header(output_path, layers_data, conv_params, label):
    with open(output_path, "w") as f:
        f.write(f"// Generated by extract_mobilenet_imagenet_int8.py  variant={label}\n")
        f.write("#ifndef MOBILENET_IMAGENET_REGEN_H\n")
        f.write("#define MOBILENET_IMAGENET_REGEN_H\n\n")
        f.write("#include <include/gemmini_params.h>\n\n")
        for entry in layers_data:
            name, lt = entry["name"], entry["lt"]
            p = conv_params.get(name, FC_PARAMS_IMAGENET.get(name))
            if lt == "fc":
                write_fc_layer(f, name, entry["w_int"], entry["b_int"],
                               entry["output_scale_str"], FC_PARAMS_IMAGENET[name])
            elif lt == "dw":
                write_dw_layer(f, name, entry["w_int"], entry["b_int"],
                               entry["output_scale_str"], entry["res_scale_str"], p)
            else:
                write_conv_layer(f, name, entry["w_int"], entry["b_int"],
                                 entry["output_scale_str"], entry["res_scale_str"], p)
        f.write("#endif\n")
    print(f"  Written: {output_path}")


# ---------------------------------------------------------------------------
# Parse mobilenet_params.h for comparison
# ---------------------------------------------------------------------------
def parse_reference_header(header_path):
    """Extract {layer_name: {"w": list[int], "b": list[int], "output_scale": float}} from header."""
    with open(header_path, "r") as f:
        text = f.read()

    layers = {}

    # Extract weight arrays (elem_t NAME_w[...] = {...})
    for m in re.finditer(r'static const elem_t (\w+)_w\[.*?\] row_align\(1\) = (\{[^;]+\});', text, re.DOTALL):
        name, raw = m.group(1), m.group(2)
        vals = [int(x) for x in re.findall(r'-?\d+', raw)]
        layers.setdefault(name, {})["w"] = vals

    # Extract bias arrays (acc_t NAME_b[...] = {...})
    for m in re.finditer(r'static const acc_t (\w+)_b\[.*?\] row_align_acc\(1\) = (\{[^;]+\});', text, re.DOTALL):
        name, raw = m.group(1), m.group(2)
        vals = [int(x) for x in re.findall(r'-?\d+', raw)]
        layers.setdefault(name, {})["b"] = vals

    # Extract output_scale from ConvParams / FcParams
    for m in re.finditer(r'(\w+?)_params\s*=\s*\{[^}]*\.output_scale=([^,}]+)[,}]', text):
        name, os_str = m.group(1), m.group(2).strip()
        # Evaluate the scale expression
        try:
            scale = eval(os_str.rstrip("f").replace("(", "(").replace(")", ")"))
        except Exception:
            scale = None
        layers.setdefault(name, {})["output_scale"] = scale
        layers[name]["output_scale_str"] = os_str

    return layers


# ---------------------------------------------------------------------------
# Compare two parsed header dicts
# ---------------------------------------------------------------------------
def compare_layers(ref, gen, variant_label, report_lines):
    report_lines.append(f"\n{'='*70}")
    report_lines.append(f"Variant: {variant_label}")
    report_lines.append(f"{'='*70}")

    total_w_match = 0
    total_w_diff = 0
    total_b_close = 0
    total_b_far = 0
    layer_names = sorted(ref.keys())

    for name in layer_names:
        if name not in gen:
            report_lines.append(f"  MISSING in generated: {name}")
            continue
        r, g = ref[name], gen[name]

        # Weight comparison
        rw = r.get("w", [])
        gw = g.get("w", [])
        if rw and gw and len(rw) == len(gw):
            rw_arr = np.array(rw, dtype=np.int8)
            gw_arr = np.array(gw, dtype=np.int8)
            exact = int(np.sum(rw_arr == gw_arr))
            diff = int(np.sum(rw_arr != gw_arr))
            max_diff = int(np.max(np.abs(rw_arr.astype(np.int16) - gw_arr.astype(np.int16))))
            pct = 100.0 * exact / len(rw)
            total_w_match += exact
            total_w_diff += diff
        else:
            exact, diff, max_diff, pct = 0, len(rw), 0, 0.0

        # Bias comparison
        rb = r.get("b", [])
        gb = g.get("b", [])
        b_max_diff = 0
        b_pct_close = 0.0
        if rb and gb and len(rb) == len(gb):
            rb_arr = np.array(rb, dtype=np.int64)
            gb_arr = np.array(gb, dtype=np.int64)
            b_diff_arr = np.abs(rb_arr - gb_arr)
            b_max_diff = int(np.max(b_diff_arr))
            close = int(np.sum(b_diff_arr <= max(1, int(np.max(np.abs(rb_arr)) * 0.01))))
            b_pct_close = 100.0 * close / len(rb) if rb else 0.0
            total_b_close += close
            total_b_far += len(rb) - close

        # Output scale comparison
        ros = r.get("output_scale")
        gos = g.get("output_scale")
        scale_match = ""
        if ros is not None and gos is not None:
            if abs(ros - gos) < 1e-9:
                scale_match = "✓ exact"
            elif abs(ros - gos) / max(abs(ros), 1e-9) < 0.01:
                scale_match = "≈ within 1%"
            else:
                scale_match = f"✗ ref={ros:.4e} gen={gos:.4e}"
        elif ros is not None:
            scale_match = f"ref={ros:.4e} gen=MISSING"

        w_flag = "✓" if diff == 0 else f"✗ {diff}/{len(rw)} differ (max_diff={max_diff})"
        b_flag = f"{b_pct_close:.0f}% within 1% (max_diff={b_max_diff})"
        report_lines.append(
            f"  {name:15s}  W:{w_flag:40s}  B:{b_flag:40s}  scale:{scale_match}"
        )

    total_w = total_w_match + total_w_diff
    report_lines.append(f"\n  WEIGHT SUMMARY: {total_w_match}/{total_w} exact matches "
                        f"({100.0*total_w_match/max(total_w,1):.1f}%)")
    report_lines.append(f"  BIAS   SUMMARY: {total_b_close} bias elements within 1%  "
                        f"({total_b_far} further)")
    return total_w_match, total_w


# ---------------------------------------------------------------------------
# Core extraction loop (parameterised by x_scale_0 and scale_fn)
# ---------------------------------------------------------------------------
def extract_all_layers(state_dict, x_scale_0, scale_fn, label):
    """Run the full extraction loop and return layers_data list."""
    mapping = build_layer_mapping()
    arch_lookup = {n: (n, k, ic, oc, s, p, dw, a) for n, k, ic, oc, s, p, dw, a in LAYER_ARCH}

    layers_data = []
    layer_y_range = {}
    x_scale = x_scale_0

    print(f"\n  [{label}]  x_scale_0={x_scale_0:.6f}")

    for gemmini_name, hf_prefix, layer_type in mapping:
        if layer_type == "fc":
            fc_w_f = state_dict["classifier.weight"].numpy()
            fc_b_f = state_dict["classifier.bias"].numpy()
            w_int, w_scale = quantize_weight_int8(fc_w_f)
            b_int_1d = quantize_bias_int32(fc_b_f, w_scale * x_scale)
            b_int_2d = np.tile(b_int_1d.reshape(-1, 1), (1, BATCH_SIZE))
            fc_y_range = max(
                float(np.max(np.abs(fc_b_f)) + np.max(np.abs(fc_w_f)) * np.sqrt(1280) * x_scale * 10),
                1.0
            )
            _os, output_scale_str = scale_fn(w_scale, x_scale, fc_y_range)
            layers_data.append({
                "name": gemmini_name, "lt": "fc",
                "w_int": w_int, "b_int": b_int_2d,
                "output_scale_str": output_scale_str,
                "output_scale": _os,
            })
        else:
            w_float, b_float, bn_gamma, bn_beta = extract_and_fold(state_dict, hf_prefix, layer_type)
            arch_info = arch_lookup.get(gemmini_name)
            has_relu = arch_info is not None and arch_info[7] == "relu"
            w_int, w_scale = quantize_weight_int8(w_float)
            b_int = quantize_bias_int32(b_float, w_scale * x_scale)
            y_range = estimate_activation_range(bn_gamma, bn_beta, has_relu)
            layer_y_range[gemmini_name] = y_range
            _os, output_scale_str = scale_fn(w_scale, x_scale, y_range)
            res_scale = 1.0
            # Always use res_scale=1.0 to match original mobilenet_params.h
            res_scale_str = "(1.0 / (1 << 0))"
            layers_data.append({
                "name": gemmini_name, "lt": layer_type,
                "w_int": w_int, "b_int": b_int,
                "output_scale_str": output_scale_str,
                "output_scale": _os,
                "res_scale_str": res_scale_str,
            })
            # Propagate x_scale
            y_scale = y_range / 127.0
            x_scale = y_scale

    return layers_data


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"Loading {MODEL_NAME} from HuggingFace...")
    model = MobileNetV2ForImageClassification.from_pretrained(MODEL_NAME)
    model.eval()
    state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    num_classes = state_dict["classifier.weight"].shape[0]
    print(f"Model has {num_classes} output classes")

    conv_params = compute_conv_params()
    buffers = compute_buffers(conv_params)

    # Path to known-good reference
    ref_path = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mobilenet_params.h")
    )
    if not os.path.exists(ref_path):
        print(f"ERROR: Reference header not found at {ref_path}")
        sys.exit(1)

    print(f"\nParsing reference header: {ref_path}")
    ref_layers = parse_reference_header(ref_path)
    print(f"  Found {len(ref_layers)} layers in reference")

    out_dir = os.path.dirname(os.path.abspath(__file__))
    report_lines = [f"MobileNetV2 ImageNet INT8 Extraction Comparison Report",
                    f"Reference: {ref_path}"]

    variants = [
        ("pow2_x1.0",   1.0,         compute_output_scale_pow2),
        ("pow2_x128",   128.0/127.0, compute_output_scale_pow2),
        ("exact_x1.0",  1.0,         compute_output_scale_exact),
        ("exact_x128",  128.0/127.0, compute_output_scale_exact),
    ]

    best_pct = -1.0
    best_label = None
    for label, x_scale_0, scale_fn in variants:
        layers_data = extract_all_layers(state_dict, x_scale_0, scale_fn, label)

        # Generate header
        out_path = os.path.join(out_dir, f"mobilenet_params_regen_{label}.h")
        write_header(out_path, layers_data, conv_params, label)

        # Build generated layer dict for comparison
        gen_layers = {}
        for entry in layers_data:
            name = entry["name"]
            w_flat = entry["w_int"].flatten().tolist()
            b_flat = entry["b_int"].flatten().tolist()
            gen_layers[name] = {
                "w": w_flat,
                "b": b_flat,
                "output_scale": entry["output_scale"],
                "output_scale_str": entry["output_scale_str"],
            }

        # Compare
        match, total = compare_layers(ref_layers, gen_layers, label, report_lines)
        pct = 100.0 * match / max(total, 1)
        if pct > best_pct:
            best_pct = pct
            best_label = label

    report_lines.append(f"\n{'='*70}")
    report_lines.append(f"BEST MATCH: variant={best_label}  weight_match={best_pct:.1f}%")
    report_lines.append(f"{'='*70}")

    report_path = os.path.join(out_dir, "comparison_report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines) + "\n")

    print(f"\nReport written to: {report_path}")
    print("\n".join(report_lines))


if __name__ == "__main__":
    main()
