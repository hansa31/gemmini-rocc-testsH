#!/usr/bin/env python3
"""
Extract FP32 weights from HuggingFace MobileNetV2 fine-tuned on CIFAR-10
(jialicheng/cifar10_mobilenet-v2) and generate:
  - ../mobilenet_cifar10_params_float.h  (FP32 weights, CIFAR-10 spatial dims)
  - ../cifar10_images.h                  (4 sample CIFAR-10 test images, resized to 224x224)

The backbone (conv_1 through conv_52) uses the CIFAR-10 fine-tuned model's
weights with BatchNorm folded into conv weights.  The FC layer outputs 10 classes.
All spatial dimensions are recomputed for 224x224 input (CIFAR-10 images resized):
  224 -> 112 -> 56 -> 28 -> 14 -> 7  (five stride-2 reductions)

Usage:
    pip install torch transformers torchvision numpy
    python extract_mobilenet_cifar10_float.py
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
# MobileNetV2 architecture definition (input-size independent)
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
    """Compute per-layer ConvParams for 32x32 CIFAR-10 input."""
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
    """Compute buffer declarations from conv params."""
    buffers = []
    for name, _k, _ic, _oc, _s, _p, dw, _a in LAYER_ARCH:
        p = conv_params[name]
        if not dw:
            buffers.append((f"{name}_in", p["n_patches"], p["patch_size"]))
        buffers.append((f"{name}_out", p["n_patches"], p["out_channels"]))
    buffers.append(("fc_53_out", NUM_CLASSES, BATCH_SIZE))
    return buffers


# ---------------------------------------------------------------------------
# HuggingFace-to-Gemmini layer mapping (identical to ImageNet version)
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

def fold_bn(conv_weight, bn_weight, bn_bias, bn_mean, bn_var, eps=1e-5):
    inv_std = 1.0 / np.sqrt(bn_var + eps)
    scale = bn_weight * inv_std
    shape = [conv_weight.shape[0]] + [1] * (conv_weight.ndim - 1)
    w_folded = conv_weight * scale.reshape(shape)
    b_folded = bn_bias - bn_weight * bn_mean * inv_std
    return w_folded, b_folded


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
# Layer extraction
# ---------------------------------------------------------------------------

def get_conv_bn_params(state_dict, prefix):
    conv_w = state_dict[f"{prefix}.convolution.weight"].numpy()
    bn_gamma = state_dict[f"{prefix}.normalization.weight"].numpy()
    bn_beta = state_dict[f"{prefix}.normalization.bias"].numpy()
    bn_mean = state_dict[f"{prefix}.normalization.running_mean"].numpy()
    bn_var = state_dict[f"{prefix}.normalization.running_var"].numpy()
    return conv_w, bn_gamma, bn_beta, bn_mean, bn_var


def extract_layer(state_dict, gemmini_name, hf_prefix, layer_type):
    if layer_type == "fc":
        fc_w = state_dict["classifier.weight"].numpy()   # [10, 1280]
        fc_b = state_dict["classifier.bias"].numpy()      # [10]
        bias_2d = np.tile(fc_b.reshape(-1, 1), (1, BATCH_SIZE))
        return fc_w, bias_2d

    conv_w, bn_gamma, bn_beta, bn_mean, bn_var = get_conv_bn_params(state_dict, hf_prefix)
    w_folded, b_folded = fold_bn(conv_w, bn_gamma, bn_beta, bn_mean, bn_var)

    if layer_type == "dw":
        return reshape_dw_weight(w_folded), b_folded
    return reshape_conv_weight(w_folded), b_folded


def validate_shapes(gemmini_name, weight, bias, layer_type, conv_params):
    if layer_type == "fc":
        assert weight.shape == (NUM_CLASSES, 1280), \
            f"{gemmini_name}: weight {weight.shape} != ({NUM_CLASSES}, 1280)"
        assert bias.shape == (NUM_CLASSES, BATCH_SIZE), \
            f"{gemmini_name}: bias {bias.shape} != ({NUM_CLASSES}, {BATCH_SIZE})"
        return
    p = conv_params[gemmini_name]
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
# Header file generation
# ---------------------------------------------------------------------------

def write_header(output_path, layers_data, conv_params, buffers):
    with open(output_path, "w") as f:
        f.write("#ifndef MOBILENET_CIFAR10_FLOAT_PARAMETERS_H\n")
        f.write("#define MOBILENET_CIFAR10_FLOAT_PARAMETERS_H\n\n")
        f.write("#include <include/gemmini_params.h>\n")
        f.write("#include <stdbool.h>\n\n")

        for gemmini_name, weight, bias, layer_type in layers_data:
            if layer_type == "fc":
                write_fc_layer(f, gemmini_name, weight, bias)
            elif layer_type == "dw":
                write_dw_layer(f, gemmini_name, weight, bias, conv_params, buffers)
            else:
                write_conv_layer(f, gemmini_name, weight, bias, conv_params, buffers)
            f.write("\n\n")

        f.write("#endif // MOBILENET_CIFAR10_FLOAT_PARAMETERS_H\n")


def write_conv_layer(f, name, weight, bias, conv_params, buffers):
    p = conv_params[name]
    f.write(f"static const elem_t {name}_w[{p['patch_size']}][{p['out_channels']}] row_align(1) = ")
    f.write(fmt_2d(weight))
    f.write(";\n")
    f.write(f"static const acc_t {name}_b[{p['out_channels']}] row_align_acc(1) = ")
    f.write(fmt_1d(bias))
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
    f.write(f".output_scale=1.0f, ")
    f.write(f".I={p['I']}, .J={p['J']}, .K={p['K']}, ")
    f.write(f".res_scale=1.0f")
    f.write("};\n")


def write_dw_layer(f, name, weight, bias, conv_params, buffers):
    p = conv_params[name]
    f.write(f"static const elem_t {name}_w[{p['out_channels']}][3][3] row_align(1) = ")
    f.write(fmt_3d(weight))
    f.write(";\n")
    f.write(f"static const acc_t {name}_b[{p['out_channels']}] row_align_acc(1) = ")
    f.write(fmt_1d(bias))
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
    f.write(f".output_scale=1.0f, ")
    f.write(f".res_scale=1.0f, ")
    f.write(f".I={p['I']}, .J={p['J']}")
    f.write("};\n")


def write_fc_layer(f, name, weight, bias):
    p = FC_PARAMS[name]
    out_f = p["out_features"]
    in_f = p["in_features"]
    f.write(f"static const elem_t {name}_w[{out_f}][{in_f}] row_align(1) = ")
    f.write(fmt_2d(weight))
    f.write(";\n")
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


# ---------------------------------------------------------------------------
# CIFAR-10 image generation
# ---------------------------------------------------------------------------

def generate_cifar10_images(output_path, indices=None):
    """Generate cifar10_images.h with 4 sample CIFAR-10 test images."""
    try:
        from torchvision.datasets import CIFAR10
    except ImportError:
        print("WARNING: torchvision not installed — skipping image generation.")
        print("  pip install torchvision")
        return None

    if indices is None:
        indices = [0, 1, 2, 3]

    dataset = CIFAR10(root="/tmp/cifar10_data", train=False, download=True)

    from PIL import Image as PILImage
    images = []
    labels = []
    for idx in indices:
        img_pil, label = dataset[idx]
        if INPUT_DIM != 32:
            img_pil = img_pil.resize((INPUT_DIM, INPUT_DIM), PILImage.BILINEAR)
        img_np = np.array(img_pil, dtype=np.int16)
        img_centered = np.clip(img_np - 128, -128, 127)
        images.append(img_centered)
        labels.append(label)

    with open(output_path, "w") as f:
        f.write("#ifndef CIFAR10_IMAGES_224_H\n")
        f.write("#define CIFAR10_IMAGES_224_H\n\n")
        f.write("#include <include/gemmini_params.h>\n\n")
        f.write(f"// CIFAR-10 test images at indices {indices}\n")
        cifar10_classes = ["airplane","automobile","bird","cat","deer",
                           "dog","frog","horse","ship","truck"]
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
        f"Expected {NUM_CLASSES} classes, got {actual_classes}"

    print("\nState dict keys:")
    for k in sorted(state_dict.keys()):
        print(f"  {k}: {list(state_dict[k].shape)}")

    # Compute CIFAR-10 spatial params
    conv_params = compute_conv_params()
    buffers = compute_buffers(conv_params)

    print(f"\nSpatial dimensions for {INPUT_DIM}x{INPUT_DIM} input:")
    for name, _k, _ic, _oc, _s, _p, _dw, _a in LAYER_ARCH:
        p = conv_params[name]
        print(f"  {name:15s}  {p['in_row_dim']:3d}x{p['in_col_dim']:<3d} -> "
              f"{p['out_row_dim']:3d}x{p['out_col_dim']:<3d}  "
              f"n_patches={p['n_patches']:5d}")

    # Extract all layers
    mapping = build_layer_mapping()
    layers_data = []
    print(f"\nExtracting {len(mapping)} layers...")
    for gemmini_name, hf_prefix, layer_type in mapping:
        print(f"  {gemmini_name:15s} <- {hf_prefix}")
        weight, bias = extract_layer(state_dict, gemmini_name, hf_prefix, layer_type)
        validate_shapes(gemmini_name, weight, bias, layer_type, conv_params)
        layers_data.append((gemmini_name, weight, bias, layer_type))

    # Write params header
    output_dir = os.path.dirname(os.path.abspath(__file__))
    params_path = os.path.join(output_dir, "..", "mobilenet_cifar10_params_float.h")
    params_path = os.path.normpath(params_path)
    print(f"\nWriting {params_path}...")
    write_header(params_path, layers_data, conv_params, buffers)

    # Generate CIFAR-10 images
    images_path = os.path.join(output_dir, "..", "cifar10_images_224.h")
    images_path = os.path.normpath(images_path)
    print(f"\nGenerating {images_path}...")
    labels = generate_cifar10_images(images_path)

    # Summary
    total_params = sum(w.size + b.size for _, w, b, _ in layers_data)
    print(f"\nTotal parameters: {total_params:,}")
    print("Done!")


if __name__ == "__main__":
    main()
