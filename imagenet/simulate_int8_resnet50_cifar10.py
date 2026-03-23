#!/usr/bin/env python3
"""
Simulated INT8 inference for ResNet-50 CIFAR-10 on Gemmini.

Exactly mimics what the Gemmini hardware does:
  1. INT8 matmul: acc = w_int8.T @ x_int8 + b_int32  (INT32 accumulator)
  2. Scale + requantize: y_int8 = clip(round(acc * output_scale), -128, 127)
  3. ResAdd: result = clip(round(skip * res_scale + main), -128, 127)
  4. ReLU: result = max(0, result) applied in CPU after resadd

Usage:
    conda run -n ImageNet_stable python simulate_int8_resnet50_cifar10.py
"""

import os
import sys
import numpy as np
import torch
from huggingface_hub import hf_hub_download

MODEL_NAME = "edadaltocg/resnet50_cifar10"
NUM_CLASSES = 10
NUM_TEST = 200

# ---------------------------------------------------------------------------
# BN folding & quantization
# ---------------------------------------------------------------------------

def fold_bn(conv_weight, bn_weight, bn_bias, bn_mean, bn_var, eps=1e-5):
    inv_std = 1.0 / np.sqrt(bn_var + eps)
    scale = bn_weight * inv_std
    shape = [conv_weight.shape[0]] + [1] * (conv_weight.ndim - 1)
    w_folded = conv_weight * scale.reshape(shape)
    b_folded = bn_bias - bn_weight * bn_mean * inv_std
    return w_folded, b_folded

def quantize_weight_int8(w_float):
    w_abs_max = np.max(np.abs(w_float))
    if w_abs_max < 1e-10:
        return np.zeros_like(w_float, dtype=np.int8), 1e-10
    w_scale = w_abs_max / 127.0
    w_int = np.clip(np.round(w_float / w_scale), -128, 127).astype(np.int8)
    return w_int, w_scale

def reshape_conv_weight(w):
    """Reshape PyTorch conv weight to Gemmini NHWC im2col format.
    PyTorch: [out_ch, C, kH, kW] -> Gemmini: [kH*kW*C, out_ch]
    """
    out_ch = w.shape[0]
    if w.ndim == 4:
        w = w.transpose(0, 2, 3, 1)  # [out_ch, C, kH, kW] -> [out_ch, kH, kW, C]
    return w.reshape(out_ch, -1).T

def quantize_bias_int32(b_float, combined_scale):
    if combined_scale < 1e-10:
        return np.zeros_like(b_float, dtype=np.int32)
    return np.clip(np.round(b_float / combined_scale), -(2**31), 2**31 - 1).astype(np.int32)

def estimate_activation_range(bn_gamma, bn_beta, has_relu):
    if has_relu:
        y_range = max(float(np.max(bn_beta + 3.0 * np.abs(bn_gamma))), 1.0)
    else:
        y_range = max(float(np.max(np.abs(bn_beta)) + 3.0 * np.max(np.abs(bn_gamma))), 1.0)
    return y_range

def compute_output_scale(w_scale, x_scale, y_range):
    y_scale = y_range / 127.0
    raw = (w_scale * x_scale) / y_scale
    if raw <= 0 or not np.isfinite(raw):
        raw = 1.0
    return raw

# ---------------------------------------------------------------------------
# im2col
# ---------------------------------------------------------------------------

def im2col(input_4d, kernel_size, stride, padding):
    """input_4d: [N, C, H, W] -> NHWC im2col [N*OH*OW, kH*kW*C], OH, OW
    Matches Gemmini's im2col patch ordering: [kH][kW][C]"""
    N, C, H, W = input_4d.shape
    kH = kW = kernel_size
    OH = (H + 2 * padding - kH) // stride + 1
    OW = (W + 2 * padding - kW) // stride + 1
    if padding > 0:
        input_4d = np.pad(input_4d, ((0,0),(0,0),(padding,padding),(padding,padding)),
                          mode='constant', constant_values=0)
    patches = np.zeros((N, OH, OW, kH * kW * C), dtype=input_4d.dtype)
    for i in range(OH):
        for j in range(OW):
            h_start = i * stride
            w_start = j * stride
            # Extract [N, C, kH, kW] patch, transpose to [N, kH, kW, C], flatten
            patch = input_4d[:, :, h_start:h_start+kH, w_start:w_start+kW]
            patch_nhwc = patch.transpose(0, 2, 3, 1)  # [N, kH, kW, C]
            patches[:, i, j, :] = patch_nhwc.reshape(N, -1)
    return patches.reshape(N * OH * OW, kH * kW * C), OH, OW

# ---------------------------------------------------------------------------
# Gemmini-equivalent ops
# ---------------------------------------------------------------------------

def gemmini_matmul(x_int8, w_int8, b_int32, output_scale):
    acc = x_int8.astype(np.int32) @ w_int8.astype(np.int32)
    acc = acc + b_int32.reshape(1, -1).astype(np.int32)
    return np.clip(np.round(acc.astype(np.float64) * output_scale), -128, 127).astype(np.int8)

def gemmini_resadd(skip_int8, main_int8, res_scale):
    result = skip_int8.astype(np.float64) * res_scale + main_int8.astype(np.float64)
    return np.clip(np.round(result), -128, 127).astype(np.int8)

def cpu_relu(x_int8):
    return np.maximum(x_int8, np.int8(0))

# ---------------------------------------------------------------------------
# Single conv layer: im2col + matmul + requantize
# ---------------------------------------------------------------------------

def run_conv_layer(input_nchw, layer_info):
    k = layer_info['kernel']
    s = layer_info['stride']
    N, C, H, W = input_nchw.shape

    if k == 1 and s == 1:
        x_flat = input_nchw.reshape(N, C, H * W).transpose(0, 2, 1).reshape(N * H * W, C)
        out_h, out_w = H, W
    elif k == 1 and s > 1:
        x_strided = input_nchw[:, :, ::s, ::s]
        N2, C2, H2, W2 = x_strided.shape
        x_flat = x_strided.reshape(N2, C2, H2 * W2).transpose(0, 2, 1).reshape(N2 * H2 * W2, C2)
        out_h, out_w = H2, W2
    else:
        x_flat, out_h, out_w = im2col(input_nchw, k, s, layer_info['padding'])

    y_flat = gemmini_matmul(x_flat, layer_info['w_int'], layer_info['b_int'], layer_info['output_scale'])
    out_ch = layer_info['w_int'].shape[1]
    return y_flat.reshape(N, out_h, out_w, out_ch).transpose(0, 3, 1, 2)

# ---------------------------------------------------------------------------
# All conv layers for quantization (flat order for scale propagation)
# ---------------------------------------------------------------------------

ALL_LAYERS = [
    ("conv_1",  "conv1", "bn1", 3, 1, 1, True),
    ("conv_2",  "layer1.0.conv1", "layer1.0.bn1", 1, 1, 0, True),
    ("conv_3",  "layer1.0.conv2", "layer1.0.bn2", 3, 1, 1, True),
    ("conv_4",  "layer1.0.conv3", "layer1.0.bn3", 1, 1, 0, False),
    ("conv_5",  "layer1.0.downsample.0", "layer1.0.downsample.1", 1, 1, 0, False),
    ("conv_6",  "layer1.1.conv1", "layer1.1.bn1", 1, 1, 0, True),
    ("conv_7",  "layer1.1.conv2", "layer1.1.bn2", 3, 1, 1, True),
    ("conv_8",  "layer1.1.conv3", "layer1.1.bn3", 1, 1, 0, False),
    ("conv_9",  "layer1.2.conv1", "layer1.2.bn1", 1, 1, 0, True),
    ("conv_10", "layer1.2.conv2", "layer1.2.bn2", 3, 1, 1, True),
    ("conv_11", "layer1.2.conv3", "layer1.2.bn3", 1, 1, 0, False),
    ("conv_12", "layer2.0.conv1", "layer2.0.bn1", 1, 1, 0, True),
    ("conv_13", "layer2.0.conv2", "layer2.0.bn2", 3, 2, 1, True),
    ("conv_14", "layer2.0.conv3", "layer2.0.bn3", 1, 1, 0, False),
    ("conv_15", "layer2.0.downsample.0", "layer2.0.downsample.1", 1, 2, 0, False),
    ("conv_16", "layer2.1.conv1", "layer2.1.bn1", 1, 1, 0, True),
    ("conv_17", "layer2.1.conv2", "layer2.1.bn2", 3, 1, 1, True),
    ("conv_18", "layer2.1.conv3", "layer2.1.bn3", 1, 1, 0, False),
    ("conv_19", "layer2.2.conv1", "layer2.2.bn1", 1, 1, 0, True),
    ("conv_20", "layer2.2.conv2", "layer2.2.bn2", 3, 1, 1, True),
    ("conv_21", "layer2.2.conv3", "layer2.2.bn3", 1, 1, 0, False),
    ("conv_22", "layer2.3.conv1", "layer2.3.bn1", 1, 1, 0, True),
    ("conv_23", "layer2.3.conv2", "layer2.3.bn2", 3, 1, 1, True),
    ("conv_24", "layer2.3.conv3", "layer2.3.bn3", 1, 1, 0, False),
    ("conv_25", "layer3.0.conv1", "layer3.0.bn1", 1, 1, 0, True),
    ("conv_26", "layer3.0.conv2", "layer3.0.bn2", 3, 2, 1, True),
    ("conv_27", "layer3.0.conv3", "layer3.0.bn3", 1, 1, 0, False),
    ("conv_28", "layer3.0.downsample.0", "layer3.0.downsample.1", 1, 2, 0, False),
    ("conv_29", "layer3.1.conv1", "layer3.1.bn1", 1, 1, 0, True),
    ("conv_30", "layer3.1.conv2", "layer3.1.bn2", 3, 1, 1, True),
    ("conv_31", "layer3.1.conv3", "layer3.1.bn3", 1, 1, 0, False),
    ("conv_32", "layer3.2.conv1", "layer3.2.bn1", 1, 1, 0, True),
    ("conv_33", "layer3.2.conv2", "layer3.2.bn2", 3, 1, 1, True),
    ("conv_34", "layer3.2.conv3", "layer3.2.bn3", 1, 1, 0, False),
    ("conv_35", "layer3.3.conv1", "layer3.3.bn1", 1, 1, 0, True),
    ("conv_36", "layer3.3.conv2", "layer3.3.bn2", 3, 1, 1, True),
    ("conv_37", "layer3.3.conv3", "layer3.3.bn3", 1, 1, 0, False),
    ("conv_38", "layer3.4.conv1", "layer3.4.bn1", 1, 1, 0, True),
    ("conv_39", "layer3.4.conv2", "layer3.4.bn2", 3, 1, 1, True),
    ("conv_40", "layer3.4.conv3", "layer3.4.bn3", 1, 1, 0, False),
    ("conv_41", "layer3.5.conv1", "layer3.5.bn1", 1, 1, 0, True),
    ("conv_42", "layer3.5.conv2", "layer3.5.bn2", 3, 1, 1, True),
    ("conv_43", "layer3.5.conv3", "layer3.5.bn3", 1, 1, 0, False),
    ("conv_44", "layer4.0.conv1", "layer4.0.bn1", 1, 1, 0, True),
    ("conv_45", "layer4.0.conv2", "layer4.0.bn2", 3, 2, 1, True),
    ("conv_46", "layer4.0.conv3", "layer4.0.bn3", 1, 1, 0, False),
    ("conv_47", "layer4.0.downsample.0", "layer4.0.downsample.1", 1, 2, 0, False),
    ("conv_48", "layer4.1.conv1", "layer4.1.bn1", 1, 1, 0, True),
    ("conv_49", "layer4.1.conv2", "layer4.1.bn2", 3, 1, 1, True),
    ("conv_50", "layer4.1.conv3", "layer4.1.bn3", 1, 1, 0, False),
    ("conv_51", "layer4.2.conv1", "layer4.2.bn1", 1, 1, 0, True),
    ("conv_52", "layer4.2.conv2", "layer4.2.bn2", 3, 1, 1, True),
    ("conv_53", "layer4.2.conv3", "layer4.2.bn3", 1, 1, 0, False),
]

RESIDUAL_SKIP = {
    "conv_4":  "conv_5",
    "conv_8":  "conv_4",
    "conv_11": "conv_8",
    "conv_14": "conv_15",
    "conv_18": "conv_14",
    "conv_21": "conv_18",
    "conv_24": "conv_21",
    "conv_27": "conv_28",
    "conv_31": "conv_27",
    "conv_34": "conv_31",
    "conv_37": "conv_34",
    "conv_40": "conv_37",
    "conv_43": "conv_40",
    "conv_46": "conv_47",
    "conv_50": "conv_46",
    "conv_53": "conv_50",
}

# Block definitions: each block has main_layers + optional projection skip
# For projection blocks: skip must be computed from block_input
# For identity blocks: skip is the previous block's resadd output
BLOCKS = [
    # Stage 0
    {"main": ["conv_2", "conv_3", "conv_4"], "proj": "conv_5", "skip_src": "conv_5"},
    {"main": ["conv_6", "conv_7", "conv_8"], "proj": None,     "skip_src": "conv_4"},
    {"main": ["conv_9", "conv_10", "conv_11"], "proj": None,   "skip_src": "conv_8"},
    # Stage 1
    {"main": ["conv_12", "conv_13", "conv_14"], "proj": "conv_15", "skip_src": "conv_15"},
    {"main": ["conv_16", "conv_17", "conv_18"], "proj": None,      "skip_src": "conv_14"},
    {"main": ["conv_19", "conv_20", "conv_21"], "proj": None,      "skip_src": "conv_18"},
    {"main": ["conv_22", "conv_23", "conv_24"], "proj": None,      "skip_src": "conv_21"},
    # Stage 2
    {"main": ["conv_25", "conv_26", "conv_27"], "proj": "conv_28", "skip_src": "conv_28"},
    {"main": ["conv_29", "conv_30", "conv_31"], "proj": None,      "skip_src": "conv_27"},
    {"main": ["conv_32", "conv_33", "conv_34"], "proj": None,      "skip_src": "conv_31"},
    {"main": ["conv_35", "conv_36", "conv_37"], "proj": None,      "skip_src": "conv_34"},
    {"main": ["conv_38", "conv_39", "conv_40"], "proj": None,      "skip_src": "conv_37"},
    {"main": ["conv_41", "conv_42", "conv_43"], "proj": None,      "skip_src": "conv_40"},
    # Stage 3
    {"main": ["conv_44", "conv_45", "conv_46"], "proj": "conv_47", "skip_src": "conv_47"},
    {"main": ["conv_48", "conv_49", "conv_50"], "proj": None,      "skip_src": "conv_46"},
    {"main": ["conv_51", "conv_52", "conv_53"], "proj": None,      "skip_src": "conv_50"},
]


def main():
    print("=" * 70)
    print("Simulated INT8 Inference for ResNet-50 CIFAR-10")
    print("=" * 70)

    # Load model weights
    print(f"\nLoading {MODEL_NAME}...")
    model_file = hf_hub_download(MODEL_NAME, "pytorch_model.bin")
    state_dict = {k: v.float().numpy() for k, v in
                  torch.load(model_file, map_location="cpu", weights_only=True).items()}

    # Load CIFAR-10 test set
    print("Loading CIFAR-10 test set...")
    from torchvision import datasets
    test_dataset = datasets.CIFAR10(root='/tmp/cifar10', train=False, download=True)
    test_images = test_dataset.data[:NUM_TEST]
    test_labels = np.array(test_dataset.targets[:NUM_TEST])

    # Preprocessing: same as prepare_cifar10.py cifar10standard
    mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float64)
    std  = np.array([0.2023, 0.1994, 0.2010], dtype=np.float64)
    FMAX = 2.75

    images_float = test_images.astype(np.float64) / 255.0
    images_norm = (images_float - mean) / std
    images_int8 = np.clip(np.round(images_norm / FMAX * 127), -128, 127).astype(np.int8)
    images_int8 = images_int8.transpose(0, 3, 1, 2)  # [N, 3, 32, 32]

    x_scale_0 = FMAX / 127.0
    print(f"Input x_scale_0 = {x_scale_0:.6f}")

    # Pre-compute y_range for all layers
    layer_y_range = {}
    for gn, ck, bk, k, s, p, relu in ALL_LAYERS:
        bn_g = state_dict[f"{bk}.weight"]
        bn_b = state_dict[f"{bk}.bias"]
        layer_y_range[gn] = estimate_activation_range(bn_g, bn_b, relu)

    # Quantize all layers
    print("Quantizing all layers...")
    layers = {}
    x_scale = x_scale_0

    for gn, ck, bk, k, s, p, relu in ALL_LAYERS:
        w = state_dict[f"{ck}.weight"]
        bn_g = state_dict[f"{bk}.weight"]
        bn_b = state_dict[f"{bk}.bias"]
        bn_m = state_dict[f"{bk}.running_mean"]
        bn_v = state_dict[f"{bk}.running_var"]
        w_folded, b_folded = fold_bn(w, bn_g, bn_b, bn_m, bn_v)

        w_reshaped = reshape_conv_weight(w_folded)

        w_int, w_scale = quantize_weight_int8(w_reshaped)
        combined_scale = w_scale * x_scale
        b_int = quantize_bias_int32(b_folded, combined_scale)

        y_range = layer_y_range[gn]
        output_scale = compute_output_scale(w_scale, x_scale, y_range)

        res_scale = 1.0
        if gn in RESIDUAL_SKIP:
            skip_src = RESIDUAL_SKIP[gn]
            res_scale = layer_y_range[skip_src] / y_range

        layers[gn] = {
            'w_int': w_int, 'b_int': b_int,
            'output_scale': output_scale,
            'kernel': k, 'stride': s, 'padding': p,
            'relu': relu, 'res_scale': res_scale,
        }

        print(f"  {gn:10s}  output_scale={output_scale:.4e}  res_scale={res_scale:.4f}  "
              f"{'RELU' if relu else 'LIN'}")

        x_scale = y_range / 127.0

    # FC layer
    fc_w = state_dict["fc.weight"]
    fc_b = state_dict["fc.bias"]
    fc_w_int, fc_w_scale = quantize_weight_int8(fc_w)
    fc_combined = fc_w_scale * x_scale
    fc_b_int = quantize_bias_int32(fc_b, fc_combined)
    print(f"  {'fc_54':10s}  w_scale={fc_w_scale:.6f}  x_scale={x_scale:.6f}")

    # ---------------------------------------------------------------------------
    # Run inference
    # ---------------------------------------------------------------------------
    print(f"\nRunning simulated INT8 inference on {NUM_TEST} images...")

    correct_top1 = 0
    correct_top5 = 0

    for img_idx in range(NUM_TEST):
        if img_idx % 50 == 0:
            print(f"  Image {img_idx}/{NUM_TEST}...")

        x = images_int8[img_idx:img_idx+1]  # [1, 3, 32, 32]

        # conv_1 (stem)
        conv1_out = run_conv_layer(x, layers["conv_1"])
        if layers["conv_1"]["relu"]:
            conv1_out = cpu_relu(conv1_out)

        stored = {"conv_1": conv1_out}
        block_input = conv1_out

        # Process residual blocks
        for block in BLOCKS:
            # Main path: sequential convolutions
            x = block_input
            for layer_name in block["main"]:
                x = run_conv_layer(x, layers[layer_name])
                if layers[layer_name]["relu"]:
                    x = cpu_relu(x)
                stored[layer_name] = x

            main_out_name = block["main"][-1]  # last conv in main path
            main_out = stored[main_out_name]

            # Skip path
            if block["proj"] is not None:
                proj_name = block["proj"]
                skip_out = run_conv_layer(block_input, layers[proj_name])
                if layers[proj_name]["relu"]:
                    skip_out = cpu_relu(skip_out)
                stored[proj_name] = skip_out
            else:
                skip_out = stored[block["skip_src"]]

            # ResAdd: result = clip(round(skip * res_scale + main), -128, 127)
            res_scale = layers[main_out_name]['res_scale']
            result = gemmini_resadd(skip_out, main_out, res_scale)
            result = cpu_relu(result)

            stored[main_out_name] = result
            block_input = result

        # Global Average Pool: [1, 2048, 4, 4] -> [1, 2048]
        final_out = block_input
        avg_pool = final_out.astype(np.float64).mean(axis=(2, 3))
        avg_int8 = np.clip(np.round(avg_pool), -128, 127).astype(np.int8)

        # FC: [1, 2048] @ [2048, 10] + bias
        fc_acc = avg_int8.astype(np.int32) @ fc_w_int.astype(np.int32).T
        fc_acc = fc_acc + fc_b_int.reshape(1, -1).astype(np.int32)

        logits = fc_acc[0].astype(np.float64)
        pred = np.argmax(logits)
        top5 = np.argsort(logits)[-5:]

        if pred == test_labels[img_idx]:
            correct_top1 += 1
        if test_labels[img_idx] in top5:
            correct_top5 += 1

    print(f"\n{'='*70}")
    print(f"Simulated INT8 Results ({NUM_TEST} images):")
    print(f"  Top-1 accuracy: {correct_top1}/{NUM_TEST} ({100*correct_top1/NUM_TEST:.1f}%)")
    print(f"  Top-5 accuracy: {correct_top5}/{NUM_TEST} ({100*correct_top5/NUM_TEST:.1f}%)")
    print(f"{'='*70}")

    if correct_top1 / NUM_TEST < 0.5:
        print("\n*** ACCURACY IS BAD - quantization itself is broken ***")
    else:
        print("\n*** Quantization looks OK ***")
        print("If FPGA accuracy is bad, the problem is in the C code / hardware interaction.")


if __name__ == "__main__":
    main()
