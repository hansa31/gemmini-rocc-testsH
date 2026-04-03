#!/usr/bin/env python3
"""
FP16 Simulation of MobileNetV2 inference on ImageNet.

This script answers: Does converting weights and activations to FP16
(with float32 accumulation, matching Gemmini hardware) cause the model
to degrade from 74% top-1 to predicting class 419?

If YES -> FP16 precision is the root cause.
If NO  -> the issue is Gemmini-hardware-specific or binary-file mismatch.

Also prints per-layer diagnostics (first 5 values, min, max, sum) that
can be compared with the C code output on the FPGA.

Usage:
  python test_fp16_simulation.py
  python test_fp16_simulation.py --bin-file /path/to/imagenet_val_10000.bin \
                                 --labels-file /path/to/imagenet_val_10000_labels.txt \
                                 --num-images 100
"""

import argparse
import os
import struct
import sys

import numpy as np
import torch
import torch.nn.functional as F
from transformers import MobileNetV2ForImageClassification, AutoImageProcessor

MODEL_NAME = "google/mobilenet_v2_1.0_224"
BN_EPS = 1e-3  # Actual MobileNetV2 BN epsilon (model uses eps=0.001)
BN_DEAD_VAR_THRESHOLD = 1e-3
IMAGE_SIZE = 224 * 224 * 3


# ==========================================================================
# FP16 simulation helpers
# ==========================================================================

def to_fp16(x):
    """Convert float32 numpy array to FP16 and back to float32.
    This simulates storing a value as elem_t (uint16 FP16 bit pattern)
    and reading it back. Truncation matches numpy's behavior.
    """
    return x.astype(np.float16).astype(np.float32)


def to_fp16_truncate(x):
    """Simulate the C code's float_to_fp16_bits (which truncates, not rounds).
    For proper comparison with the Gemmini hardware, we use truncation.
    """
    # numpy float16 rounds to nearest-even. For exact C match, we'd need
    # custom truncation. But the difference is at most 1 ULP in FP16.
    # Use numpy float16 for now -- if this works, truncation isn't the issue.
    return x.astype(np.float16).astype(np.float32)


# ==========================================================================
# BN folding (same as extraction script)
# ==========================================================================

def fold_bn(conv_w, bn_w, bn_b, bn_mean, bn_var, eps=BN_EPS):
    inv_std = 1.0 / np.sqrt(bn_var + eps)
    scale = bn_w * inv_std
    shape = [conv_w.shape[0]] + [1] * (conv_w.ndim - 1)
    w = conv_w * scale.reshape(shape)
    b = bn_b - bn_w * bn_mean * inv_std
    dead = bn_var < BN_DEAD_VAR_THRESHOLD
    if np.any(dead):
        w[dead] = 0.0
        b[dead] = 0.0
    return w, b


def get_bn_params(sd, prefix):
    return (
        sd[f"{prefix}.convolution.weight"].numpy(),
        sd[f"{prefix}.normalization.weight"].numpy(),
        sd[f"{prefix}.normalization.bias"].numpy(),
        sd[f"{prefix}.normalization.running_mean"].numpy(),
        sd[f"{prefix}.normalization.running_var"].numpy(),
    )


# ==========================================================================
# Convolution helper (float32 accumulation, like Gemmini hardware)
# ==========================================================================

def conv2d_fp16(x, w, b, stride=1, padding=0, groups=1, store_fp16=True):
    """Perform conv2d with float32 accumulation.
    Both x and w are in float32 (simulating their FP16 values already loaded).
    The accumulation is float32 (matching Gemmini acc_t = float).
    The output is optionally stored as FP16 (matching Gemmini elem_t write-back).
    """
    x_t = torch.from_numpy(x).float()
    w_t = torch.from_numpy(w).float()
    b_t = torch.from_numpy(b).float()
    out = F.conv2d(x_t, w_t, b_t, stride=stride, padding=padding, groups=groups)
    result = out.numpy()
    if store_fp16:
        result = to_fp16(result)
    return result


# ==========================================================================
# Layer diagnostics
# ==========================================================================

def layer_diag(name, x, img_idx=0):
    """Print per-layer diagnostics for one image in the batch."""
    single = x[img_idx] if x.ndim == 4 else x
    flat = single.flatten()
    first5 = flat[:5]
    return {
        "name": name,
        "min": float(flat.min()),
        "max": float(flat.max()),
        "sum": float(flat.sum()),
        "nonzero": int(np.count_nonzero(flat)),
        "total": len(flat),
        "first5": [float(v) for v in first5],
    }


def print_diag(d):
    f5 = ", ".join(f"{v:.4f}" for v in d["first5"])
    print(f"  [{d['name']:15s}] min={d['min']:.4f}, max={d['max']:.4f}, "
          f"sum={d['sum']:.2f}, nz={d['nonzero']}/{d['total']}")
    print(f"                   first5=[{f5}]")


# ==========================================================================
# Full inference pipeline
# ==========================================================================

def fp16_inference(sd, img_np, use_fp16=True, verbose=False):
    """Run MobileNetV2 with optional FP16 simulation.

    Args:
        sd: state_dict from HuggingFace model
        img_np: [1, 3, H, W] float32 numpy, already normalized
        use_fp16: if True, store weights/activations as FP16
        verbose: if True, print per-layer diagnostics
    """
    act_fn = lambda x: np.clip(x, 0, 6)  # ReLU6 always

    # Convert input to FP16 if simulating
    if use_fp16:
        x = to_fp16(img_np)
    else:
        x = img_np.copy()

    diagnostics = []

    if verbose:
        d = layer_diag("input", x)
        print_diag(d)
        diagnostics.append(d)

    # Stem: conv_1
    w, b = fold_bn(*get_bn_params(sd, "mobilenet_v2.conv_stem.first_conv"))
    if use_fp16:
        w, b_acc = to_fp16(w), b.astype(np.float32)  # bias stays float32 (acc_t)
    else:
        w, b_acc = w, b
    x = conv2d_fp16(x, w, b_acc, stride=2, padding=1, store_fp16=use_fp16)
    x = act_fn(x)
    if use_fp16:
        x = to_fp16(x)  # clamp_relu6 writes back as FP16
    if verbose:
        d = layer_diag("conv_1", x)
        print_diag(d)
        diagnostics.append(d)

    # Stem: conv_dw_2
    w, b = fold_bn(*get_bn_params(sd, "mobilenet_v2.conv_stem.conv_3x3"))
    if use_fp16:
        w, b_acc = to_fp16(w), b.astype(np.float32)
    else:
        w, b_acc = w, b
    x = conv2d_fp16(x, w, b_acc, stride=1, padding=1, groups=w.shape[0], store_fp16=use_fp16)
    x = act_fn(x)
    if use_fp16:
        x = to_fp16(x)
    if verbose:
        d = layer_diag("conv_dw_2", x)
        print_diag(d)
        diagnostics.append(d)

    # Stem: conv_3 (NO_ACTIVATION)
    w, b = fold_bn(*get_bn_params(sd, "mobilenet_v2.conv_stem.reduce_1x1"))
    if use_fp16:
        w, b_acc = to_fp16(w), b.astype(np.float32)
    else:
        w, b_acc = w, b
    x = conv2d_fp16(x, w, b_acc, store_fp16=use_fp16)
    # No activation
    if verbose:
        d = layer_diag("conv_3", x)
        print_diag(d)
        diagnostics.append(d)

    # Inverted residual blocks
    BLOCK_CONFIG = [
        (6, 24, 2, 2),
        (6, 32, 3, 2),
        (6, 64, 4, 2),
        (6, 96, 3, 1),
        (6, 160, 3, 2),
        (6, 320, 1, 1),
    ]

    hf_layer_idx = 0
    gemmini_idx = 4

    for t, out_c, repeats, first_stride in BLOCK_CONFIG:
        for rep in range(repeats):
            stride = first_stride if rep == 0 else 1
            residual = x if (stride == 1 and x.shape[1] == out_c) else None

            prefix = f"mobilenet_v2.layer.{hf_layer_idx}"

            # Expand 1x1
            w, b = fold_bn(*get_bn_params(sd, f"{prefix}.expand_1x1"))
            if use_fp16:
                w, b_acc = to_fp16(w), b.astype(np.float32)
            else:
                w, b_acc = w, b
            x = conv2d_fp16(x, w, b_acc, store_fp16=use_fp16)
            x = act_fn(x)
            if use_fp16:
                x = to_fp16(x)
            conv_name = f"conv_{gemmini_idx}"
            gemmini_idx += 1

            # Depthwise 3x3
            w, b = fold_bn(*get_bn_params(sd, f"{prefix}.conv_3x3"))
            if use_fp16:
                w, b_acc = to_fp16(w), b.astype(np.float32)
            else:
                w, b_acc = w, b
            x = conv2d_fp16(x, w, b_acc, stride=stride, padding=1,
                            groups=w.shape[0], store_fp16=use_fp16)
            x = act_fn(x)
            if use_fp16:
                x = to_fp16(x)
            dw_name = f"conv_dw_{gemmini_idx}"
            gemmini_idx += 1

            # Project 1x1 (NO_ACTIVATION)
            w, b = fold_bn(*get_bn_params(sd, f"{prefix}.reduce_1x1"))
            if use_fp16:
                w, b_acc = to_fp16(w), b.astype(np.float32)
            else:
                w, b_acc = w, b
            x = conv2d_fp16(x, w, b_acc, store_fp16=use_fp16)
            proj_name = f"conv_{gemmini_idx}"

            if residual is not None:
                x = x + residual
                if use_fp16:
                    x = to_fp16(x)  # resadd writes back as FP16

            if verbose and gemmini_idx in [9, 18, 30, 45, 51]:
                d = layer_diag(proj_name, x)
                print_diag(d)
                diagnostics.append(d)

            gemmini_idx += 1
            hf_layer_idx += 1

    # Final 1x1: conv_52
    w, b = fold_bn(*get_bn_params(sd, "mobilenet_v2.conv_1x1"))
    if use_fp16:
        w, b_acc = to_fp16(w), b.astype(np.float32)
    else:
        w, b_acc = w, b
    x = conv2d_fp16(x, w, b_acc, store_fp16=use_fp16)
    x = act_fn(x)
    if use_fp16:
        x = to_fp16(x)
    if verbose:
        d = layer_diag("conv_52", x)
        print_diag(d)
        diagnostics.append(d)

    # Global average pool
    avg = np.mean(x.astype(np.float32), axis=(2, 3), keepdims=True)
    if use_fp16:
        avg = to_fp16(avg)
    if verbose:
        d = layer_diag("average", avg)
        print_diag(d)
        diagnostics.append(d)

    # FC
    fc_w = sd["classifier.weight"].numpy()
    fc_b = sd["classifier.bias"].numpy()
    if fc_w.shape[0] == 1001:
        fc_w = fc_w[1:, :]
        fc_b = fc_b[1:]
    if use_fp16:
        fc_w = to_fp16(fc_w)
        # fc_b stays float32 (acc_t)
    logits = avg.reshape(1, -1) @ fc_w.T + fc_b  # float32 accumulation
    if use_fp16:
        logits = to_fp16(logits)

    if verbose:
        d = layer_diag("fc_53_out", logits)
        print_diag(d)
        diagnostics.append(d)

    return logits[0], diagnostics


# ==========================================================================
# Binary file reading (same as C code)
# ==========================================================================

def read_image_from_bin(fp, image_size=IMAGE_SIZE):
    """Read one image from binary. Returns (224, 224, 3) float32 normalized."""
    raw = np.frombuffer(fp.read(image_size), dtype=np.int8)
    if len(raw) < image_size:
        return None
    # Same normalization as C code:
    # float pixel = (float)((int)raw_batch[i] + 128);
    # batch_images[i] = float_to_elem_bits(pixel / 127.5f - 1.0f);
    pixel = raw.astype(np.float32) + 128.0  # recover uint8 [0, 255]
    normalized = pixel / 127.5 - 1.0  # [-1, 1]
    return normalized.reshape(224, 224, 3)


def read_labels(path, num):
    labels = []
    with open(path) as f:
        for line in f:
            labels.append(int(line.strip()))
            if len(labels) >= num:
                break
    return labels


# ==========================================================================
# Main
# ==========================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin-file", default=None,
                        help="Path to imagenet binary (e.g. imagenet_val_10000.bin)")
    parser.add_argument("--labels-file", default=None,
                        help="Path to labels file")
    parser.add_argument("--num-images", type=int, default=20,
                        help="Number of images to test")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-layer diagnostics for first image")
    args = parser.parse_args()

    print(f"Loading {MODEL_NAME}...")
    model = MobileNetV2ForImageClassification.from_pretrained(MODEL_NAME)
    model.eval()
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    # ===== Test 1: Synthetic images (same as C code images.h) =====
    print(f"\n{'='*70}")
    print("TEST 1: Synthetic images (matching images.h)")
    print(f"{'='*70}")
    rng = np.random.RandomState(42)
    for img_idx in range(4):
        arr = rng.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        # Normal normalization: pixel / 127.5 - 1.0
        img_fp32 = arr.astype(np.float32) / 127.5 - 1.0
        img_nchw = img_fp32.transpose(2, 0, 1)[np.newaxis]

        logits_f32, _ = fp16_inference(sd, img_nchw, use_fp16=False, verbose=False)
        logits_fp16, diags = fp16_inference(sd, img_nchw, use_fp16=True,
                                             verbose=(img_idx == 0 and True))
        pred_f32 = int(np.argmax(logits_f32))
        pred_fp16 = int(np.argmax(logits_fp16))

        top5_f32 = np.argsort(logits_f32)[-5:][::-1]
        top5_fp16 = np.argsort(logits_fp16)[-5:][::-1]

        print(f"\n  Image {img_idx}:")
        print(f"    Float32: pred={pred_f32}, top5={list(top5_f32)}")
        print(f"    FP16 sim: pred={pred_fp16}, top5={list(top5_fp16)}")
        if logits_f32.shape == logits_fp16.shape:
            corr = float(np.corrcoef(logits_f32, logits_fp16)[0, 1])
            print(f"    Logit correlation: {corr:.6f}")

    # ===== Test 2: Binary file images =====
    if args.bin_file and os.path.isfile(args.bin_file):
        labels = []
        if args.labels_file and os.path.isfile(args.labels_file):
            labels = read_labels(args.labels_file, args.num_images)

        print(f"\n{'='*70}")
        print(f"TEST 2: Binary file images ({args.num_images} from {args.bin_file})")
        print(f"{'='*70}")

        top1_f32 = 0
        top1_fp16 = 0
        top5_f32_cnt = 0
        top5_fp16_cnt = 0

        with open(args.bin_file, "rb") as fp:
            for i in range(args.num_images):
                img_hwc = read_image_from_bin(fp)
                if img_hwc is None:
                    print(f"  EOF at image {i}")
                    break
                img_nchw = img_hwc.transpose(2, 0, 1)[np.newaxis]

                verbose = (i == 0 and args.verbose)
                if verbose:
                    # Print raw input stats
                    flat = img_hwc.flatten()
                    print(f"\n  Image 0 input: min={flat.min():.4f}, max={flat.max():.4f}, "
                          f"mean={flat.mean():.4f}")
                    print(f"    first10_raw = [{', '.join(f'{v:.4f}' for v in flat[:10])}]")

                logits_f32, _ = fp16_inference(sd, img_nchw, use_fp16=False, verbose=False)
                logits_fp16, _ = fp16_inference(sd, img_nchw, use_fp16=True, verbose=verbose)

                pred_f32 = int(np.argmax(logits_f32))
                pred_fp16 = int(np.argmax(logits_fp16))

                if i < labels.__len__():
                    label = labels[i]
                    top5_f32_idx = np.argsort(logits_f32)[-5:][::-1]
                    top5_fp16_idx = np.argsort(logits_fp16)[-5:][::-1]
                    if pred_f32 == label:
                        top1_f32 += 1
                    if label in top5_f32_idx:
                        top5_f32_cnt += 1
                    if pred_fp16 == label:
                        top1_fp16 += 1
                    if label in top5_fp16_idx:
                        top5_fp16_cnt += 1

                    if i < 8:
                        print(f"  Image {i} (label={label}): f32_pred={pred_f32}, "
                              f"fp16_pred={pred_fp16}"
                              f"{'  <-- f32 correct' if pred_f32 == label else ''}"
                              f"{'  <-- fp16 correct' if pred_fp16 == label else ''}")

                if (i + 1) % 20 == 0:
                    n = i + 1
                    print(f"  [{n}/{args.num_images}] "
                          f"Float32 top-1: {top1_f32}/{n} ({100*top1_f32/n:.1f}%), "
                          f"top-5: {top5_f32_cnt}/{n} ({100*top5_f32_cnt/n:.1f}%) | "
                          f"FP16 top-1: {top1_fp16}/{n} ({100*top1_fp16/n:.1f}%), "
                          f"top-5: {top5_fp16_cnt}/{n} ({100*top5_fp16_cnt/n:.1f}%)")

        n = min(args.num_images, len(labels))
        if n > 0:
            print(f"\n  --- FINAL ({n} images) ---")
            print(f"  Float32: top-1 = {top1_f32}/{n} ({100*top1_f32/n:.1f}%), "
                  f"top-5 = {top5_f32_cnt}/{n} ({100*top5_f32_cnt/n:.1f}%)")
            print(f"  FP16 sim: top-1 = {top1_fp16}/{n} ({100*top1_fp16/n:.1f}%), "
                  f"top-5 = {top5_fp16_cnt}/{n} ({100*top5_fp16_cnt/n:.1f}%)")
            print()
            if top1_fp16 < top1_f32 * 0.5:
                print("  *** FP16 accuracy is significantly degraded! ***")
                print("  This means FP16 precision loss (not Gemmini hardware) is the issue.")
                print("  Consider: mixed precision, FP16 quantization-aware training, or BF16.")
            else:
                print("  FP16 simulation accuracy is comparable to float32.")
                print("  The issue is likely Gemmini-hardware-specific or binary file mismatch.")
    else:
        # Try default locations
        for candidate in [
            "/home/hansa/Downloads/test1/imagenet_val_50000.bin",
            "/home/hansa/Downloads/10K/imagenet_val_10000.bin",
        ]:
            if os.path.isfile(candidate):
                print(f"\n  Found binary at: {candidate}")
                print(f"  Re-run with: python {sys.argv[0]} --bin-file {candidate} "
                      f"--labels-file <labels.txt> --num-images 100 --verbose")
                break


if __name__ == "__main__":
    main()
