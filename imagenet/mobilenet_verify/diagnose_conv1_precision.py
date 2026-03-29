#!/usr/bin/env python3
"""Diagnose conv_1 (first layer) INT8 precision in MobileNetV2 pixel-128 mode.

In pixel-128 mode the ImageNet normalization is folded into conv_1 weights:
    w_folded[:,c,:,:] /= (255 * std[c])
    b_folded += sum_over_spatial(w[:,c,:,:]) * ((128/255 - mean[c]) / std[c])

This shrinks weights by ~57x, so per-tensor INT8 quantization wastes most of
the dynamic range.  This script quantifies the problem and tests mitigations.

Usage:
    conda run -n ImageNet python diagnose_conv1_precision.py
"""

import numpy as np
import torch
from transformers import MobileNetV2ForImageClassification
import cv2
import sys

# ============================================================================
# Constants
# ============================================================================
MODEL_NAME    = "google/mobilenet_v2_1.0_224"
BN_EPS        = 0.001
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float64)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float64)
IMAGE_PATH    = "/home/hansa/Downloads/Images200/ILSVRC2012_val_00000001.JPEG"

# ============================================================================
# Load model
# ============================================================================
print("=" * 78)
print("LOADING MODEL")
print("=" * 78)
model = MobileNetV2ForImageClassification.from_pretrained(MODEL_NAME)
model.eval()
sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}

# ============================================================================
# Helpers
# ============================================================================

def fold_bn(w, g, b, m, v, eps=BN_EPS):
    """Fold BatchNorm into conv weights: w_folded, b_folded."""
    inv = 1.0 / np.sqrt(v + eps)
    s = g * inv
    shape = [w.shape[0]] + [1] * (w.ndim - 1)
    return w * s.reshape(shape), b - g * m * inv


def get_conv_bn(sd, prefix):
    """Load conv+BN params from HuggingFace state dict."""
    keys = ["convolution.weight", "normalization.weight", "normalization.bias",
            "normalization.running_mean", "normalization.running_var"]
    return tuple(sd[f"{prefix}.{k}"].float().numpy().astype(np.float64) for k in keys)


def quantize_weight_int8(w):
    """Per-tensor symmetric INT8 quantization. Returns (w_int8, w_scale)."""
    mx = float(np.max(np.abs(w)))
    if mx < 1e-10:
        return np.zeros_like(w, dtype=np.int8), 1e-10
    s = mx / 127.0
    return np.clip(np.round(w / s), -128, 127).astype(np.int8), s


def quantize_bias_int32(b, combined_scale):
    """Quantize bias with combined_scale = w_scale * x_scale."""
    if combined_scale < 1e-10:
        return np.zeros_like(b, dtype=np.int32)
    return np.clip(np.round(b / combined_scale), -(2**31), 2**31 - 1).astype(np.int32)


def im2col(x_nchw, kernel, stride, padding):
    """Extract im2col patches in (kH, kW, C) order for Gemmini."""
    N, C, H, W = x_nchw.shape
    kH = kW = kernel
    OH = (H + 2 * padding - kH) // stride + 1
    OW = (W + 2 * padding - kW) // stride + 1
    if padding > 0:
        x_nchw = np.pad(x_nchw,
                        ((0, 0), (0, 0), (padding, padding), (padding, padding)),
                        mode='constant', constant_values=0)
    patches = np.zeros((N, OH, OW, kH * kW * C), dtype=x_nchw.dtype)
    for i in range(OH):
        for j in range(OW):
            patch = x_nchw[:, :, i*stride:i*stride+kH, j*stride:j*stride+kW]
            # Gemmini order: (kH, kW, C) -> flatten
            patches[:, i, j, :] = patch.transpose(0, 2, 3, 1).reshape(N, -1)
    return patches.reshape(N * OH * OW, kH * kW * C), OH, OW


def reshape_conv_weight(w):
    """[out_ch, in_ch, kH, kW] -> [patch_size, out_ch] in (kH,kW,in_ch) order."""
    out_ch = w.shape[0]
    if w.ndim == 4:
        w = w.transpose(0, 2, 3, 1)  # [out_ch, kH, kW, in_ch]
    return w.reshape(out_ch, -1).T


def gemmini_conv(x_nchw, w_int, b_int, output_scale, kernel, stride, padding):
    """Gemmini-equivalent regular conv via im2col + matmul."""
    N, C, H, W = x_nchw.shape
    x_flat, out_h, out_w = im2col(x_nchw, kernel, stride, padding)
    acc = x_flat.astype(np.int32) @ w_int.astype(np.int32)
    acc += b_int.reshape(1, -1).astype(np.int32)
    y_flat = np.clip(np.round(acc.astype(np.float64) * output_scale),
                     -128, 127).astype(np.int8)
    out_ch = w_int.shape[1]
    return y_flat.reshape(N, out_h, out_w, out_ch).transpose(0, 3, 1, 2)


def gemmini_conv_raw_acc(x_nchw, w_int, b_int, kernel, stride, padding):
    """Same as gemmini_conv but return raw INT32 accumulator (no output scaling)."""
    N, C, H, W = x_nchw.shape
    x_flat, out_h, out_w = im2col(x_nchw, kernel, stride, padding)
    acc = x_flat.astype(np.int32) @ w_int.astype(np.int32)
    acc += b_int.reshape(1, -1).astype(np.int32)
    out_ch = w_int.shape[1]
    return acc.reshape(N, out_h, out_w, out_ch).transpose(0, 3, 1, 2)


def per_channel_corr(float_ref, int8_dequant):
    """Compute per-channel Pearson correlation between float reference and
    dequantized INT8 output.  Both are [N, C, H, W]."""
    C = float_ref.shape[1]
    corrs = np.zeros(C, dtype=np.float64)
    for c in range(C):
        f = float_ref[0, c].flatten()
        q = int8_dequant[0, c].flatten()
        if np.std(f) < 1e-12 or np.std(q) < 1e-12:
            corrs[c] = 0.0
        else:
            corrs[c] = np.corrcoef(f, q)[0, 1]
    return corrs


def per_channel_mae(float_ref, int8_dequant):
    """Per-channel mean absolute error."""
    C = float_ref.shape[1]
    maes = np.zeros(C, dtype=np.float64)
    for c in range(C):
        maes[c] = np.mean(np.abs(float_ref[0, c] - int8_dequant[0, c]))
    return maes


# ============================================================================
# Step 0: Load image and prepare both input representations
# ============================================================================
print("\n" + "=" * 78)
print("STEP 0: Load image and prepare inputs")
print("=" * 78)

img_bgr = cv2.imread(IMAGE_PATH)
if img_bgr is None:
    print(f"ERROR: cannot read {IMAGE_PATH}")
    sys.exit(1)
img_224 = cv2.resize(img_bgr, (224, 224))
img_rgb = cv2.cvtColor(img_224, cv2.COLOR_BGR2RGB)

# Float normalized input (standard ImageNet preprocessing)
img_f = img_rgb.astype(np.float64) / 255.0
img_norm = np.empty_like(img_f)
for c in range(3):
    img_norm[:, :, c] = (img_f[:, :, c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
x_float_np = img_norm.transpose(2, 0, 1)[np.newaxis]  # [1,3,224,224] float
x_float_torch = torch.tensor(x_float_np, dtype=torch.float32)

# Pixel-128 INT8 input
img_pixel128 = (img_rgb.astype(np.int16) - 128).clip(-128, 127).astype(np.int8)
x_pixel128 = img_pixel128.transpose(2, 0, 1)[np.newaxis]  # [1,3,224,224] int8

print(f"  Image: {IMAGE_PATH}")
print(f"  Float input range: [{x_float_np.min():.3f}, {x_float_np.max():.3f}]")
print(f"  Pixel-128 range:   [{x_pixel128.min()}, {x_pixel128.max()}]")

# ============================================================================
# Step 1: Capture float conv_1 outputs (pre-ReLU6 and post-ReLU6)
# ============================================================================
print("\n" + "=" * 78)
print("STEP 1: Float reference activations (conv_1)")
print("=" * 78)

float_acts = {}

def capture(name):
    def hook(module, inp, out):
        float_acts[name] = out.detach().float().numpy().astype(np.float64)
    return hook

# Hook into conv_1 BN output (= pre-ReLU6)
parts = "mobilenet_v2.conv_stem.first_conv.normalization".split(".")
mod = model
for p in parts:
    mod = getattr(mod, p)
h = mod.register_forward_hook(capture("conv_1_bn"))

with torch.no_grad():
    model(x_float_torch)
h.remove()

float_conv1_pre_relu6 = float_acts["conv_1_bn"]          # [1,32,112,112]
float_conv1_post_relu6 = np.clip(float_conv1_pre_relu6, 0, 6.0)

print(f"  conv_1 pre-ReLU6 range:  [{float_conv1_pre_relu6.min():.4f}, "
      f"{float_conv1_pre_relu6.max():.4f}]")
print(f"  conv_1 post-ReLU6 range: [{float_conv1_post_relu6.min():.4f}, "
      f"{float_conv1_post_relu6.max():.4f}]")
print(f"  Shape: {float_conv1_post_relu6.shape}")

# Per-channel statistics
for c in range(32):
    ch = float_conv1_post_relu6[0, c]
    zero_frac = np.mean(ch == 0) * 100
    if c < 8 or c >= 28:
        print(f"    ch{c:02d}: mean={ch.mean():.4f}  max={ch.max():.4f}  "
              f"zero%={zero_frac:.1f}%")
if 32 > 16:
    print(f"    ... (showing first 8 and last 4 of 32 channels)")

# ============================================================================
# Step 2: conv_1 weight analysis (standard vs pixel-128 folded)
# ============================================================================
print("\n" + "=" * 78)
print("STEP 2: conv_1 weight analysis")
print("=" * 78)

conv1_w, bn1_g, bn1_b, bn1_m, bn1_v = get_conv_bn(
    sd, "mobilenet_v2.conv_stem.first_conv")

# --- 2a: Standard BN-folded weights (no pixel-128 folding) ---
w_std, b_std = fold_bn(conv1_w, bn1_g, bn1_b, bn1_m, bn1_v)
wg_std = reshape_conv_weight(w_std)  # [27, 32]
w_std_max = np.max(np.abs(wg_std))
w_std_scale = w_std_max / 127.0

print(f"\n  [2a] Standard BN-folded weights (NOT pixel-128):")
print(f"    Shape: conv1_w = {conv1_w.shape} -> reshaped = {wg_std.shape}")
print(f"    w_max  = {w_std_max:.6f}")
print(f"    w_scale = {w_std_scale:.6f}")

# --- 2b: Pixel-128 folded weights ---
w_p128, b_p128 = fold_bn(conv1_w, bn1_g, bn1_b, bn1_m, bn1_v)
# Fold the normalization offset into bias
norm_offset = (128.0 / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
for c in range(3):
    b_p128 += w_p128[:, c, :, :].sum(axis=(1, 2)) * norm_offset[c]
# Fold the input scaling into weights
for c in range(3):
    w_p128[:, c, :, :] /= (255.0 * IMAGENET_STD[c])

wg_p128 = reshape_conv_weight(w_p128)  # [27, 32]
w_p128_max = np.max(np.abs(wg_p128))
w_p128_scale = w_p128_max / 127.0

print(f"\n  [2b] Pixel-128 folded weights:")
print(f"    Shape: {wg_p128.shape}")
print(f"    w_max   = {w_p128_max:.6f}")
print(f"    w_scale = {w_p128_scale:.6f}")
print(f"    Magnitude reduction: {w_std_max / w_p128_max:.1f}x")

# --- 2c: Quantization precision comparison ---
w_std_int, _ = quantize_weight_int8(wg_std)
w_p128_int, _ = quantize_weight_int8(wg_p128)

std_unique_levels = len(np.unique(w_std_int))
p128_unique_levels = len(np.unique(w_p128_int))
std_nonzero = np.count_nonzero(w_std_int)
p128_nonzero = np.count_nonzero(w_p128_int)

print(f"\n  [2c] Quantization precision comparison:")
print(f"    Standard:  {std_unique_levels} unique INT8 levels, "
      f"{std_nonzero}/{w_std_int.size} non-zero weights")
print(f"    Pixel-128: {p128_unique_levels} unique INT8 levels, "
      f"{p128_nonzero}/{w_p128_int.size} non-zero weights")

# ============================================================================
# Step 3: Per-output-channel weight diagnostics (32 channels)
# ============================================================================
print("\n" + "=" * 78)
print("STEP 3: Per-output-channel weight diagnostics (conv_1, 32 channels)")
print("=" * 78)

num_out_ch = 32
patch_size = 27  # 3*3*3

print(f"\n  {'Ch':>3s}  {'w_mag_std':>10s}  {'w_mag_p128':>10s}  "
      f"{'pc_scale':>10s}  {'pt_scale':>10s}  {'eff_lvls':>8s}  "
      f"{'nz/27':>5s}  {'q_err_rms':>10s}")
print("  " + "-" * 80)

per_ch_scales_p128 = np.zeros(num_out_ch)
per_ch_eff_levels = np.zeros(num_out_ch, dtype=int)
per_ch_nonzero = np.zeros(num_out_ch, dtype=int)
per_ch_qerr_rms = np.zeros(num_out_ch)

for ch in range(num_out_ch):
    # Weight column for this output channel
    w_col_std = wg_std[:, ch]       # [27]
    w_col_p128 = wg_p128[:, ch]     # [27]

    w_mag_std = np.max(np.abs(w_col_std))
    w_mag_p128 = np.max(np.abs(w_col_p128))

    # Per-channel scale (what it would be if each channel had its own scale)
    pc_scale = w_mag_p128 / 127.0 if w_mag_p128 > 1e-10 else 1e-10
    per_ch_scales_p128[ch] = pc_scale

    # Quantize this channel's weights with per-tensor scale
    w_col_int_pt = np.clip(np.round(w_col_p128 / w_p128_scale), -128, 127).astype(np.int8)
    # Effective unique levels used by this channel
    eff = len(np.unique(w_col_int_pt))
    per_ch_eff_levels[ch] = eff

    # Non-zero count
    nz = np.count_nonzero(w_col_int_pt)
    per_ch_nonzero[ch] = nz

    # Weight quantization error (RMS)
    w_dequant = w_col_int_pt.astype(np.float64) * w_p128_scale
    qerr = np.sqrt(np.mean((w_col_p128 - w_dequant) ** 2))
    per_ch_qerr_rms[ch] = qerr

    print(f"  {ch:3d}  {w_mag_std:10.6f}  {w_mag_p128:10.6f}  "
          f"{pc_scale:10.6e}  {w_p128_scale:10.6e}  {eff:8d}  "
          f"{nz:3d}/27  {qerr:10.6e}")

print(f"\n  Summary:")
print(f"    Per-tensor w_scale (pixel-128): {w_p128_scale:.6e}")
print(f"    Per-channel w_scale range:      [{per_ch_scales_p128.min():.6e}, "
      f"{per_ch_scales_p128.max():.6e}]")
print(f"    Scale ratio (max/min ch):       "
      f"{per_ch_scales_p128.max() / max(per_ch_scales_p128.min(), 1e-20):.1f}x")
print(f"    Channels with <=3 non-zero weights: "
      f"{np.sum(per_ch_nonzero <= 3)}")
print(f"    Channels with <=5 effective levels: "
      f"{np.sum(per_ch_eff_levels <= 5)}")

# ============================================================================
# Step 4: Full INT8 conv_1 (pixel-128 mode) vs float
# ============================================================================
print("\n" + "=" * 78)
print("STEP 4: Full INT8 conv_1 (pixel-128) vs float reference")
print("=" * 78)

# Quantize pixel-128 weights + bias
w1_int_p128, w1_scale_p128 = quantize_weight_int8(wg_p128)
x_scale_p128 = 128.0 / 127.0  # pixel-128 input: max |x| = 128, scale = 128/127
combined_scale_p128 = w1_scale_p128 * x_scale_p128
b1_int_p128 = quantize_bias_int32(b_p128, combined_scale_p128)

# Output range: ReLU6, so y_range = 6.0
yr1 = 6.0
y_scale = yr1 / 127.0
os1_p128 = combined_scale_p128 / y_scale

print(f"  w_scale  = {w1_scale_p128:.6e}")
print(f"  x_scale  = {x_scale_p128:.6e}")
print(f"  combined = {combined_scale_p128:.6e}")
print(f"  y_range  = {yr1}")
print(f"  output_scale = {os1_p128:.6e}")

# Run INT8 conv_1
out1_p128 = gemmini_conv(x_pixel128, w1_int_p128, b1_int_p128, os1_p128, 3, 2, 1)
out1_p128_relu = np.maximum(out1_p128, np.int8(0))  # ReLU
# (Note: ReLU6 clipping happens implicitly via y_range = 6.0)

# Dequantize for comparison
out1_p128_dequant = out1_p128_relu.astype(np.float64) * y_scale

# Overall correlation
overall_corr = np.corrcoef(float_conv1_post_relu6.flatten(),
                            out1_p128_dequant.flatten())[0, 1]
overall_mae = np.mean(np.abs(float_conv1_post_relu6 - out1_p128_dequant))
print(f"\n  Overall correlation (post-ReLU6): {overall_corr:.6f}")
print(f"  Overall MAE:                      {overall_mae:.6f}")

# Per-channel analysis
ch_corrs_p128 = per_channel_corr(float_conv1_post_relu6, out1_p128_dequant)
ch_maes_p128 = per_channel_mae(float_conv1_post_relu6, out1_p128_dequant)

print(f"\n  Per-channel correlation (pixel-128 conv_1):")
print(f"  {'Ch':>3s}  {'Corr':>8s}  {'MAE':>8s}  {'FloatMax':>9s}  "
      f"{'INT8Max':>8s}  {'Dead?':>5s}")
print("  " + "-" * 55)

dead_channels = []
for ch in range(num_out_ch):
    f_max = float_conv1_post_relu6[0, ch].max()
    i_max = out1_p128_dequant[0, ch].max()
    dead = "YES" if (ch_corrs_p128[ch] < 0.1 or i_max < 1e-6) else ""
    if dead:
        dead_channels.append(ch)
    print(f"  {ch:3d}  {ch_corrs_p128[ch]:8.4f}  {ch_maes_p128[ch]:8.4f}  "
          f"{f_max:9.4f}  {i_max:8.4f}  {dead:>5s}")

print(f"\n  Dead/very-low-correlation channels: {dead_channels}")
print(f"  Channels with corr < 0.5: "
      f"{sorted(np.where(ch_corrs_p128 < 0.5)[0].tolist())}")
print(f"  Mean per-channel corr: {ch_corrs_p128.mean():.4f}")
print(f"  Median per-channel corr: {np.median(ch_corrs_p128):.4f}")

# ============================================================================
# Step 5: CRITICAL COMPARISON -- standard mode vs pixel-128 mode
# ============================================================================
print("\n" + "=" * 78)
print("STEP 5: CRITICAL COMPARISON -- Standard vs Pixel-128 mode")
print("=" * 78)

# --- 5a: Standard mode ---
# Quantize the float normalized input
fmax = float(np.max(np.abs(x_float_np)))
x_scale_std = fmax / 127.0
x_std_int8 = np.clip(np.round(x_float_np / x_scale_std),
                      -128, 127).astype(np.int8)

# Standard BN-folded weights (no pixel-128 folding)
w1_int_std, w1_scale_std = quantize_weight_int8(wg_std)
combined_scale_std = w1_scale_std * x_scale_std
b1_int_std = quantize_bias_int32(b_std, combined_scale_std)
os1_std = combined_scale_std / y_scale

print(f"\n  [5a] Standard mode:")
print(f"    Float input fmax = {fmax:.4f}")
print(f"    x_scale  = {x_scale_std:.6e}")
print(f"    w_scale  = {w1_scale_std:.6e}")
print(f"    combined = {combined_scale_std:.6e}")
print(f"    output_scale = {os1_std:.6e}")

out1_std = gemmini_conv(x_std_int8, w1_int_std, b1_int_std, os1_std, 3, 2, 1)
out1_std_relu = np.maximum(out1_std, np.int8(0))
out1_std_dequant = out1_std_relu.astype(np.float64) * y_scale

corr_std = np.corrcoef(float_conv1_post_relu6.flatten(),
                        out1_std_dequant.flatten())[0, 1]
mae_std = np.mean(np.abs(float_conv1_post_relu6 - out1_std_dequant))

ch_corrs_std = per_channel_corr(float_conv1_post_relu6, out1_std_dequant)

print(f"    Overall correlation: {corr_std:.6f}")
print(f"    Overall MAE:         {mae_std:.6f}")
print(f"    Mean ch corr:        {ch_corrs_std.mean():.4f}")
print(f"    Median ch corr:      {np.median(ch_corrs_std):.4f}")

# --- 5b: Pixel-128 mode (already computed in Step 4) ---
print(f"\n  [5b] Pixel-128 mode (from Step 4):")
print(f"    x_scale  = {x_scale_p128:.6e}")
print(f"    w_scale  = {w1_scale_p128:.6e}")
print(f"    combined = {combined_scale_p128:.6e}")
print(f"    output_scale = {os1_p128:.6e}")
print(f"    Overall correlation: {overall_corr:.6f}")
print(f"    Overall MAE:         {overall_mae:.6f}")
print(f"    Mean ch corr:        {ch_corrs_p128.mean():.4f}")
print(f"    Median ch corr:      {np.median(ch_corrs_p128):.4f}")

# --- 5c: Side-by-side per-channel comparison ---
print(f"\n  [5c] Per-channel correlation comparison:")
print(f"  {'Ch':>3s}  {'Standard':>9s}  {'Pixel128':>9s}  {'Delta':>8s}  {'Verdict':>12s}")
print("  " + "-" * 50)

for ch in range(num_out_ch):
    delta = ch_corrs_std[ch] - ch_corrs_p128[ch]
    if delta > 0.3:
        verdict = "MUCH WORSE"
    elif delta > 0.1:
        verdict = "WORSE"
    elif delta > 0.02:
        verdict = "slightly worse"
    elif abs(delta) <= 0.02:
        verdict = "similar"
    else:
        verdict = "better"
    print(f"  {ch:3d}  {ch_corrs_std[ch]:9.4f}  {ch_corrs_p128[ch]:9.4f}  "
          f"{delta:+8.4f}  {verdict:>12s}")

improvement = ch_corrs_std.mean() - ch_corrs_p128.mean()
print(f"\n  VERDICT: Standard mode mean corr = {ch_corrs_std.mean():.4f}, "
      f"Pixel-128 = {ch_corrs_p128.mean():.4f}")
print(f"           Difference = {improvement:+.4f}")
if improvement > 0.05:
    print("           --> Pixel-128 folding IS a significant source of error!")
elif improvement > 0.01:
    print("           --> Pixel-128 folding causes moderate additional error.")
else:
    print("           --> Pixel-128 folding is NOT the dominant error source.")

# ============================================================================
# Step 6: Per-channel weight quantization for conv_1 (pixel-128 mode)
# ============================================================================
print("\n" + "=" * 78)
print("STEP 6: Per-channel weight quantization test (pixel-128 mode)")
print("=" * 78)

# For conv_1: weight matrix is [27, 32] (patch_size x out_ch).
# Per-channel = each output channel column gets its own w_scale.
# This requires per-channel output_scale in the Gemmini matmul, which
# standard Gemmini does NOT support -- but we test it to measure potential.

w1_int_pc = np.zeros_like(wg_p128, dtype=np.int8)  # [27, 32]
w1_scales_pc = np.zeros(num_out_ch, dtype=np.float64)

for ch in range(num_out_ch):
    col = wg_p128[:, ch]
    mx = float(np.max(np.abs(col)))
    if mx < 1e-10:
        w1_scales_pc[ch] = 1e-10
    else:
        w1_scales_pc[ch] = mx / 127.0
        w1_int_pc[:, ch] = np.clip(np.round(col / w1_scales_pc[ch]),
                                    -128, 127).astype(np.int8)

print(f"  Per-channel w_scales range: [{w1_scales_pc.min():.6e}, "
      f"{w1_scales_pc.max():.6e}]")
print(f"  Per-tensor w_scale:          {w1_scale_p128:.6e}")

# We need per-channel bias and output_scale
# combined_scale_pc[ch] = w1_scales_pc[ch] * x_scale_p128
# output_scale_pc[ch] = combined_scale_pc[ch] / y_scale
b1_int_pc = np.zeros(num_out_ch, dtype=np.int32)
os1_pc = np.zeros(num_out_ch, dtype=np.float64)
for ch in range(num_out_ch):
    cs = w1_scales_pc[ch] * x_scale_p128
    b1_int_pc[ch] = quantize_bias_int32(np.array([b_p128[ch]]), cs)[0]
    os1_pc[ch] = cs / y_scale

# Run per-channel conv (manual -- apply per-channel output_scale to accumulator)
x_flat_p128, oh, ow = im2col(x_pixel128, 3, 2, 1)
acc_pc = x_flat_p128.astype(np.int32) @ w1_int_pc.astype(np.int32)  # [N*OH*OW, 32]
acc_pc += b1_int_pc.reshape(1, -1).astype(np.int32)

# Per-channel output scale
y_pc = np.clip(np.round(acc_pc.astype(np.float64) * os1_pc.reshape(1, -1)),
               -128, 127).astype(np.int8)
out1_pc = y_pc.reshape(1, oh, ow, num_out_ch).transpose(0, 3, 1, 2)
out1_pc_relu = np.maximum(out1_pc, np.int8(0))
out1_pc_dequant = out1_pc_relu.astype(np.float64) * y_scale

corr_pc = np.corrcoef(float_conv1_post_relu6.flatten(),
                       out1_pc_dequant.flatten())[0, 1]
mae_pc = np.mean(np.abs(float_conv1_post_relu6 - out1_pc_dequant))

ch_corrs_pc = per_channel_corr(float_conv1_post_relu6, out1_pc_dequant)

print(f"\n  Per-channel quantization results:")
print(f"    Overall correlation:  {corr_pc:.6f}")
print(f"    Overall MAE:          {mae_pc:.6f}")
print(f"    Mean ch corr:         {ch_corrs_pc.mean():.4f}")
print(f"    Median ch corr:       {np.median(ch_corrs_pc):.4f}")

print(f"\n  Improvement over per-tensor pixel-128:")
print(f"    Overall corr: {overall_corr:.6f} -> {corr_pc:.6f} "
      f"(delta = {corr_pc - overall_corr:+.6f})")
print(f"    Mean ch corr: {ch_corrs_p128.mean():.4f} -> {ch_corrs_pc.mean():.4f} "
      f"(delta = {ch_corrs_pc.mean() - ch_corrs_p128.mean():+.4f})")

print(f"\n  Per-channel comparison (per-tensor vs per-channel pixel-128):")
print(f"  {'Ch':>3s}  {'PerTensor':>10s}  {'PerChannel':>10s}  {'Delta':>8s}")
print("  " + "-" * 40)
for ch in range(num_out_ch):
    delta = ch_corrs_pc[ch] - ch_corrs_p128[ch]
    print(f"  {ch:3d}  {ch_corrs_p128[ch]:10.4f}  {ch_corrs_pc[ch]:10.4f}  "
          f"{delta:+8.4f}")

# ============================================================================
# Step 7: Print actual weights for 2 best and 2 worst channels
# ============================================================================
print("\n" + "=" * 78)
print("STEP 7: Actual weights for best/worst channels (by pixel-128 corr)")
print("=" * 78)

sorted_by_corr = np.argsort(ch_corrs_p128)
worst_2 = sorted_by_corr[:2].tolist()
best_2  = sorted_by_corr[-2:].tolist()

def print_channel_weights(ch, label):
    print(f"\n  --- Channel {ch} ({label}, corr={ch_corrs_p128[ch]:.4f}) ---")

    # Float weights (pixel-128 folded)
    w_float = wg_p128[:, ch]  # [27]
    w_int_pt = w1_int_p128[:, ch]
    w_int_pc_ch = w1_int_pc[:, ch]

    # Reshape to [kH=3, kW=3, in_ch=3] for display
    w_float_3d = w_float.reshape(3, 3, 3)
    w_int_pt_3d = w_int_pt.reshape(3, 3, 3)
    w_int_pc_3d = w_int_pc_ch.reshape(3, 3, 3)

    # Also show standard (non-folded) weights
    w_std_ch = wg_std[:, ch]
    w_int_std_ch = w1_int_std[:, ch]
    w_std_3d = w_std_ch.reshape(3, 3, 3)
    w_int_std_3d = w_int_std_ch.reshape(3, 3, 3)

    for in_c in range(3):
        c_name = ["R", "G", "B"][in_c]
        print(f"    Input channel {in_c} ({c_name}):")
        print(f"      Float std:  {w_std_3d[:,:,in_c]}")
        print(f"      INT8  std:  {w_int_std_3d[:,:,in_c]}")
        print(f"      Float p128: {w_float_3d[:,:,in_c]}")
        print(f"      INT8  pt:   {w_int_pt_3d[:,:,in_c]}")
        print(f"      INT8  pc:   {w_int_pc_3d[:,:,in_c]}")

    print(f"    Weight stats (pixel-128 folded):")
    print(f"      max|w|    = {np.max(np.abs(w_float)):.6e}")
    print(f"      per-tensor INT8 range: [{w_int_pt.min()}, {w_int_pt.max()}]")
    print(f"      per-ch     INT8 range: [{w_int_pc_ch.min()}, {w_int_pc_ch.max()}]")
    print(f"      non-zero (per-tensor): {np.count_nonzero(w_int_pt)}/27")
    print(f"      non-zero (per-ch):     {np.count_nonzero(w_int_pc_ch)}/27")

    # Quantization error comparison
    deq_pt = w_int_pt.astype(np.float64) * w_p128_scale
    deq_pc = w_int_pc_ch.astype(np.float64) * w1_scales_pc[ch]
    err_pt = np.sqrt(np.mean((w_float - deq_pt)**2))
    err_pc = np.sqrt(np.mean((w_float - deq_pc)**2))
    print(f"      RMS quant error (per-tensor): {err_pt:.6e}")
    print(f"      RMS quant error (per-ch):     {err_pc:.6e}")

print("  2 WORST channels:")
for ch in worst_2:
    print_channel_weights(ch, "WORST")

print("\n  2 BEST channels:")
for ch in best_2:
    print_channel_weights(ch, "BEST")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 78)
print("SUMMARY")
print("=" * 78)
print(f"""
  conv_1 weight magnitude:
    Standard BN-folded:  w_max = {w_std_max:.4f},  w_scale = {w_std_scale:.6e}
    Pixel-128 folded:    w_max = {w_p128_max:.4f},  w_scale = {w_p128_scale:.6e}
    Magnitude reduction: {w_std_max / w_p128_max:.1f}x

  conv_1 output correlation (overall):
    Standard mode (norm input, BN-folded weights): {corr_std:.6f}
    Pixel-128 mode (per-tensor weights):           {overall_corr:.6f}
    Pixel-128 mode (per-channel weights):          {corr_pc:.6f}

  conv_1 per-channel correlation (mean):
    Standard mode:                {ch_corrs_std.mean():.4f}
    Pixel-128 per-tensor:         {ch_corrs_p128.mean():.4f}
    Pixel-128 per-channel:        {ch_corrs_pc.mean():.4f}

  Dead channels (corr < 0.1, pixel-128 per-tensor): {dead_channels}
  Low channels  (corr < 0.5, pixel-128 per-tensor): {sorted(np.where(ch_corrs_p128 < 0.5)[0].tolist())}
""")

if ch_corrs_pc.mean() - ch_corrs_p128.mean() > 0.05:
    print("  RECOMMENDATION: Per-channel quantization for conv_1 weights would")
    print("  significantly improve pixel-128 mode.  Consider implementing per-channel")
    print("  output_scale in the Gemmini conv_1 path, or pre-scaling the weights.")
elif corr_std - overall_corr > 0.05:
    print("  RECOMMENDATION: The pixel-128 folding itself causes significant error.")
    print("  Consider higher-precision (INT16) weights for conv_1, or using")
    print("  standard normalization with a separate quantization step.")
else:
    print("  NOTE: conv_1 precision is acceptable in both modes.")

print("\nDone.")
