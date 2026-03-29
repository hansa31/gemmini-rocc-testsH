#!/usr/bin/env python3
"""Diagnose the massive correlation drop at conv_3 in MobileNetV2 INT8 PTQ.

conv_3 is the first 1x1 reduce layer: 32 -> 16 channels, NO activation (linear).
This script traces exactly where information is lost by:
  1. Running float reference and capturing activations
  2. Running INT8 through conv_1 and conv_dw_2 (standard pipeline)
  3. Dequantizing the INT8 conv_dw_2 output and computing float conv_3
  4. Analyzing accumulator distributions, output_scale, y_range, per-channel stats
  5. Testing alternative y_range strategies and skipping output requantization
"""
import numpy as np
import torch
from transformers import MobileNetV2ForImageClassification
import cv2

MODEL_NAME = "google/mobilenet_v2_1.0_224"
BN_EPS = 0.001
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float64)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float64)

# ============================================================================
# Load model
# ============================================================================
print("Loading model...")
model = MobileNetV2ForImageClassification.from_pretrained(MODEL_NAME)
model.eval()
sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
nc = sd["classifier.weight"].shape[0]

# ============================================================================
# BN folding helpers
# ============================================================================

def fold_bn(w, g, b, m, v, eps=BN_EPS):
    inv = 1.0 / np.sqrt(v + eps)
    s = g * inv
    shape = [w.shape[0]] + [1] * (w.ndim - 1)
    return w * s.reshape(shape), b - g * m * inv


def get_bn(sd, pfx):
    return tuple(sd[f"{pfx}.{k}"].float().numpy().astype(np.float64) for k in
                 ["convolution.weight", "normalization.weight", "normalization.bias",
                  "normalization.running_mean", "normalization.running_var"])


# ============================================================================
# Quantisation helpers
# ============================================================================

def qw(w):
    mx = float(np.max(np.abs(w)))
    if mx < 1e-10:
        return np.zeros_like(w, dtype=np.int8), 1e-10
    s = mx / 127.0
    return np.clip(np.round(w / s), -128, 127).astype(np.int8), s


def qb(b, cs):
    if cs < 1e-10:
        return np.zeros_like(b, dtype=np.int32)
    return np.clip(np.round(b / cs), -(2**31), 2**31 - 1).astype(np.int32)


# ============================================================================
# Convolution operators
# ============================================================================

def conv1x1_op(x, w_int, b_int, os_val):
    """1x1 conv returning INT8 output."""
    N, C, H, W = x.shape
    xf = x.reshape(N, C, H * W).transpose(0, 2, 1).reshape(N * H * W, C)
    acc = xf.astype(np.int32) @ w_int.astype(np.int32) + b_int.reshape(1, -1).astype(np.int32)
    yf = np.clip(np.round(acc.astype(np.float64) * os_val), -128, 127).astype(np.int8)
    return yf.reshape(N, H, W, w_int.shape[1]).transpose(0, 3, 1, 2)


def conv1x1_op_return_acc(x, w_int, b_int):
    """1x1 conv returning raw INT32 accumulator (before output scaling)."""
    N, C, H, W = x.shape
    xf = x.reshape(N, C, H * W).transpose(0, 2, 1).reshape(N * H * W, C)
    acc = xf.astype(np.int32) @ w_int.astype(np.int32) + b_int.reshape(1, -1).astype(np.int32)
    return acc.reshape(N, H, W, w_int.shape[1]).transpose(0, 3, 1, 2)


def conv_op(x, w_int, b_int, os_val, kernel, stride, pad):
    N, C, H, W = x.shape
    if kernel == 1 and stride == 1:
        xf = x.reshape(N, C, H * W).transpose(0, 2, 1).reshape(N * H * W, C)
        oh = ow = H
    else:
        kH = kW = kernel
        OH = (H + 2 * pad - kH) // stride + 1
        OW = (W + 2 * pad - kW) // stride + 1
        if pad > 0:
            x = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), constant_values=0)
        patches = np.zeros((N, OH, OW, kH * kW * C), dtype=x.dtype)
        for i in range(OH):
            for j in range(OW):
                p = x[:, :, i * stride:i * stride + kH, j * stride:j * stride + kW]
                patches[:, i, j, :] = p.transpose(0, 2, 3, 1).reshape(N, -1)
        xf = patches.reshape(N * OH * OW, kH * kW * C)
        oh = OH; ow = OW
    acc = xf.astype(np.int32) @ w_int.astype(np.int32) + b_int.reshape(1, -1).astype(np.int32)
    yf = np.clip(np.round(acc.astype(np.float64) * os_val), -128, 127).astype(np.int8)
    return yf.reshape(N, oh, ow, w_int.shape[1]).transpose(0, 3, 1, 2)


def dw_conv_op(x, w_int, b_int, os_val, stride, pad):
    N, C, H, W = x.shape; kH = kW = 3
    OH = (H + 2 * pad - kH) // stride + 1
    OW = (W + 2 * pad - kW) // stride + 1
    if pad > 0:
        x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), constant_values=0)
    else:
        x_pad = x
    acc = np.zeros((N, C, OH, OW), dtype=np.int64)
    x32 = x_pad.astype(np.int32); w32 = w_int.astype(np.int32)
    for ki in range(kH):
        for kj in range(kW):
            rows = np.arange(OH) * stride + ki
            cols = np.arange(OW) * stride + kj
            acc += x32[:, :, rows[:, None], cols[None, :]] * w32[:, ki, kj].reshape(1, C, 1, 1)
    acc += b_int.reshape(1, C, 1, 1).astype(np.int64)
    return np.clip(np.round(acc.astype(np.float64) * os_val), -128, 127).astype(np.int8)


# ============================================================================
# Layer definitions (only what we need: conv_1, conv_dw_2, conv_3)
# ============================================================================

LAYERS = [
    ("conv_1",    "mobilenet_v2.conv_stem.first_conv", 3, 2, 1, False, True),
    ("conv_dw_2", "mobilenet_v2.conv_stem.conv_3x3",   3, 1, 1, True,  True),
    ("conv_3",    "mobilenet_v2.conv_stem.reduce_1x1", 1, 1, 0, False, False),
]

# ============================================================================
# Float reference run -- capture activations
# ============================================================================

float_acts = {}

def capture(name):
    def hook(module, inp, out):
        float_acts[name] = out.detach().float().numpy().astype(np.float64)
    return hook

hooks = []
for gn, pfx, k, s, pad, dw, relu in LAYERS:
    parts = pfx.split(".")
    mod2 = model
    for pt in parts:
        mod2 = getattr(mod2, pt)
    h = mod2.normalization.register_forward_hook(capture(gn))
    hooks.append(h)

# Also capture the conv_dw_2 pre-BN output and conv_3 input (post-ReLU6 of conv_dw_2)
def capture_input(name):
    def hook(module, inp, out):
        float_acts[name + "_input"] = inp[0].detach().float().numpy().astype(np.float64)
    return hook

# Hook conv_3's convolution to capture its input
parts = "mobilenet_v2.conv_stem.reduce_1x1".split(".")
mod_conv3 = model
for pt in parts:
    mod_conv3 = getattr(mod_conv3, pt)
h = mod_conv3.convolution.register_forward_hook(capture_input("conv_3"))
hooks.append(h)

img_bgr = cv2.imread("/home/hansa/Downloads/Images200/ILSVRC2012_val_00000001.JPEG")
img = cv2.resize(img_bgr, (224, 224))
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_f = img_rgb.astype(np.float64) / 255.0
for c in range(3):
    img_f[:, :, c] = (img_f[:, :, c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
x_float = torch.tensor(img_f.transpose(2, 0, 1)[np.newaxis], dtype=torch.float32)

with torch.no_grad():
    out = model(x_float)
    float_logits = out.logits[0].numpy()

for h in hooks:
    h.remove()

float_pred = int(np.argmax(float_logits[1:])) if nc == 1001 else int(np.argmax(float_logits))
print(f"Float prediction: {float_pred}")

# ============================================================================
# Prepare INT8 input
# ============================================================================

img_int8 = (img_rgb.astype(np.int16) - 128).clip(-128, 127).astype(np.int8)
x_in = img_int8.transpose(2, 0, 1)[np.newaxis]

# ============================================================================
# Run INT8 pipeline through conv_1 and conv_dw_2
# ============================================================================

print("\n" + "=" * 90)
print("STEP 1: Run INT8 pipeline through conv_1 and conv_dw_2")
print("=" * 90)

# --- conv_1 ---
cw1, g1, b1, m1, v1 = get_bn(sd, "mobilenet_v2.conv_stem.first_conv")
wf1, bf1 = fold_bn(cw1, g1, b1, m1, v1)
# Absorb ImageNet normalisation
no = (128.0 / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
for c in range(3):
    bf1 += wf1[:, c, :, :].sum(axis=(1, 2)) * no[c]
for c in range(3):
    wf1[:, c, :, :] /= (255.0 * IMAGENET_STD[c])

x_scale_input = 128.0 / 127.0  # input scale for img_int8
yr_conv1 = 6.0  # ReLU6
wg1 = wf1.reshape(wf1.shape[0], -1).T
wi1, ws1 = qw(wg1)
bi1 = qb(bf1, ws1 * x_scale_input)
os1 = (ws1 * x_scale_input) / (yr_conv1 / 127.0)

int8_conv1 = conv_op(x_in, wi1, bi1, os1, 3, 2, 1)
int8_conv1 = np.maximum(int8_conv1, np.int8(0))  # ReLU6 clamp

x_scale_conv1 = yr_conv1 / 127.0

print(f"  conv_1: x_scale_input={x_scale_input:.6f}, w_scale={ws1:.6f}, "
      f"y_range={yr_conv1}, output_scale={os1:.6f}")
print(f"  conv_1 INT8 output: shape={int8_conv1.shape}, "
      f"min={int8_conv1.min()}, max={int8_conv1.max()}")

# --- conv_dw_2 ---
cw2, g2, b2, m2, v2 = get_bn(sd, "mobilenet_v2.conv_stem.conv_3x3")
wf2, bf2 = fold_bn(cw2, g2, b2, m2, v2)
yr_dw2 = 6.0  # ReLU6
wg2 = wf2.squeeze(1)  # (C, 3, 3) for depthwise
wg2_flat = wg2.reshape(wg2.shape[0], -1)
wi2_flat, ws2 = qw(wg2_flat)
wi2 = wi2_flat.reshape(wg2.shape)
bi2 = qb(bf2, ws2 * x_scale_conv1)
os2 = (ws2 * x_scale_conv1) / (yr_dw2 / 127.0)

int8_dw2 = dw_conv_op(int8_conv1, wi2, bi2, os2, 1, 1)
int8_dw2 = np.maximum(int8_dw2, np.int8(0))  # ReLU6 clamp

x_scale_dw2 = yr_dw2 / 127.0  # = 6.0 / 127.0 = 0.04724

print(f"\n  conv_dw_2: x_scale={x_scale_conv1:.6f}, w_scale={ws2:.6f}, "
      f"y_range={yr_dw2}, output_scale={os2:.6f}")
print(f"  conv_dw_2 INT8 output: shape={int8_dw2.shape}, "
      f"min={int8_dw2.min()}, max={int8_dw2.max()}")
print(f"  conv_dw_2 x_scale for next layer: {x_scale_dw2:.6f}")

# ============================================================================
# STEP 2: Analyze the dequantized conv_dw_2 output vs float reference
# ============================================================================

print("\n" + "=" * 90)
print("STEP 2: Dequantized conv_dw_2 output vs float reference")
print("=" * 90)

float_dw2_relu6 = np.clip(float_acts["conv_dw_2"], 0, 6.0)
dequant_dw2 = int8_dw2.astype(np.float64) * x_scale_dw2

corr_dw2 = np.corrcoef(float_dw2_relu6.flatten(), dequant_dw2.flatten())[0, 1]
mae_dw2 = np.mean(np.abs(float_dw2_relu6 - dequant_dw2))
print(f"  Correlation (float vs dequant): {corr_dw2:.6f}")
print(f"  MAE (float vs dequant): {mae_dw2:.6f}")
print(f"  Float range: [{float_dw2_relu6.min():.4f}, {float_dw2_relu6.max():.4f}]")
print(f"  Dequant range: [{dequant_dw2.min():.4f}, {dequant_dw2.max():.4f}]")
print(f"  Quantization step (x_scale): {x_scale_dw2:.6f}")

# ============================================================================
# STEP 3: Prepare conv_3 weights and analyze
# ============================================================================

print("\n" + "=" * 90)
print("STEP 3: conv_3 BN-folded weights and bias analysis")
print("=" * 90)

cw3, g3, b3, m3, v3 = get_bn(sd, "mobilenet_v2.conv_stem.reduce_1x1")
wf3, bf3 = fold_bn(cw3, g3, b3, m3, v3)

print(f"  conv_3: 1x1 conv, {wf3.shape[1]} -> {wf3.shape[0]} channels (32 -> 16)")
print(f"  conv_3 has NO activation (linear)")
print()

# BN statistics
print("  BN statistics for conv_3:")
print(f"    gamma:        min={g3.min():.6f}  max={g3.max():.6f}  mean={g3.mean():.6f}")
print(f"    beta:         min={b3.min():.6f}  max={b3.max():.6f}  mean={b3.mean():.6f}")
print(f"    running_mean: min={m3.min():.6f}  max={m3.max():.6f}  mean={m3.mean():.6f}")
print(f"    running_var:  min={v3.min():.6f}  max={v3.max():.6f}  mean={v3.mean():.6f}")
print()

# y_range computation (BN estimate for linear layer)
yr_bn = max(float(np.max(np.abs(b3) + 6.0 * np.abs(g3))), 1.0)
print(f"  BN-estimated y_range: max(|beta| + 6*|gamma|) = {yr_bn:.4f}")
y_scale_bn = yr_bn / 127.0
print(f"  Corresponding y_scale: {y_scale_bn:.6f}")
print(f"  Quantization step at output: {y_scale_bn:.6f}")
print()

# Float conv_3 output range (ground truth)
float_conv3 = float_acts["conv_3"]
float_conv3_abs_max = float(np.max(np.abs(float_conv3)))
print(f"  ACTUAL float conv_3 output range: [{float_conv3.min():.4f}, {float_conv3.max():.4f}]")
print(f"  ACTUAL abs max: {float_conv3_abs_max:.4f}")
print(f"  Ratio: BN_y_range / actual_abs_max = {yr_bn / float_conv3_abs_max:.2f}x")
print(f"  (>1 means BN overestimates range, wasting INT8 dynamic range)")

# Folded weight analysis
wg3 = wf3.reshape(wf3.shape[0], -1).T  # (32, 16) for matmul
wi3, ws3 = qw(wg3)
print(f"\n  Folded weight matrix shape (for matmul): {wg3.shape}")
print(f"  Folded weight abs max: {np.max(np.abs(wg3)):.6f}")
print(f"  w_scale: {ws3:.8f}")

# Per output-channel weight magnitudes
print("\n  Per output-channel folded weight stats (16 channels):")
for ch in range(wf3.shape[0]):
    w_ch = wf3[ch].flatten()
    print(f"    ch{ch:2d}: abs_max={np.max(np.abs(w_ch)):.6f}  "
          f"abs_mean={np.mean(np.abs(w_ch)):.6f}  "
          f"bias={bf3[ch]:.6f}  "
          f"gamma={g3[ch]:.6f}  beta={b3[ch]:.6f}  var={v3[ch]:.6f}")

# ============================================================================
# STEP 4: INT8 conv_3 computation with full accumulator analysis
# ============================================================================

print("\n" + "=" * 90)
print("STEP 4: INT8 conv_3 -- accumulator analysis BEFORE output scaling")
print("=" * 90)

# Quantize weights and bias
combined_scale_3 = ws3 * x_scale_dw2
bi3 = qb(bf3, combined_scale_3)
os3 = combined_scale_3 / y_scale_bn

print(f"  x_scale (from conv_dw_2): {x_scale_dw2:.8f}")
print(f"  w_scale:                   {ws3:.8f}")
print(f"  combined_scale (w*x):      {combined_scale_3:.8f}")
print(f"  y_scale (BN est):          {y_scale_bn:.8f}")
print(f"  output_scale = combined/y: {os3:.8f}")
print()

# Get raw INT32 accumulators
acc_int32 = conv1x1_op_return_acc(int8_dw2, wi3, bi3)
# acc_int32 shape: (1, 16, H, W)

print(f"  Accumulator (INT32) shape: {acc_int32.shape}")
print(f"  Accumulator global range: [{acc_int32.min()}, {acc_int32.max()}]")
print()

# Per-channel accumulator analysis
print("  Per-channel accumulator statistics:")
print(f"  {'ch':>4s}  {'min':>10s}  {'max':>10s}  {'mean':>10s}  {'std':>10s}  "
      f"{'after_scale_min':>16s}  {'after_scale_max':>16s}  {'clipped%':>10s}")
for ch in range(16):
    acc_ch = acc_int32[0, ch].flatten().astype(np.float64)
    scaled = acc_ch * os3
    clipped = np.sum((scaled < -128) | (scaled > 127)) / scaled.size * 100.0
    print(f"  {ch:4d}  {int(acc_ch.min()):10d}  {int(acc_ch.max()):10d}  "
          f"{acc_ch.mean():10.1f}  {acc_ch.std():10.1f}  "
          f"{scaled.min():16.2f}  {scaled.max():16.2f}  {clipped:10.2f}%")

# ============================================================================
# STEP 5: Compute INT8 conv_3 output and compare
# ============================================================================

print("\n" + "=" * 90)
print("STEP 5: INT8 conv_3 output -- correlation and utilization analysis")
print("=" * 90)

# Standard INT8 output
int8_conv3 = conv1x1_op(int8_dw2, wi3, bi3, os3)

print(f"  INT8 conv_3 output shape: {int8_conv3.shape}")
print(f"  INT8 output range: [{int8_conv3.min()}, {int8_conv3.max()}]")
print()

# Overall correlation
dequant_conv3 = int8_conv3.astype(np.float64) * y_scale_bn
corr_overall = np.corrcoef(float_conv3.flatten(), dequant_conv3.flatten())[0, 1]
mae_overall = np.mean(np.abs(float_conv3.flatten() - dequant_conv3.flatten()))
print(f"  Overall correlation (float vs INT8 dequant): {corr_overall:.6f}")
print(f"  Overall MAE:                                 {mae_overall:.6f}")
print()

# INT8 utilization
unique_vals = np.unique(int8_conv3)
print(f"  Distinct INT8 values used: {len(unique_vals)} out of 256 possible")
print(f"  Utilization: {len(unique_vals)/256*100:.1f}%")
print(f"  Value range used: [{unique_vals.min()}, {unique_vals.max()}] "
      f"({unique_vals.max() - unique_vals.min() + 1} levels)")
print()

# Per-channel correlation analysis
print("  Per-channel analysis (16 output channels):")
print(f"  {'ch':>4s}  {'corr':>8s}  {'MAE':>8s}  {'float_range':>14s}  "
      f"{'int8_range':>12s}  {'n_unique':>10s}  {'float_std':>10s}")
per_ch_corrs = []
for ch in range(16):
    f_ch = float_conv3[0, ch].flatten()
    i_ch = int8_conv3[0, ch].flatten()
    d_ch = i_ch.astype(np.float64) * y_scale_bn

    if f_ch.std() < 1e-10 or d_ch.std() < 1e-10:
        corr_ch = 0.0
    else:
        corr_ch = np.corrcoef(f_ch, d_ch)[0, 1]
    mae_ch = np.mean(np.abs(f_ch - d_ch))
    n_uniq = len(np.unique(i_ch))
    per_ch_corrs.append(corr_ch)

    print(f"  {ch:4d}  {corr_ch:8.4f}  {mae_ch:8.4f}  "
          f"[{f_ch.min():6.3f},{f_ch.max():6.3f}]  "
          f"[{i_ch.min():4d},{i_ch.max():4d}]  "
          f"{n_uniq:10d}  {f_ch.std():10.4f}")

worst_ch = int(np.argmin(per_ch_corrs))
best_ch = int(np.argmax(per_ch_corrs))
print(f"\n  Worst channel: {worst_ch} (corr={per_ch_corrs[worst_ch]:.4f})")
print(f"  Best channel:  {best_ch} (corr={per_ch_corrs[best_ch]:.4f})")

# ============================================================================
# STEP 6: Dequantized-input float conv_3 (isolate input quantization error)
# ============================================================================

print("\n" + "=" * 90)
print("STEP 6: Float conv_3 using DEQUANTIZED INT8 input")
print("  (isolates how much error comes from quantizing intermediate activations")
print("   vs the weight/bias quantization and output requantization)")
print("=" * 90)

# Use dequantized INT8 conv_dw_2 output as float input to conv_3
# This uses the ORIGINAL float weights (no weight quantization)
dequant_input = int8_dw2.astype(np.float64) * x_scale_dw2  # dequantized to float

# Float conv_3 with dequantized input
# wf3 shape: (16, 32, 1, 1), need (32, 16) for matmul
N, C_in, H, W = dequant_input.shape
C_out = wf3.shape[0]
inp_flat = dequant_input.reshape(N, C_in, H * W).transpose(0, 2, 1).reshape(N * H * W, C_in)
w_float_matmul = wf3.reshape(C_out, C_in).T  # (32, 16)
float_from_dequant = inp_flat @ w_float_matmul + bf3.reshape(1, -1)
float_from_dequant = float_from_dequant.reshape(N, H, W, C_out).transpose(0, 3, 1, 2)

corr_dequant_float = np.corrcoef(float_conv3.flatten(), float_from_dequant.flatten())[0, 1]
mae_dequant_float = np.mean(np.abs(float_conv3.flatten() - float_from_dequant.flatten()))

print(f"\n  Using dequantized INT8 input + float weights + float bias (no output quant):")
print(f"  Correlation with true float conv_3: {corr_dequant_float:.6f}")
print(f"  MAE:                                {mae_dequant_float:.6f}")
print()

# Now also do: dequantized input + INT8 weights (dequantized) + INT32 bias (dequantized)
w_int8_deq = wi3.astype(np.float64) * ws3  # dequantized weights
b_int32_deq = bi3.astype(np.float64) * combined_scale_3  # dequantized bias
float_from_intweights = inp_flat @ w_int8_deq + b_int32_deq.reshape(1, -1)
float_from_intweights = float_from_intweights.reshape(N, H, W, C_out).transpose(0, 3, 1, 2)

corr_intw = np.corrcoef(float_conv3.flatten(), float_from_intweights.flatten())[0, 1]
mae_intw = np.mean(np.abs(float_conv3.flatten() - float_from_intweights.flatten()))

print(f"  Using dequantized INT8 input + dequantized INT8 weights + dequantized INT32 bias:")
print(f"  Correlation with true float conv_3: {corr_intw:.6f}")
print(f"  MAE:                                {mae_intw:.6f}")
print()

# Fully-float reference using actual float input (from hook)
float_input_conv3 = float_acts.get("conv_3_input")
if float_input_conv3 is not None:
    corr_input = np.corrcoef(float_dw2_relu6.flatten(), dequant_input.flatten())[0, 1]
    print(f"  For reference, float conv_3 input (post-ReLU6 of conv_dw_2):")
    print(f"    Float input range: [{float_input_conv3.min():.4f}, {float_input_conv3.max():.4f}]")
    print(f"    Dequant input range: [{dequant_input.min():.4f}, {dequant_input.max():.4f}]")
    print(f"    Input correlation: {corr_input:.6f}")

# ============================================================================
# STEP 7: What if we use ACTUAL float output range as y_range?
# ============================================================================

print("\n" + "=" * 90)
print("STEP 7: Alternative y_range -- use ACTUAL float conv_3 output range")
print("=" * 90)

yr_actual = float_conv3_abs_max
y_scale_actual = yr_actual / 127.0
os3_actual = combined_scale_3 / y_scale_actual

print(f"  BN-estimated y_range:  {yr_bn:.4f}  (y_scale={y_scale_bn:.6f})")
print(f"  Actual float y_range:  {yr_actual:.4f}  (y_scale={y_scale_actual:.6f})")
print(f"  output_scale (BN est): {os3:.8f}")
print(f"  output_scale (actual): {os3_actual:.8f}")
print()

# Requantize with actual range
acc_scaled_actual = acc_int32.astype(np.float64) * os3_actual
int8_conv3_actual = np.clip(np.round(acc_scaled_actual), -128, 127).astype(np.int8)
dequant_conv3_actual = int8_conv3_actual.astype(np.float64) * y_scale_actual

corr_actual = np.corrcoef(float_conv3.flatten(), dequant_conv3_actual.flatten())[0, 1]
mae_actual = np.mean(np.abs(float_conv3.flatten() - dequant_conv3_actual.flatten()))
unique_actual = np.unique(int8_conv3_actual)

print(f"  With ACTUAL y_range:")
print(f"  Correlation: {corr_actual:.6f}  (was {corr_overall:.6f} with BN estimate)")
print(f"  MAE:         {mae_actual:.6f}  (was {mae_overall:.6f})")
print(f"  INT8 utilization: {len(unique_actual)}/256 = {len(unique_actual)/256*100:.1f}%")
print(f"  INT8 range used: [{unique_actual.min()}, {unique_actual.max()}]")
print()

# Per-channel with actual range
print("  Per-channel with ACTUAL y_range:")
print(f"  {'ch':>4s}  {'corr_bn':>9s}  {'corr_actual':>12s}  {'improvement':>12s}")
for ch in range(16):
    f_ch = float_conv3[0, ch].flatten()
    d_bn = int8_conv3[0, ch].flatten().astype(np.float64) * y_scale_bn
    d_ac = int8_conv3_actual[0, ch].flatten().astype(np.float64) * y_scale_actual

    if f_ch.std() < 1e-10:
        c_bn = c_ac = 0.0
    else:
        c_bn = np.corrcoef(f_ch, d_bn)[0, 1] if d_bn.std() > 1e-10 else 0.0
        c_ac = np.corrcoef(f_ch, d_ac)[0, 1] if d_ac.std() > 1e-10 else 0.0
    delta = c_ac - c_bn
    print(f"  {ch:4d}  {c_bn:9.4f}  {c_ac:12.4f}  {delta:+12.4f}")

# Also try a range of y_range multipliers
print("\n  Sweep: y_range = factor * actual_abs_max")
print(f"  {'factor':>8s}  {'y_range':>10s}  {'corr':>8s}  {'MAE':>8s}  {'utilization':>12s}")
for factor in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, yr_bn / yr_actual]:
    yr_test = yr_actual * factor
    ys_test = yr_test / 127.0
    os_test = combined_scale_3 / ys_test
    scaled_test = acc_int32.astype(np.float64) * os_test
    i8_test = np.clip(np.round(scaled_test), -128, 127).astype(np.int8)
    dq_test = i8_test.astype(np.float64) * ys_test
    c_test = np.corrcoef(float_conv3.flatten(), dq_test.flatten())[0, 1]
    m_test = np.mean(np.abs(float_conv3.flatten() - dq_test.flatten()))
    u_test = len(np.unique(i8_test))
    label = " <-- BN estimate" if abs(factor - yr_bn / yr_actual) < 0.01 else ""
    label = " <-- actual range" if abs(factor - 1.0) < 0.01 else label
    print(f"  {factor:8.3f}  {yr_test:10.4f}  {c_test:8.4f}  {m_test:8.4f}  "
          f"{u_test:4d}/256{label}")

# ============================================================================
# STEP 8: What if we DON'T requantize conv_3 output to INT8?
# ============================================================================

print("\n" + "=" * 90)
print("STEP 8: Skip output requantization -- keep float accumulator result")
print("=" * 90)

# The accumulator result (INT32) * combined_scale gives the true fixed-point output
# Without requantization, we keep full precision
float_from_acc = acc_int32.astype(np.float64) * combined_scale_3

corr_no_requant = np.corrcoef(float_conv3.flatten(), float_from_acc.flatten())[0, 1]
mae_no_requant = np.mean(np.abs(float_conv3.flatten() - float_from_acc.flatten()))

print(f"  Keep accumulator as float (no INT8 requant):")
print(f"  Correlation: {corr_no_requant:.6f}")
print(f"  MAE:         {mae_no_requant:.6f}")
print()
print(f"  Compare all approaches for conv_3:")
print(f"  {'Method':<55s}  {'Correlation':>12s}  {'MAE':>10s}")
print(f"  {'-'*55}  {'-'*12}  {'-'*10}")
print(f"  {'Full float (reference = 1.0)':<55s}  {'1.000000':>12s}  {'0.000000':>10s}")
print(f"  {'Dequant input + float weights (no output quant)':<55s}  "
      f"{corr_dequant_float:>12.6f}  {mae_dequant_float:>10.6f}")
print(f"  {'Dequant input + INT8 weights (no output quant)':<55s}  "
      f"{corr_intw:>12.6f}  {mae_intw:>10.6f}")
print(f"  {'INT8 matmul, keep accum as float (no requant)':<55s}  "
      f"{corr_no_requant:>12.6f}  {mae_no_requant:>10.6f}")
print(f"  {'INT8 matmul + requant with ACTUAL y_range':<55s}  "
      f"{corr_actual:>12.6f}  {mae_actual:>10.6f}")
print(f"  {'INT8 matmul + requant with BN-estimated y_range':<55s}  "
      f"{corr_overall:>12.6f}  {mae_overall:>10.6f}")

# ============================================================================
# STEP 9: Detailed breakdown -- where is the information lost?
# ============================================================================

print("\n" + "=" * 90)
print("STEP 9: Information loss breakdown")
print("=" * 90)

# Error from input quantization (conv_dw_2 output)
err_input = corr_dequant_float  # correlation when only input is quantized
# Error from weight quantization
err_weight = corr_intw  # correlation when input + weights are quantized
# Error from output requantization with BN range
err_output_bn = corr_overall  # full INT8 pipeline with BN y_range
# Error from output requantization with actual range
err_output_actual = corr_actual

print(f"\n  Correlation at each stage:")
print(f"  1. Float reference:                          1.000000")
print(f"  2. After input quantization only:            {err_input:.6f}  "
      f"(drop: {1.0 - err_input:.6f})")
print(f"  3. After input + weight quantization:        {err_weight:.6f}  "
      f"(additional drop: {err_input - err_weight:.6f})")
print(f"  4. After INT8 accumulator (no requant):      {corr_no_requant:.6f}  "
      f"(additional drop: {err_weight - corr_no_requant:.6f})")
print(f"  5. After requant with ACTUAL y_range:        {err_output_actual:.6f}  "
      f"(additional drop: {corr_no_requant - err_output_actual:.6f})")
print(f"  6. After requant with BN y_range:            {err_output_bn:.6f}  "
      f"(additional drop: {corr_no_requant - err_output_bn:.6f})")
print()

# Identify the biggest source of error
drops = {
    "Input quantization (conv_dw_2 -> INT8)":     1.0 - err_input,
    "Weight quantization (float -> INT8)":         err_input - err_weight,
    "INT8 matmul rounding":                        err_weight - corr_no_requant,
    "Output requant with BN y_range":              corr_no_requant - err_output_bn,
}

print("  Error attribution (correlation drops):")
sorted_drops = sorted(drops.items(), key=lambda x: -x[1])
for name, drop in sorted_drops:
    bar = "#" * max(1, int(drop * 200))
    print(f"    {name:<50s}  {drop:+.6f}  {bar}")

print()
biggest = sorted_drops[0]
print(f"  BIGGEST source of error: {biggest[0]} (corr drop = {biggest[1]:.6f})")

# ============================================================================
# STEP 10: Additional insight -- output_scale magnitude
# ============================================================================

print("\n" + "=" * 90)
print("STEP 10: output_scale analysis for Gemmini hardware")
print("=" * 90)

print(f"\n  output_scale = (w_scale * x_scale) / y_scale")
print(f"  For conv_3:")
print(f"    w_scale = {ws3:.8f}")
print(f"    x_scale = {x_scale_dw2:.8f}  (6.0/127.0)")
print(f"    y_scale (BN) = {y_scale_bn:.8f}")
print(f"    output_scale (BN) = {os3:.8f}")
print()
print(f"  Interpretation:")
if os3 < 1.0:
    print(f"    output_scale < 1.0 means the accumulator values are being SHRUNK")
    print(f"    Each accumulator step maps to {os3:.6f} output steps")
    print(f"    Many accumulator values will map to the SAME INT8 output -> info loss")
else:
    print(f"    output_scale >= 1.0 means the accumulator values are being EXPANDED")
    print(f"    This is fine for precision (no squashing), but may clip at +/-128")

# How many accumulator values per INT8 output level?
acc_range = float(acc_int32.max() - acc_int32.min())
if os3 > 0:
    acc_per_level = 1.0 / os3
    print(f"\n  Accumulator values per INT8 output level: {acc_per_level:.2f}")
    print(f"  (1/output_scale = 1/{os3:.6f} = {acc_per_level:.2f})")
    if acc_per_level > 2:
        print(f"  WARNING: ~{acc_per_level:.0f} distinct accumulator values map to "
              f"each INT8 level")
        print(f"  This means ~{(1 - 1.0/acc_per_level)*100:.1f}% of accumulator precision is discarded!")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "=" * 90)
print("SUMMARY")
print("=" * 90)
print(f"""
conv_3 is a 32->16 channel 1x1 convolution with NO activation (linear).

Key findings:
  - Input x_scale (from conv_dw_2 ReLU6): {x_scale_dw2:.6f}
  - Weight scale:                          {ws3:.6f}
  - BN-estimated y_range:                  {yr_bn:.4f}
  - Actual float output range:             {float_conv3_abs_max:.4f}
  - BN overestimation ratio:               {yr_bn / float_conv3_abs_max:.2f}x
  - output_scale (BN):                     {os3:.8f}
  - INT8 output utilization:               {len(unique_vals)}/256

Correlation comparison:
  - Float reference:                       1.000000
  - INT8 with BN y_range:                  {corr_overall:.6f}
  - INT8 with actual y_range:              {corr_actual:.6f}
  - No output requant (keep float):        {corr_no_requant:.6f}
  - Dequant input + float weights:         {corr_dequant_float:.6f}

The biggest error source is: {biggest[0]}
""")
