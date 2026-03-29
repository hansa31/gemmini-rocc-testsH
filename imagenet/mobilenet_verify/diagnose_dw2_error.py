#!/usr/bin/env python3
"""Diagnose WHY conv_dw_2 has only 0.77 correlation in MobileNetV2 INT8 PTQ.

conv_dw_2 is the first depthwise 3x3 conv (32 channels, stride 1, pad 1, ReLU6).
This script decomposes the error into:
  - Input quantization error (from conv_1 INT8 output)
  - Weight quantization error (per-tensor scale hurts small-weight channels)
  - Output requantization error
  - Dead BN channel analysis (low-variance channels dominate per-tensor w_scale)

It also tests whether re-injecting the correct conv_dw_2 output fixes downstream
conv_3 correlation, confirming error propagation.
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
    """Per-tensor symmetric INT8 weight quantisation."""
    mx = float(np.max(np.abs(w)))
    if mx < 1e-10:
        return np.zeros_like(w, dtype=np.int8), 1e-10
    s = mx / 127.0
    return np.clip(np.round(w / s), -128, 127).astype(np.int8), s


def qb(b, cs):
    if cs < 1e-10:
        return np.zeros_like(b, dtype=np.int32)
    return np.clip(np.round(b / cs), -(2**31), 2**31 - 1).astype(np.int32)


def qw_perchannel_dw(w_3d):
    """Per-channel symmetric INT8 quantisation for DW weights [C, kH, kW].
    Returns: w_int8[C, kH, kW], w_scales[C]
    """
    C = w_3d.shape[0]
    w_int = np.zeros_like(w_3d, dtype=np.int8)
    w_scales = np.zeros(C, dtype=np.float64)
    for c in range(C):
        mx = float(np.max(np.abs(w_3d[c])))
        if mx < 1e-10:
            w_scales[c] = 1e-10
        else:
            w_scales[c] = mx / 127.0
            w_int[c] = np.clip(np.round(w_3d[c] / w_scales[c]),
                                -128, 127).astype(np.int8)
    return w_int, w_scales


# ============================================================================
# Convolution operators
# ============================================================================

def conv_op(x, w_int, b_int, os_val, kernel, stride, pad):
    N, C, H, W = x.shape
    kH = kW = kernel
    OH = (H + 2 * pad - kH) // stride + 1
    OW = (W + 2 * pad - kW) // stride + 1
    if pad > 0:
        x = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)),
                    constant_values=0)
    patches = np.zeros((N, OH, OW, kH * kW * C), dtype=x.dtype)
    for i in range(OH):
        for j in range(OW):
            p = x[:, :, i * stride:i * stride + kH,
                  j * stride:j * stride + kW]
            patches[:, i, j, :] = p.transpose(0, 2, 3, 1).reshape(N, -1)
    xf = patches.reshape(N * OH * OW, kH * kW * C)
    acc = (xf.astype(np.int32) @ w_int.astype(np.int32)
           + b_int.reshape(1, -1).astype(np.int32))
    yf = np.clip(np.round(acc.astype(np.float64) * os_val),
                  -128, 127).astype(np.int8)
    return yf.reshape(N, OH, OW, w_int.shape[1]).transpose(0, 3, 1, 2)


def conv1x1_op(x, w_int, b_int, os_val):
    """1x1 conv returning INT8 output."""
    N, C, H, W = x.shape
    xf = x.reshape(N, C, H * W).transpose(0, 2, 1).reshape(N * H * W, C)
    acc = (xf.astype(np.int32) @ w_int.astype(np.int32)
           + b_int.reshape(1, -1).astype(np.int32))
    yf = np.clip(np.round(acc.astype(np.float64) * os_val),
                  -128, 127).astype(np.int8)
    return yf.reshape(N, H, W, w_int.shape[1]).transpose(0, 3, 1, 2)


def dw_conv_op(x, w_int, b_int, os_val, stride, pad):
    """DW 3x3 conv with scalar output_scale, returns INT8."""
    N, C, H, W = x.shape
    kH = kW = 3
    OH = (H + 2 * pad - kH) // stride + 1
    OW = (W + 2 * pad - kW) // stride + 1
    if pad > 0:
        x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)),
                        constant_values=0)
    else:
        x_pad = x
    acc = np.zeros((N, C, OH, OW), dtype=np.int64)
    x32 = x_pad.astype(np.int32)
    w32 = w_int.astype(np.int32)
    for ki in range(kH):
        for kj in range(kW):
            rows = np.arange(OH) * stride + ki
            cols = np.arange(OW) * stride + kj
            acc += (x32[:, :, rows[:, None], cols[None, :]]
                    * w32[:, ki, kj].reshape(1, C, 1, 1))
    acc += b_int.reshape(1, C, 1, 1).astype(np.int64)
    return np.clip(np.round(acc.astype(np.float64) * os_val),
                    -128, 127).astype(np.int8)


def dw_conv_op_return_acc(x, w_int, b_int, stride, pad):
    """DW 3x3 conv returning raw INT64 accumulator (before output scaling)."""
    N, C, H, W = x.shape
    kH = kW = 3
    OH = (H + 2 * pad - kH) // stride + 1
    OW = (W + 2 * pad - kW) // stride + 1
    if pad > 0:
        x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)),
                        constant_values=0)
    else:
        x_pad = x
    acc = np.zeros((N, C, OH, OW), dtype=np.int64)
    x32 = x_pad.astype(np.int32)
    w32 = w_int.astype(np.int32)
    for ki in range(kH):
        for kj in range(kW):
            rows = np.arange(OH) * stride + ki
            cols = np.arange(OW) * stride + kj
            acc += (x32[:, :, rows[:, None], cols[None, :]]
                    * w32[:, ki, kj].reshape(1, C, 1, 1))
    acc += b_int.reshape(1, C, 1, 1).astype(np.int64)
    return acc


def dw_conv_float(x_float, w_float, b_float, stride, pad):
    """DW 3x3 conv in full float64 precision."""
    N, C, H, W = x_float.shape
    kH = kW = 3
    OH = (H + 2 * pad - kH) // stride + 1
    OW = (W + 2 * pad - kW) // stride + 1
    if pad > 0:
        x_pad = np.pad(x_float, ((0, 0), (0, 0), (pad, pad), (pad, pad)),
                        constant_values=0.0)
    else:
        x_pad = x_float.copy()
    out = np.zeros((N, C, OH, OW), dtype=np.float64)
    for ki in range(kH):
        for kj in range(kW):
            rows = np.arange(OH) * stride + ki
            cols = np.arange(OW) * stride + kj
            out += (x_pad[:, :, rows[:, None], cols[None, :]]
                    * w_float[:, ki, kj].reshape(1, C, 1, 1))
    out += b_float.reshape(1, C, 1, 1)
    return out


# ============================================================================
# Capture float activations
# ============================================================================

float_acts = {}


def capture(name):
    def hook(module, inp, out):
        float_acts[name] = out.detach().float().numpy().astype(np.float64)
    return hook


def capture_pre_bn(name):
    """Capture the input to BN (= raw conv output before BN normalization)."""
    def hook(module, inp, out):
        float_acts[name] = inp[0].detach().float().numpy().astype(np.float64)
    return hook


hooks = []

# Hook BN outputs (post-BN, pre-ReLU6) for conv_1, conv_dw_2, conv_3
layer_map = {
    "conv_1":    "mobilenet_v2.conv_stem.first_conv",
    "conv_dw_2": "mobilenet_v2.conv_stem.conv_3x3",
    "conv_3":    "mobilenet_v2.conv_stem.reduce_1x1",
}
for gn, pfx in layer_map.items():
    parts = pfx.split(".")
    mod2 = model
    for pt in parts:
        mod2 = getattr(mod2, pt)
    h = mod2.normalization.register_forward_hook(capture(gn))
    hooks.append(h)

# Also capture the PRE-BN conv_dw_2 output (raw conv output)
parts_dw2 = "mobilenet_v2.conv_stem.conv_3x3".split(".")
mod_dw2 = model
for pt in parts_dw2:
    mod_dw2 = getattr(mod_dw2, pt)
h = mod_dw2.normalization.register_forward_hook(
    capture_pre_bn("conv_dw_2_pre_bn"))
hooks.append(h)

# Load image
img_bgr = cv2.imread(
    "/home/hansa/Downloads/Images200/ILSVRC2012_val_00000001.JPEG")
img = cv2.resize(img_bgr, (224, 224))
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_f = img_rgb.astype(np.float64) / 255.0
for c in range(3):
    img_f[:, :, c] = (img_f[:, :, c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
x_float = torch.tensor(img_f.transpose(2, 0, 1)[np.newaxis],
                        dtype=torch.float32)

with torch.no_grad():
    out = model(x_float)
    float_logits = out.logits[0].numpy()

for h in hooks:
    h.remove()

float_pred = (int(np.argmax(float_logits[1:])) if nc == 1001
              else int(np.argmax(float_logits)))
print(f"Float prediction: {float_pred}")

# Compute float post-ReLU6 references
float_conv1_r6 = np.clip(float_acts["conv_1"], 0, 6.0)
float_dw2_r6 = np.clip(float_acts["conv_dw_2"], 0, 6.0)
float_conv3 = float_acts["conv_3"]  # no activation (linear)

# ============================================================================
# Prepare INT8 input
# ============================================================================

img_int8 = (img_rgb.astype(np.int16) - 128).clip(-128, 127).astype(np.int8)
x_in = img_int8.transpose(2, 0, 1)[np.newaxis]

# ============================================================================
# STEP 1: INT8 conv_1
# ============================================================================

print("\n" + "=" * 90)
print("STEP 1: Run INT8 conv_1")
print("=" * 90)

cw1, g1, b1, m1, v1 = get_bn(sd, "mobilenet_v2.conv_stem.first_conv")
wf1, bf1 = fold_bn(cw1, g1, b1, m1, v1)

# Absorb ImageNet normalisation into conv_1 weights
no = (128.0 / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
for c in range(3):
    bf1 += wf1[:, c, :, :].sum(axis=(1, 2)) * no[c]
for c in range(3):
    wf1[:, c, :, :] /= (255.0 * IMAGENET_STD[c])

x_scale_input = 128.0 / 127.0
yr_conv1 = 6.0
wg1 = wf1.reshape(wf1.shape[0], -1).T  # [27, 32]
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

# ============================================================================
# STEP 2: Prepare conv_dw_2 weights
# ============================================================================

print("\n" + "=" * 90)
print("STEP 2: Prepare conv_dw_2 weights and quantize")
print("=" * 90)

cw2, g2, b2, m2, v2 = get_bn(sd, "mobilenet_v2.conv_stem.conv_3x3")
wf2, bf2 = fold_bn(cw2, g2, b2, m2, v2)
yr_dw2 = 6.0

# For DW conv: weight shape is [C, 1, kH, kW] -> squeeze to [C, kH, kW]
wg2 = wf2.squeeze(1)  # (32, 3, 3)
wg2_flat = wg2.reshape(wg2.shape[0], -1)  # (32, 9)

# Per-tensor quantization
wi2_flat, ws2 = qw(wg2_flat)
wi2 = wi2_flat.reshape(wg2.shape)
bi2 = qb(bf2, ws2 * x_scale_conv1)
os2 = (ws2 * x_scale_conv1) / (yr_dw2 / 127.0)

print(f"  conv_dw_2: 32ch DW 3x3, stride=1, pad=1, ReLU6")
print(f"  x_scale={x_scale_conv1:.6f}, w_scale={ws2:.6f}, "
      f"y_range={yr_dw2}, output_scale={os2:.6f}")
print(f"  Folded weight abs max (global): {np.max(np.abs(wg2)):.6f}")

# ============================================================================
# STEP 3a: Analyze input to conv_dw_2
# ============================================================================

print("\n" + "=" * 90)
print("STEP 3a: INPUT to conv_dw_2 -- float conv_1 output vs INT8 conv_1 output")
print("=" * 90)

dequant_conv1 = int8_conv1.astype(np.float64) * x_scale_conv1
corr_input = np.corrcoef(float_conv1_r6.flatten(),
                          dequant_conv1.flatten())[0, 1]
mae_input = np.mean(np.abs(float_conv1_r6 - dequant_conv1))

print(f"  Float conv_1 (post-ReLU6) range: [{float_conv1_r6.min():.4f}, "
      f"{float_conv1_r6.max():.4f}]")
print(f"  INT8 dequant conv_1 range:        [{dequant_conv1.min():.4f}, "
      f"{dequant_conv1.max():.4f}]")
print(f"  Correlation (float vs INT8 conv_1 output): {corr_input:.6f}")
print(f"  MAE: {mae_input:.6f}")

# Quantize the float conv_1 output to INT8 for comparison
float_c1_quantized = np.clip(
    np.round(float_conv1_r6 / x_scale_conv1), -128, 127).astype(np.int8)
diff_c1 = np.abs(int8_conv1.astype(np.int16)
                  - float_c1_quantized.astype(np.int16))
print(f"  INT8 diff (pipeline vs float-quantized): max={diff_c1.max()} "
      f"mean={diff_c1.mean():.2f} "
      f"pct_exact={100 * np.mean(diff_c1 == 0):.1f}%")

# Per-channel input correlation
print("\n  Per-channel input correlation (32 channels):")
print(f"  {'ch':>4s}  {'corr':>8s}  {'MAE':>8s}  "
      f"{'float_mean':>12s}  {'int8_mean':>12s}")
input_corrs = []
for ch in range(32):
    f_ch = float_conv1_r6[0, ch].flatten()
    d_ch = dequant_conv1[0, ch].flatten()
    if f_ch.std() < 1e-10 or d_ch.std() < 1e-10:
        c = 0.0
    else:
        c = np.corrcoef(f_ch, d_ch)[0, 1]
    m = np.mean(np.abs(f_ch - d_ch))
    input_corrs.append(c)
    flag = "  <-- LOW" if c < 0.95 else ""
    print(f"  {ch:4d}  {c:8.4f}  {m:8.4f}  "
          f"{f_ch.mean():12.4f}  {d_ch.mean():12.4f}{flag}")

# ============================================================================
# STEP 3b: Per-channel DW weight analysis
# ============================================================================

print("\n" + "=" * 90)
print("STEP 3b: Per-channel WEIGHT analysis (32 DW channels)")
print("=" * 90)

per_ch_max = np.max(np.abs(wg2), axis=(1, 2))
global_max = np.max(per_ch_max)

print(f"  Global weight abs max: {global_max:.6f}")
print(f"  Per-tensor w_scale: {ws2:.6f} (= {global_max:.6f} / 127)")
print()

# Per-channel scales
pc_wi2, pc_ws2 = qw_perchannel_dw(wg2)

print(f"  {'ch':>4s}  {'abs_max':>10s}  {'ratio':>8s}  "
      f"{'tensor_scale':>14s}  {'chan_scale':>14s}  "
      f"{'scale_ratio':>12s}  {'int8_absmax':>12s}  {'int8_nzero':>10s}")
for ch in range(32):
    ch_max = per_ch_max[ch]
    ratio = ch_max / global_max if global_max > 0 else 0
    # INT8 weight using per-tensor scale
    w_ch_int8 = wi2_flat[ch]  # (9,) int8
    i8_absmax = int(np.max(np.abs(w_ch_int8)))
    i8_nzero = int(np.sum(w_ch_int8 == 0))
    flag = " <-- CRUSHED" if ratio < 0.1 else ""
    sr = ws2 / pc_ws2[ch] if pc_ws2[ch] > 1e-10 else 0
    print(f"  {ch:4d}  {ch_max:10.6f}  {ratio:8.4f}  "
          f"{ws2:14.6f}  {pc_ws2[ch]:14.6f}  "
          f"{sr:12.2f}x  {i8_absmax:12d}  {i8_nzero:10d}{flag}")

crushed = np.sum(per_ch_max < 0.1 * global_max)
print(f"\n  Channels with abs_max < 10% of global max: {crushed}/32")
print(f"  These channels lose most of their weight precision "
      f"with per-tensor quantization.")

# Show the actual INT8 weights for smallest and largest channels
smallest_ch = int(np.argmin(per_ch_max))
largest_ch = int(np.argmax(per_ch_max))
print(f"\n  Smallest weight channel (ch {smallest_ch}, "
      f"abs_max={per_ch_max[smallest_ch]:.6f}):")
print(f"    Float 3x3 kernel:\n{wg2[smallest_ch]}")
print(f"    INT8 3x3 kernel (per-tensor):\n{wi2[smallest_ch]}")
print(f"    INT8 3x3 kernel (per-channel):\n{pc_wi2[smallest_ch]}")

print(f"\n  Largest weight channel (ch {largest_ch}, "
      f"abs_max={per_ch_max[largest_ch]:.6f}):")
print(f"    Float 3x3 kernel:\n{wg2[largest_ch]}")
print(f"    INT8 3x3 kernel (per-tensor):\n{wi2[largest_ch]}")
print(f"    INT8 3x3 kernel (per-channel):\n{pc_wi2[largest_ch]}")

# ============================================================================
# STEP 3c: Run INT8 DW conv and check per-channel output correlation
# ============================================================================

print("\n" + "=" * 90)
print("STEP 3c: conv_dw_2 OUTPUT -- per-channel correlation analysis")
print("=" * 90)

# Run DW conv with INT8 pipeline
int8_dw2 = dw_conv_op(int8_conv1, wi2, bi2, os2, 1, 1)
int8_dw2_relu = np.maximum(int8_dw2, np.int8(0))

x_scale_dw2 = yr_dw2 / 127.0
dequant_dw2 = int8_dw2_relu.astype(np.float64) * x_scale_dw2

corr_dw2_overall = np.corrcoef(float_dw2_r6.flatten(),
                                dequant_dw2.flatten())[0, 1]
mae_dw2_overall = np.mean(np.abs(float_dw2_r6 - dequant_dw2))

print(f"  Overall conv_dw_2 correlation (INT8 pipeline): "
      f"{corr_dw2_overall:.6f}")
print(f"  Overall MAE: {mae_dw2_overall:.6f}")
print()

print(f"  Per-channel output correlation:")
print(f"  {'ch':>4s}  {'corr':>8s}  {'MAE':>8s}  {'float_std':>10s}  "
      f"{'int8_std':>10s}  {'w_absmax':>10s}  {'w_ratio':>10s}")
out_corrs = []
for ch in range(32):
    f_ch = float_dw2_r6[0, ch].flatten()
    d_ch = dequant_dw2[0, ch].flatten()
    if f_ch.std() < 1e-10 or d_ch.std() < 1e-10:
        c = 0.0
    else:
        c = np.corrcoef(f_ch, d_ch)[0, 1]
    m = np.mean(np.abs(f_ch - d_ch))
    out_corrs.append(c)
    flag = " <-- BAD" if c < 0.8 else ""
    print(f"  {ch:4d}  {c:8.4f}  {m:8.4f}  {f_ch.std():10.4f}  "
          f"{d_ch.std():10.4f}  {per_ch_max[ch]:10.6f}  "
          f"{per_ch_max[ch] / global_max:10.4f}{flag}")

worst_out = int(np.argmin(out_corrs))
best_out = int(np.argmax(out_corrs))
print(f"\n  Worst channel: {worst_out} (corr={out_corrs[worst_out]:.4f})")
print(f"  Best channel:  {best_out} (corr={out_corrs[best_out]:.4f})")

# ============================================================================
# STEP 4: Separate error sources -- oracle experiments
# ============================================================================

print("\n" + "=" * 90)
print("STEP 4: Oracle experiments -- separate input / weight / output error")
print("=" * 90)

# ---- Experiment A: "full float DW" ----
# Take FLOAT conv_1 output (post-ReLU6), run through FLOAT DW weights
# This should be ~1.0 correlation
float_dw2_recompute = dw_conv_float(float_conv1_r6, wg2, bf2, 1, 1)
float_dw2_recompute_r6 = np.clip(float_dw2_recompute, 0, 6.0)
corr_full_float = np.corrcoef(float_dw2_r6.flatten(),
                               float_dw2_recompute_r6.flatten())[0, 1]
print(f"\n  A. FULL FLOAT DW (float input + float weights):")
print(f"     Correlation: {corr_full_float:.6f}  (should be ~1.0)")

# ---- Experiment B: "oracle input" ----
# Take FLOAT conv_1 output, quantize to INT8, run through INT8 DW conv
# This isolates the weight quantization error (input is as good as possible)
float_c1_as_int8 = np.clip(
    np.round(float_conv1_r6 / x_scale_conv1), -128, 127).astype(np.int8)
float_c1_as_int8 = np.maximum(float_c1_as_int8, np.int8(0))

out_oracle_input = dw_conv_op(float_c1_as_int8, wi2, bi2, os2, 1, 1)
out_oracle_input_relu = np.maximum(out_oracle_input, np.int8(0))
dequant_oracle_input = (out_oracle_input_relu.astype(np.float64)
                        * x_scale_dw2)
corr_oracle_input = np.corrcoef(float_dw2_r6.flatten(),
                                 dequant_oracle_input.flatten())[0, 1]
print(f"\n  B. ORACLE INPUT (float conv_1 quantized + INT8 DW weights):")
print(f"     Correlation: {corr_oracle_input:.6f}")
print(f"     (This isolates weight+output quant error, removing input error)")

# ---- Experiment C: "oracle weights" ----
# Take INT8 conv_1 output, dequantize, run through FLOAT DW weights
# This isolates the input quantization error
dequant_int8_c1 = int8_conv1.astype(np.float64) * x_scale_conv1
float_dw2_from_int8input = dw_conv_float(dequant_int8_c1, wg2, bf2, 1, 1)
float_dw2_from_int8input_r6 = np.clip(float_dw2_from_int8input, 0, 6.0)
corr_oracle_weights = np.corrcoef(
    float_dw2_r6.flatten(),
    float_dw2_from_int8input_r6.flatten())[0, 1]
print(f"\n  C. ORACLE WEIGHTS (INT8 conv_1 dequant + float DW weights, "
      f"float output):")
print(f"     Correlation: {corr_oracle_weights:.6f}")
print(f"     (This shows error from conv_1 INT8 output only)")

# ---- Experiment D: oracle weights + output requantization ----
# Same as C but requantize the output to INT8
float_dw2_from_int8input_r6_int8 = np.clip(
    np.round(float_dw2_from_int8input_r6 / x_scale_dw2),
    -128, 127).astype(np.int8)
float_dw2_from_int8input_r6_int8 = np.maximum(
    float_dw2_from_int8input_r6_int8, np.int8(0))
dequant_oracle_w_requant = (
    float_dw2_from_int8input_r6_int8.astype(np.float64) * x_scale_dw2)
corr_oracle_w_requant = np.corrcoef(
    float_dw2_r6.flatten(),
    dequant_oracle_w_requant.flatten())[0, 1]
print(f"\n  D. ORACLE WEIGHTS + OUTPUT REQUANT (INT8 input dequant + "
      f"float DW + requant to INT8):")
print(f"     Correlation: {corr_oracle_w_requant:.6f}")
print(f"     (Shows input error + output requantization error, "
      f"no weight error)")

# ---- Experiment E: INT8 input + INT8 weights, keep accumulator as float ----
acc_raw = dw_conv_op_return_acc(int8_conv1, wi2, bi2, 1, 1)
combined_scale_dw2 = ws2 * x_scale_conv1
float_from_acc = acc_raw.astype(np.float64) * combined_scale_dw2
float_from_acc_r6 = np.clip(float_from_acc, 0, 6.0)
corr_no_requant = np.corrcoef(float_dw2_r6.flatten(),
                                float_from_acc_r6.flatten())[0, 1]
print(f"\n  E. INT8 INPUT + INT8 WEIGHTS, NO OUTPUT REQUANT "
      f"(keep float accum):")
print(f"     Correlation: {corr_no_requant:.6f}")
print(f"     (Shows error from input + weight quantization, "
      f"no output requant)")

# ---- Summary table ----
print(f"\n  {'Experiment':<60s}  {'Corr':>8s}  {'Drop from 1.0':>14s}")
print(f"  {'-' * 60}  {'-' * 8}  {'-' * 14}")
print(f"  {'A. Full float DW (reference)':<60s}  "
      f"{corr_full_float:8.6f}  "
      f"{1.0 - corr_full_float:14.6f}")
print(f"  {'C. Oracle weights (float DW, INT8 input, no requant)':<60s}  "
      f"{corr_oracle_weights:8.6f}  "
      f"{1.0 - corr_oracle_weights:14.6f}")
print(f"  {'D. Oracle weights + output requant':<60s}  "
      f"{corr_oracle_w_requant:8.6f}  "
      f"{1.0 - corr_oracle_w_requant:14.6f}")
print(f"  {'B. Oracle input (float input quant + INT8 DW weights)':<60s}  "
      f"{corr_oracle_input:8.6f}  "
      f"{1.0 - corr_oracle_input:14.6f}")
print(f"  {'E. INT8 input + INT8 weights, no output requant':<60s}  "
      f"{corr_no_requant:8.6f}  "
      f"{1.0 - corr_no_requant:14.6f}")
print(f"  {'FULL INT8 PIPELINE (actual result)':<60s}  "
      f"{corr_dw2_overall:8.6f}  "
      f"{1.0 - corr_dw2_overall:14.6f}")

# Error attribution
print(f"\n  Error attribution (correlation drops):")
err_from_input = corr_full_float - corr_oracle_weights
err_from_output_requant = corr_no_requant - corr_dw2_overall

print(f"    Error from input quantization (conv_1 INT8):       "
      f"~{err_from_input:+.6f}")
print(f"    Error from output requantization:                   "
      f"~{err_from_output_requant:+.6f}")
print(f"    Error from weight quantization (approx):            "
      f"~{corr_oracle_weights - corr_no_requant:+.6f}")
print(f"    Combined (should sum to ~total drop):               "
      f"~{1.0 - corr_dw2_overall:+.6f}")

# ============================================================================
# STEP 5: Dead BN channel analysis
# ============================================================================

print("\n" + "=" * 90)
print("STEP 5: Dead BN channel analysis for conv_dw_2")
print("=" * 90)

print(f"\n  BN running_var for all 32 DW channels:")
print(f"  {'ch':>4s}  {'running_var':>14s}  {'gamma':>10s}  "
      f"{'beta':>10s}  {'running_mean':>14s}  "
      f"{'BN_scale':>12s}  {'dead?':>8s}")

dead_threshold = 0.01
dead_channels = []
for ch in range(32):
    bn_scale = float(g2[ch]) / np.sqrt(float(v2[ch]) + BN_EPS)
    is_dead = v2[ch] < dead_threshold
    if is_dead:
        dead_channels.append(ch)
    flag = "YES" if is_dead else ""
    print(f"  {ch:4d}  {v2[ch]:14.6f}  {g2[ch]:10.6f}  "
          f"{b2[ch]:10.6f}  {m2[ch]:14.6f}  "
          f"{bn_scale:12.4f}  {flag:>8s}")

print(f"\n  Dead channels (var < {dead_threshold}): "
      f"{len(dead_channels)}/32")
if dead_channels:
    print(f"  Dead channel indices: {dead_channels}")

# Per-channel weight magnitude AFTER BN folding
print(f"\n  Per-channel folded weight magnitude (max abs) "
      f"and INT8 utilization:")
print(f"  {'ch':>4s}  {'folded_maxabs':>14s}  {'per_ch_scale':>14s}  "
      f"{'tensor_scale':>14s}  {'int8_maxabs':>12s}  "
      f"{'utilization':>12s}")
for ch in range(32):
    fm = per_ch_max[ch]
    pcs = pc_ws2[ch]
    # INT8 weight with per-tensor scale
    i8max = int(np.max(np.abs(wi2_flat[ch])))
    utilization = i8max / 127.0 * 100
    flag = " <-- DEAD" if ch in dead_channels else ""
    print(f"  {ch:4d}  {fm:14.6f}  {pcs:14.8f}  {ws2:14.8f}  "
          f"{i8max:12d}  {utilization:11.1f}%{flag}")

# Show INT8 weights for dead channels
if dead_channels:
    print(f"\n  INT8 weights for dead channels "
          f"(per-tensor quantization):")
    for ch in dead_channels:
        print(f"\n    Channel {ch} (var={v2[ch]:.6f}, "
              f"folded_maxabs={per_ch_max[ch]:.6f}):")
        print(f"      Float 3x3:\n{wg2[ch]}")
        print(f"      INT8 3x3 (per-tensor):\n{wi2[ch]}")
        print(f"      INT8 3x3 (per-channel):\n{pc_wi2[ch]}")
        # Quantization error for this channel's weights
        w_dequant = wi2[ch].astype(np.float64) * ws2
        w_err = np.abs(wg2[ch] - w_dequant)
        rel_err = w_err / (np.abs(wg2[ch]) + 1e-10)
        print(f"      Weight quant error (abs): "
              f"max={w_err.max():.6f} mean={w_err.mean():.6f}")
        print(f"      Weight quant error (rel): "
              f"max={rel_err.max():.4f} mean={rel_err.mean():.4f}")

# ============================================================================
# STEP 6: Re-injection test -- fix conv_dw_2 and check conv_3
# ============================================================================

print("\n" + "=" * 90)
print("STEP 6: Re-injection test -- does fixing conv_dw_2 fix conv_3?")
print("=" * 90)

# Prepare conv_3 weights (1x1, 32 -> 16, linear)
cw3, g3, b3, m3, v3 = get_bn(sd, "mobilenet_v2.conv_stem.reduce_1x1")
wf3, bf3 = fold_bn(cw3, g3, b3, m3, v3)

# y_range for conv_3 (linear, no activation) -- BN estimate
yr_conv3_bn = max(
    float(np.max(np.abs(bf3) + 6.0 * np.abs(g3))), 1.0)
y_scale_conv3 = yr_conv3_bn / 127.0

wg3 = wf3.reshape(wf3.shape[0], -1).T  # (32, 16)
wi3, ws3 = qw(wg3)

print(f"\n  conv_3: 1x1, 32->16, linear (no activation)")
print(f"  w_scale={ws3:.8f}, y_range={yr_conv3_bn:.4f}, "
      f"y_scale={y_scale_conv3:.8f}")

# ---- Test A: Full INT8 pipeline conv_3 ----
bi3_pipeline = qb(bf3, ws3 * x_scale_dw2)
os3_pipeline = (ws3 * x_scale_dw2) / y_scale_conv3

int8_conv3_pipeline = conv1x1_op(int8_dw2_relu, wi3,
                                  bi3_pipeline, os3_pipeline)
dequant_conv3_pipeline = (int8_conv3_pipeline.astype(np.float64)
                          * y_scale_conv3)
corr_conv3_pipeline = np.corrcoef(
    float_conv3.flatten(), dequant_conv3_pipeline.flatten())[0, 1]

print(f"\n  A. conv_3 from INT8 pipeline "
      f"(int8_dw2 -> int8_conv3):")
print(f"     Correlation: {corr_conv3_pipeline:.6f}")

# ---- Test B: Re-inject float conv_dw_2 output (quantized) ----
# Quantize FLOAT conv_dw_2 output to INT8 at y_range=6.0
float_dw2_r6_int8 = np.clip(
    np.round(float_dw2_r6 / x_scale_dw2), -128, 127).astype(np.int8)
float_dw2_r6_int8 = np.maximum(float_dw2_r6_int8, np.int8(0))

# Run conv_3 with this "correct" input
int8_conv3_reinjected = conv1x1_op(float_dw2_r6_int8, wi3,
                                    bi3_pipeline, os3_pipeline)
dequant_conv3_reinjected = (int8_conv3_reinjected.astype(np.float64)
                            * y_scale_conv3)
corr_conv3_reinjected = np.corrcoef(
    float_conv3.flatten(), dequant_conv3_reinjected.flatten())[0, 1]

print(f"\n  B. conv_3 from RE-INJECTED float conv_dw_2 "
      f"(quantized to INT8):")
print(f"     Correlation: {corr_conv3_reinjected:.6f}")
print(f"     Improvement over pipeline: "
      f"{corr_conv3_reinjected - corr_conv3_pipeline:+.6f}")

# ---- Test C: Re-inject using oracle input to conv_dw_2 ----
int8_dw2_oracle_input_relu = np.maximum(out_oracle_input, np.int8(0))
int8_conv3_oracle_dw = conv1x1_op(int8_dw2_oracle_input_relu, wi3,
                                    bi3_pipeline, os3_pipeline)
dequant_conv3_oracle_dw = (int8_conv3_oracle_dw.astype(np.float64)
                           * y_scale_conv3)
corr_conv3_oracle_dw = np.corrcoef(
    float_conv3.flatten(), dequant_conv3_oracle_dw.flatten())[0, 1]

print(f"\n  C. conv_3 from oracle-input DW "
      f"(float conv_1 quant -> INT8 DW -> conv_3):")
print(f"     Correlation: {corr_conv3_oracle_dw:.6f}")

# ---- Summary ----
print(f"\n  conv_3 correlation summary:")
print(f"  {'Source':<60s}  {'corr':>8s}")
print(f"  {'-' * 60}  {'-' * 8}")
print(f"  {'A. Full INT8 pipeline':<60s}  "
      f"{corr_conv3_pipeline:8.6f}")
print(f"  {'C. Oracle input DW -> INT8 conv_3':<60s}  "
      f"{corr_conv3_oracle_dw:8.6f}")
print(f"  {'B. Re-injected float dw2 output (quantized) -> conv_3':<60s}  "
      f"{corr_conv3_reinjected:8.6f}")

if corr_conv3_reinjected > corr_conv3_pipeline + 0.05:
    print(f"\n  CONCLUSION: Re-injecting correct conv_dw_2 output "
          f"SIGNIFICANTLY improves conv_3.")
    print(f"  The conv_dw_2 error IS the primary cause of "
          f"downstream degradation.")
else:
    print(f"\n  CONCLUSION: Re-injecting correct conv_dw_2 output "
          f"does NOT significantly help.")
    print(f"  conv_3's own quantization is also a major source "
          f"of error.")

# ============================================================================
# STEP 7: What-if analysis -- per-channel DW quantization
# ============================================================================

print("\n" + "=" * 90)
print("STEP 7: What-if -- per-channel DW weight quantization")
print("=" * 90)

# Per-channel DW conv: each channel gets its own w_scale
# But output_scale must still be a scalar for Gemmini
# Strategy: use per-channel w_scale for quantization, then per-channel
# output_scale (which DW conv naturally supports)

# Per-channel weight quantization
pc_bi2 = np.zeros(32, dtype=np.int32)
for ch in range(32):
    cs = pc_ws2[ch] * x_scale_conv1
    if cs < 1e-10:
        pc_bi2[ch] = 0
    else:
        pc_bi2[ch] = np.clip(
            np.round(bf2[ch] / cs), -(2**31), 2**31 - 1).astype(np.int32)

# Per-channel output scales
pc_os2 = (pc_ws2 * x_scale_conv1) / (yr_dw2 / 127.0)

# Run per-channel DW conv
N, C, H, W = int8_conv1.shape
kH = kW = 3
OH = (H + 2 * 1 - kH) // 1 + 1
OW = (W + 2 * 1 - kW) // 1 + 1
x_pad = np.pad(int8_conv1, ((0, 0), (0, 0), (1, 1), (1, 1)),
                constant_values=0)
acc_pc = np.zeros((N, C, OH, OW), dtype=np.int64)
x32 = x_pad.astype(np.int32)
w32_pc = pc_wi2.astype(np.int32)
for ki in range(kH):
    for kj in range(kW):
        rows = np.arange(OH) * 1 + ki
        cols = np.arange(OW) * 1 + kj
        acc_pc += (x32[:, :, rows[:, None], cols[None, :]]
                   * w32_pc[:, ki, kj].reshape(1, C, 1, 1))
acc_pc += pc_bi2.reshape(1, C, 1, 1).astype(np.int64)

# Apply per-channel output scales
int8_dw2_pc = np.clip(
    np.round(acc_pc.astype(np.float64)
             * pc_os2.reshape(1, C, 1, 1)),
    -128, 127).astype(np.int8)
int8_dw2_pc_relu = np.maximum(int8_dw2_pc, np.int8(0))
dequant_dw2_pc = int8_dw2_pc_relu.astype(np.float64) * x_scale_dw2

corr_dw2_pc = np.corrcoef(float_dw2_r6.flatten(),
                           dequant_dw2_pc.flatten())[0, 1]
mae_dw2_pc = np.mean(np.abs(float_dw2_r6 - dequant_dw2_pc))

print(f"\n  Per-channel DW weight quantization:")
print(f"  Overall correlation: {corr_dw2_pc:.6f} "
      f"(was {corr_dw2_overall:.6f} per-tensor)")
print(f"  Overall MAE:         {mae_dw2_pc:.6f} "
      f"(was {mae_dw2_overall:.6f})")
print(f"  Improvement:         "
      f"{corr_dw2_pc - corr_dw2_overall:+.6f}")

# Per-channel comparison
print(f"\n  Per-channel output correlation "
      f"(per-tensor vs per-channel DW quant):")
print(f"  {'ch':>4s}  {'per_tensor':>12s}  "
      f"{'per_channel':>12s}  {'delta':>10s}")
for ch in range(32):
    f_ch = float_dw2_r6[0, ch].flatten()
    d_pt = (int8_dw2_relu[0, ch].flatten().astype(np.float64)
            * x_scale_dw2)
    d_pc = (int8_dw2_pc_relu[0, ch].flatten().astype(np.float64)
            * x_scale_dw2)
    if f_ch.std() < 1e-10:
        c_pt = c_pc = 0.0
    else:
        c_pt = (np.corrcoef(f_ch, d_pt)[0, 1]
                if d_pt.std() > 1e-10 else 0.0)
        c_pc = (np.corrcoef(f_ch, d_pc)[0, 1]
                if d_pc.std() > 1e-10 else 0.0)
    delta = c_pc - c_pt
    flag = " <-- IMPROVED" if delta > 0.05 else ""
    print(f"  {ch:4d}  {c_pt:12.4f}  {c_pc:12.4f}  "
          f"{delta:+10.4f}{flag}")

# ============================================================================
# STEP 8: What-if -- dead channel zeroing + per-tensor quant
# ============================================================================

print("\n" + "=" * 90)
print("STEP 8: What-if -- dead channel zeroing + per-tensor DW quant")
print("=" * 90)

# Zero the folded weights and biases of dead channels, then re-quantize
wg2_zeroed = wg2.copy()
bf2_zeroed = bf2.copy()
for ch in dead_channels:
    wg2_zeroed[ch] = 0.0
    bf2_zeroed[ch] = 0.0

wg2z_flat = wg2_zeroed.reshape(wg2_zeroed.shape[0], -1)
wi2z_flat, ws2z = qw(wg2z_flat)
wi2z = wi2z_flat.reshape(wg2_zeroed.shape)
bi2z = qb(bf2_zeroed, ws2z * x_scale_conv1)
os2z = (ws2z * x_scale_conv1) / (yr_dw2 / 127.0)

int8_dw2_z = dw_conv_op(int8_conv1, wi2z, bi2z, os2z, 1, 1)
int8_dw2_z_relu = np.maximum(int8_dw2_z, np.int8(0))
dequant_dw2_z = int8_dw2_z_relu.astype(np.float64) * x_scale_dw2

corr_dw2_z = np.corrcoef(float_dw2_r6.flatten(),
                          dequant_dw2_z.flatten())[0, 1]
mae_dw2_z = np.mean(np.abs(float_dw2_r6 - dequant_dw2_z))

ws2z_safe = ws2z if ws2z > 1e-10 else 1e-10
print(f"\n  Dead channel zeroing (threshold={dead_threshold}):")
print(f"  Zeroed channels: {dead_channels}")
print(f"  New w_scale: {ws2z:.6f} (was {ws2:.6f})")
print(f"  Scale reduction: {ws2 / ws2z_safe:.2f}x")
print(f"  Overall correlation: {corr_dw2_z:.6f} "
      f"(was {corr_dw2_overall:.6f})")
print(f"  Overall MAE:         {mae_dw2_z:.6f} "
      f"(was {mae_dw2_overall:.6f})")
print(f"  Improvement:         "
      f"{corr_dw2_z - corr_dw2_overall:+.6f}")

# Also feed dead-zeroed output into conv_3
bi3_z = qb(bf3, ws3 * x_scale_dw2)
os3_z = (ws3 * x_scale_dw2) / y_scale_conv3
int8_conv3_from_zeroed = conv1x1_op(int8_dw2_z_relu, wi3, bi3_z, os3_z)
dequant_conv3_from_zeroed = (int8_conv3_from_zeroed.astype(np.float64)
                             * y_scale_conv3)
corr_conv3_from_zeroed = np.corrcoef(
    float_conv3.flatten(), dequant_conv3_from_zeroed.flatten())[0, 1]
print(f"\n  conv_3 with dead-zeroed DW: "
      f"corr={corr_conv3_from_zeroed:.6f} "
      f"(was {corr_conv3_pipeline:.6f})")

# ============================================================================
# STEP 9: Pre-BN conv_dw_2 analysis
# ============================================================================

print("\n" + "=" * 90)
print("STEP 9: Pre-BN conv_dw_2 output analysis")
print("=" * 90)

if "conv_dw_2_pre_bn" in float_acts:
    pre_bn = float_acts["conv_dw_2_pre_bn"]
    print(f"  Pre-BN conv_dw_2 output shape: {pre_bn.shape}")
    print(f"  Global range: [{pre_bn.min():.4f}, {pre_bn.max():.4f}]")
    print()

    print(f"  Per-channel pre-BN statistics:")
    print(f"  {'ch':>4s}  {'mean':>10s}  {'std':>10s}  "
          f"{'min':>10s}  {'max':>10s}  "
          f"{'bn_var':>10s}  {'bn_mean':>10s}")
    for ch in range(32):
        pb_ch = pre_bn[0, ch].flatten()
        flag = " <-- DEAD" if ch in dead_channels else ""
        print(f"  {ch:4d}  {pb_ch.mean():10.4f}  "
              f"{pb_ch.std():10.4f}  "
              f"{pb_ch.min():10.4f}  {pb_ch.max():10.4f}  "
              f"{v2[ch]:10.6f}  {m2[ch]:10.4f}{flag}")
else:
    print("  (Pre-BN activation not captured)")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 90)
print("SUMMARY: Why conv_dw_2 has low correlation")
print("=" * 90)

ws2z_safe = ws2z if ws2z > 1e-10 else 1e-10
print(f"""
conv_dw_2: depthwise 3x3, 32 channels, stride 1, pad 1, ReLU6
  INT8 pipeline correlation:       {corr_dw2_overall:.6f}

Error decomposition:
  A. Full float DW (reference):    {corr_full_float:.6f}
  C. Oracle weights (input err):   {corr_oracle_weights:.6f}  (input quant costs {corr_full_float - corr_oracle_weights:.6f})
  B. Oracle input (weight err):    {corr_oracle_input:.6f}  (weight quant costs {corr_full_float - corr_oracle_input:.6f})
  E. Both INT8, no requant:        {corr_no_requant:.6f}  (output requant costs {corr_no_requant - corr_dw2_overall:.6f})
  Full INT8 pipeline:              {corr_dw2_overall:.6f}

Dead channels (BN var < {dead_threshold}): {len(dead_channels)}/32  {dead_channels}
  Per-tensor w_scale:              {ws2:.6f}
  After dead-channel zeroing:      {ws2z:.6f} ({ws2 / ws2z_safe:.2f}x smaller)
  DW corr with dead zeroing:       {corr_dw2_z:.6f} (delta {corr_dw2_z - corr_dw2_overall:+.6f})
  DW corr with per-channel quant:  {corr_dw2_pc:.6f} (delta {corr_dw2_pc - corr_dw2_overall:+.6f})

Downstream impact on conv_3:
  Full pipeline conv_3:            {corr_conv3_pipeline:.6f}
  Re-inject float dw2 -> conv_3:   {corr_conv3_reinjected:.6f} (delta {corr_conv3_reinjected - corr_conv3_pipeline:+.6f})
  Dead-zeroed dw2 -> conv_3:       {corr_conv3_from_zeroed:.6f} (delta {corr_conv3_from_zeroed - corr_conv3_pipeline:+.6f})
""")
