#!/usr/bin/env python3
"""Test: zero dead BN channels' folded weights+biases entirely, then trace correlation.

Key insight vs variance-floor clamping (test_varfloor.py):
  Clamping variance still leaves dead channels with nonzero (but smaller) weights
  that can dominate the per-tensor w_scale.  Zeroing them completely removes their
  impact on w_scale, letting the remaining live channels use more of the INT8 range.

Also tests a combined approach: dead-channel zeroing + per-channel DW weight
quantization (each DW channel gets its own w_scale, but output_scale stays per-tensor).
"""
import numpy as np, torch
from transformers import MobileNetV2ForImageClassification
import cv2, os

MODEL_NAME = "google/mobilenet_v2_1.0_224"
BN_EPS = 0.001
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float64)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float64)

print("Loading model...")
model = MobileNetV2ForImageClassification.from_pretrained(MODEL_NAME)
model.eval()
sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
nc = sd["classifier.weight"].shape[0]

# ---------------------------------------------------------------------------
# BN folding helpers
# ---------------------------------------------------------------------------

def fold_bn(w, g, b, m, v, eps=BN_EPS):
    """Standard BN fold (no modification)."""
    inv = 1.0 / np.sqrt(v + eps)
    s = g * inv
    shape = [w.shape[0]] + [1] * (w.ndim - 1)
    return w * s.reshape(shape), b - g * m * inv


def fold_bn_dead_zero(w, g, b, m, v, eps=BN_EPS, threshold=0.01):
    """BN fold with dead-channel zeroing.

    After folding, any channel whose running_var < threshold has its folded
    weight AND bias set to exactly 0.  This removes dead channels from
    influencing per-tensor w_scale entirely.
    """
    inv = 1.0 / np.sqrt(v + eps)
    s = g * inv
    shape = [w.shape[0]] + [1] * (w.ndim - 1)
    w_folded = w * s.reshape(shape)
    b_folded = b - g * m * inv

    dead = v < threshold
    n_dead = int(np.sum(dead))

    # Zero out dead channels completely
    if n_dead > 0:
        w_folded[dead] = 0.0
        b_folded[dead] = 0.0

    return w_folded, b_folded, n_dead


def get_bn(sd, pfx):
    return tuple(sd[f"{pfx}.{k}"].float().numpy().astype(np.float64) for k in
                 ["convolution.weight", "normalization.weight", "normalization.bias",
                  "normalization.running_mean", "normalization.running_var"])


# ---------------------------------------------------------------------------
# Quantisation helpers
# ---------------------------------------------------------------------------

def qw(w):
    """Per-tensor symmetric INT8 weight quantisation."""
    mx = float(np.max(np.abs(w)))
    if mx < 1e-10:
        return np.zeros_like(w, dtype=np.int8), 1e-10
    s = mx / 127.0
    return np.clip(np.round(w / s), -128, 127).astype(np.int8), s


def qw_perchannel_dw(w_2d):
    """Per-channel symmetric INT8 quantisation for DW conv weights.

    w_2d shape: (C, kH*kW) -- each row is one channel's kernel.
    Returns:
        w_int: (C, kH*kW) int8
        scales: (C,) float64  -- one scale per channel
    """
    C = w_2d.shape[0]
    w_int = np.zeros_like(w_2d, dtype=np.int8)
    scales = np.zeros(C, dtype=np.float64)
    for c in range(C):
        mx = float(np.max(np.abs(w_2d[c])))
        if mx < 1e-10:
            scales[c] = 1e-10
        else:
            scales[c] = mx / 127.0
            w_int[c] = np.clip(np.round(w_2d[c] / scales[c]), -128, 127).astype(np.int8)
    return w_int, scales


def qb(b, cs):
    if cs < 1e-10:
        return np.zeros_like(b, dtype=np.int32)
    return np.clip(np.round(b / cs), -(2**31), 2**31 - 1).astype(np.int32)


def qb_perchannel(b, scales):
    """Quantise bias with per-channel combined scale (w_scale[c] * x_scale)."""
    C = b.shape[0]
    bi = np.zeros(C, dtype=np.int32)
    for c in range(C):
        cs = scales[c]
        if cs < 1e-10:
            continue
        bi[c] = int(np.clip(np.round(b[c] / cs), -(2**31), 2**31 - 1))
    return bi


# ---------------------------------------------------------------------------
# Convolution operators
# ---------------------------------------------------------------------------

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
    """Standard DW conv with per-tensor output_scale (scalar os_val)."""
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


def dw_conv_op_perchannel(x, w_int, b_int, per_ch_os, stride, pad):
    """DW conv with per-channel accumulation scales, producing per-tensor INT8 output.

    per_ch_os: (C,) array  --  output_scale per channel = (w_scale[c]*x_scale) / y_scale
    The result is still a per-tensor INT8 tensor (uniform y_scale), so downstream
    layers don't need to change.
    """
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
    # Per-channel rescale to shared y_scale
    out = np.clip(
        np.round(acc.astype(np.float64) * per_ch_os.reshape(1, C, 1, 1)),
        -128, 127
    ).astype(np.int8)
    return out


# ---------------------------------------------------------------------------
# Layer definitions
# ---------------------------------------------------------------------------

ALL_LAYERS = [
    ("conv_1",    "mobilenet_v2.conv_stem.first_conv", 3, 2, 1, False, True),
    ("conv_dw_2", "mobilenet_v2.conv_stem.conv_3x3",   3, 1, 1, True,  True),
    ("conv_3",    "mobilenet_v2.conv_stem.reduce_1x1", 1, 1, 0, False, False),
]
_idx = 4
_dw_strides = {0: 2, 2: 2, 5: 2, 12: 2}
for _hi in range(16):
    _pfx = f"mobilenet_v2.layer.{_hi}"
    _ds = _dw_strides.get(_hi, 1)
    ALL_LAYERS.append((f"conv_{_idx}",    f"{_pfx}.expand_1x1", 1, 1,  0, False, True));  _idx += 1
    ALL_LAYERS.append((f"conv_dw_{_idx}", f"{_pfx}.conv_3x3",   3, _ds, 1, True,  True));  _idx += 1
    ALL_LAYERS.append((f"conv_{_idx}",    f"{_pfx}.reduce_1x1", 1, 1,  0, False, False)); _idx += 1
ALL_LAYERS.append(("conv_52", "mobilenet_v2.conv_1x1", 1, 1, 0, False, True))

RESIDUAL_SKIP = {
    "conv_9":  "conv_6",  "conv_15": "conv_12", "conv_18": "conv_15",
    "conv_24": "conv_21", "conv_27": "conv_24", "conv_30": "conv_27",
    "conv_36": "conv_33", "conv_39": "conv_36", "conv_45": "conv_42",
    "conv_48": "conv_45",
}

KEY_LAYERS = ["conv_1", "conv_dw_2", "conv_3", "conv_6", "conv_12",
              "conv_33", "conv_52"]

# ---------------------------------------------------------------------------
# Float reference run
# ---------------------------------------------------------------------------

float_acts = {}

def capture(name):
    def hook(module, inp, out):
        float_acts[name] = out.detach().float().numpy().astype(np.float64)
    return hook

hooks = []
for gn, pfx, k, s, pad, dw, relu in ALL_LAYERS:
    parts = pfx.split(".")
    mod2 = model
    for pt in parts:
        mod2 = getattr(mod2, pt)
    h = mod2.normalization.register_forward_hook(capture(gn))
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
print(f"Float pred: {float_pred}")

# ---------------------------------------------------------------------------
# Prepare INT8 input
# ---------------------------------------------------------------------------

img_int8 = (img_rgb.astype(np.int16) - 128).clip(-128, 127).astype(np.int8)
x_in = img_int8.transpose(2, 0, 1)[np.newaxis]


# ---------------------------------------------------------------------------
# Helper: run one INT8 inference pass
# ---------------------------------------------------------------------------

def run_int8_pass(threshold, mode="zero_only"):
    """Run full INT8 inference with dead-channel zeroing.

    mode:
        "zero_only"  -- dead-channel zeroing + per-tensor w_scale (like baseline)
        "zero+pchdw" -- dead-channel zeroing + per-channel DW w_scale
    """
    out_int = x_in.copy()
    stored = {}
    x_scale = 128.0 / 127.0
    layer_y_range = {}
    total_dead = 0

    for gn, pfx, k, s, pad, dw, relu in ALL_LAYERS:
        cw, gamma, beta, bn_m, bn_v = get_bn(sd, pfx)

        # --- BN fold with dead-channel zeroing ---
        if threshold > 0:
            wf, bf, n_dead = fold_bn_dead_zero(cw, gamma, beta, bn_m, bn_v,
                                                threshold=threshold)
            total_dead += n_dead
        else:
            wf, bf = fold_bn(cw, gamma, beta, bn_m, bn_v)

        # --- Absorb ImageNet normalisation into first layer ---
        if gn == "conv_1":
            no = (128.0 / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
            for c in range(3):
                bf += wf[:, c, :, :].sum(axis=(1, 2)) * no[c]
            for c in range(3):
                wf[:, c, :, :] /= (255.0 * IMAGENET_STD[c])

        # --- y_range ---
        yr = 6.0 if relu else max(float(np.max(np.abs(beta) + 6.0 * np.abs(gamma))), 1.0)
        layer_y_range[gn] = yr

        # --- Quantise & compute ---
        if dw:
            wg = wf.squeeze(1)  # (C, kH, kW)

            if mode == "zero+pchdw":
                # Per-channel DW weight quantisation
                wg_flat = wg.reshape(wg.shape[0], -1)  # (C, 9)
                wi_flat, w_scales = qw_perchannel_dw(wg_flat)  # per-channel scales
                wi = wi_flat.reshape(wg.shape)  # back to (C, kH, kW)

                # Per-channel bias quantisation
                combined_scales = w_scales * x_scale  # (C,)
                bi = qb_perchannel(bf, combined_scales)

                # Per-channel output_scale:  (w_scale[c]*x_scale) / y_scale
                y_scale = yr / 127.0
                per_ch_os = combined_scales / y_scale  # (C,)

                out_int = dw_conv_op_perchannel(out_int, wi, bi, per_ch_os, s, pad)
            else:
                # Standard per-tensor DW
                wg_flat = wg.reshape(wg.shape[0], -1)  # (C, 9)
                wi_flat, ws = qw(wg_flat)
                wi = wi_flat.reshape(wg.shape)
                bi = qb(bf, ws * x_scale)
                os_val = (ws * x_scale) / (yr / 127.0)
                out_int = dw_conv_op(out_int, wi, bi, os_val, s, pad)
        else:
            wg = wf.reshape(wf.shape[0], -1).T
            wi, ws = qw(wg)
            bi = qb(bf, ws * x_scale)
            os_val = (ws * x_scale) / (yr / 127.0)
            out_int = conv_op(out_int, wi, bi, os_val, k, s, pad)

        if relu:
            out_int = np.maximum(out_int, np.int8(0))

        stored[gn] = out_int

        # --- Residual addition ---
        if gn in RESIDUAL_SKIP:
            skip_src = RESIDUAL_SKIP[gn]
            skip = stored[skip_src]
            rs = layer_y_range[skip_src] / yr
            out_int = np.clip(
                np.round(skip.astype(np.float64) * rs + out_int.astype(np.float64)),
                -128, 127
            ).astype(np.int8)
            stored[gn] = out_int

        x_scale = yr / 127.0

    # --- Global avg pool ---
    avg = out_int.astype(np.float64).mean(axis=(2, 3))
    avg_int8 = np.clip(np.round(avg), -128, 127).astype(np.int8)

    # Correlation of avg-pool output with float reference
    float_conv52_r6 = np.clip(float_acts["conv_52"], 0, 6.0)
    float_avg = float_conv52_r6.mean(axis=(2, 3))
    corr_avg = np.corrcoef(float_avg[0], avg_int8[0].astype(np.float64))[0, 1]

    # --- FC layer ---
    fc_w = sd["classifier.weight"].numpy().astype(np.float64)
    fc_b = sd["classifier.bias"].numpy().astype(np.float64)
    if nc == 1001:
        fc_w = fc_w[1:, :]
        fc_b = fc_b[1:]
    fc_wi, fc_ws = qw(fc_w)
    fc_bi = qb(fc_b, fc_ws * x_scale)
    logits = fc_wi.astype(np.int32) @ avg_int8[0].astype(np.int32) + fc_bi.astype(np.int32)
    pred = int(np.argmax(logits))

    # --- Per-layer correlation at key layers ---
    corrs = {}
    for layer_name in KEY_LAYERS:
        float_ref = float_acts.get(layer_name)
        if float_ref is None:
            continue
        has_relu = any(layer_name == l[0] and l[6] for l in ALL_LAYERS)
        if has_relu:
            float_ref = np.clip(float_ref, 0, 6.0)
        yr_val = layer_y_range.get(layer_name, 1.0)
        int8_val = stored.get(layer_name)
        if int8_val is not None and float_ref.shape == int8_val.shape:
            dequant = int8_val.astype(np.float64) * (yr_val / 127.0)
            corrs[layer_name] = np.corrcoef(float_ref.flatten(), dequant.flatten())[0, 1]

    return pred, corr_avg, corrs, total_dead, stored, layer_y_range


# ---------------------------------------------------------------------------
# Experiment 1: Dead-channel zeroing at various thresholds (per-tensor w_scale)
# ---------------------------------------------------------------------------

print("\n" + "=" * 90)
print("EXPERIMENT 1: Dead-channel zeroing (per-tensor w_scale)")
print("=" * 90)
thresholds = [0.0, 0.001, 0.01, 0.1, 1.0]

for thr in thresholds:
    pred, corr_avg, corrs, total_dead, _, _ = run_int8_pass(thr, mode="zero_only")
    corr_str = "  ".join(f"{k}:{v:.3f}" for k, v in corrs.items())
    match_str = "MATCH" if pred == float_pred else "MISMATCH"
    print(f"  thr={thr:<6.3f}  dead_channels={total_dead:4d}  pred={pred:4d} ({match_str})"
          f"  avg_corr={corr_avg:.4f}  {corr_str}")

# ---------------------------------------------------------------------------
# Experiment 2: Dead-channel zeroing + per-channel DW weight quantisation
# ---------------------------------------------------------------------------

print("\n" + "=" * 90)
print("EXPERIMENT 2: Dead-channel zeroing + per-channel DW w_scale")
print("  (each DW channel gets its own w_scale; output_scale stays per-tensor)")
print("=" * 90)

for thr in thresholds:
    pred, corr_avg, corrs, total_dead, _, _ = run_int8_pass(thr, mode="zero+pchdw")
    corr_str = "  ".join(f"{k}:{v:.3f}" for k, v in corrs.items())
    match_str = "MATCH" if pred == float_pred else "MISMATCH"
    print(f"  thr={thr:<6.3f}  dead_channels={total_dead:4d}  pred={pred:4d} ({match_str})"
          f"  avg_corr={corr_avg:.4f}  {corr_str}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print("\n" + "=" * 90)
print("SUMMARY")
print("=" * 90)
print(f"Float prediction: {float_pred}")
print()
print("The goal is to find the combination that matches the float prediction")
print("with the highest per-layer correlations, especially at conv_52 and avg_pool.")
print()
print("dead-channel zeroing removes channels with running_var < threshold entirely")
print("  => those channels contribute ZERO to per-tensor w_scale")
print("  => remaining live channels get more INT8 dynamic range")
print()
print("per-channel DW w_scale gives each DW filter its own scale")
print("  => no single outlier channel can dominate the weight range")
print("  => combined with zeroing, this should be the best approach")
