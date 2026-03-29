#!/usr/bin/env python3
"""Test: combined approach to fix MobileNetV2 INT8 quantization.

Key insight: Dead BN channels (running_var ~ 0) exist at MULTIPLE layers.
When their preceding layer output is non-zero due to quantization noise,
this noise propagates into layers where those channels either have:
  (a) exploded weights (from 1/sqrt(var+eps)), or
  (b) crushed weights (from per-tensor w_scale dominated by exploded channels).
Either way it creates garbage.

Strategy - "Consistent Dead Channel Zeroing":
1. Identify dead channels at each BN layer (running_var < threshold)
2. For each layer: zero own dead channels' folded weights+biases
3. Cross-layer propagation: if the NEXT layer is DW and has dead channels,
   also zero the current layer's output columns for those channels
   (prevents quantization noise from reaching dead DW channels)
4. Optionally combine with per-channel DW weight quantization
"""
import numpy as np
import torch
from transformers import MobileNetV2ForImageClassification
import cv2

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
              "conv_21", "conv_33", "conv_42", "conv_52"]

# ---------------------------------------------------------------------------
# BN folding helpers
# ---------------------------------------------------------------------------

def get_bn(sd, pfx):
    return tuple(sd[f"{pfx}.{k}"].float().numpy().astype(np.float64) for k in
                 ["convolution.weight", "normalization.weight", "normalization.bias",
                  "normalization.running_mean", "normalization.running_var"])


def fold_bn(w, g, b, m, v, eps=BN_EPS):
    """Standard BN fold (no modification)."""
    inv = 1.0 / np.sqrt(v + eps)
    s = g * inv
    shape = [w.shape[0]] + [1] * (w.ndim - 1)
    return w * s.reshape(shape), b - g * m * inv


def fold_bn_combined_dead_zero(w, g, b, m, v, eps=BN_EPS,
                                own_dead=None, output_dead=None):
    """BN fold with combined dead-channel zeroing.

    own_dead:    bool array (out_ch,) - dead from this layer's BN running_var
    output_dead: bool array (out_ch,) - dead from next DW layer's BN running_var
                 (set only when the next layer is DW, so channel dims match 1:1)

    Both masks index output channels (rows of the weight tensor).
    Combined zeroing ensures:
      - Dead channels produce zero output and zero bias
      - Zeroed channels don't inflate per-tensor w_scale
    """
    inv = 1.0 / np.sqrt(v + eps)
    s = g * inv
    shape = [w.shape[0]] + [1] * (w.ndim - 1)
    wf = w * s.reshape(shape)
    bf = b - g * m * inv

    kill = np.zeros(w.shape[0], dtype=bool)
    if own_dead is not None:
        kill |= own_dead
    if output_dead is not None:
        kill |= output_dead

    n_killed = int(np.sum(kill))
    n_own = int(np.sum(own_dead)) if own_dead is not None else 0
    n_prop = n_killed - n_own  # channels killed ONLY due to cross-layer propagation

    if n_killed > 0:
        wf[kill] = 0.0
        bf[kill] = 0.0

    return wf, bf, n_killed, n_own, n_prop


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


def qw_perchannel_dw(wg):
    """Per-channel symmetric INT8 quantisation for DW conv weights.

    wg shape: (C, kH, kW)
    Returns: w_int (C, kH, kW) int8, scales (C,) float64
    """
    C = wg.shape[0]
    flat = wg.reshape(C, -1)
    w_int_flat = np.zeros_like(flat, dtype=np.int8)
    scales = np.zeros(C, dtype=np.float64)
    for c in range(C):
        mx = float(np.max(np.abs(flat[c])))
        if mx < 1e-10:
            scales[c] = 1e-10
        else:
            scales[c] = mx / 127.0
            w_int_flat[c] = np.clip(np.round(flat[c] / scales[c]),
                                    -128, 127).astype(np.int8)
    return w_int_flat.reshape(wg.shape), scales


def qb(b, cs):
    if cs < 1e-10:
        return np.zeros_like(b, dtype=np.int32)
    return np.clip(np.round(b / cs), -(2**31), 2**31 - 1).astype(np.int32)


def qb_perchannel(b, scales):
    """Quantise bias with per-channel combined scale."""
    C = b.shape[0]
    bi = np.zeros(C, dtype=np.int32)
    for c in range(C):
        if scales[c] < 1e-10:
            continue
        bi[c] = int(np.clip(np.round(b[c] / scales[c]), -(2**31), 2**31 - 1))
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
            x = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)),
                        constant_values=0)
        patches = np.zeros((N, OH, OW, kH * kW * C), dtype=x.dtype)
        for i in range(OH):
            for j in range(OW):
                p = x[:, :, i*stride:i*stride+kH, j*stride:j*stride+kW]
                patches[:, i, j, :] = p.transpose(0, 2, 3, 1).reshape(N, -1)
        xf = patches.reshape(N * OH * OW, kH * kW * C)
        oh = OH; ow = OW
    acc = (xf.astype(np.int32) @ w_int.astype(np.int32)
           + b_int.reshape(1, -1).astype(np.int32))
    yf = np.clip(np.round(acc.astype(np.float64) * os_val),
                 -128, 127).astype(np.int8)
    return yf.reshape(N, oh, ow, w_int.shape[1]).transpose(0, 3, 1, 2)


def dw_conv_op(x, w_int, b_int, os_val, stride, pad):
    """DW conv with per-tensor output_scale (scalar os_val)."""
    N, C, H, W = x.shape; kH = kW = 3
    OH = (H + 2*pad - kH) // stride + 1
    OW = (W + 2*pad - kW) // stride + 1
    if pad > 0:
        x_pad = np.pad(x, ((0,0),(0,0),(pad,pad),(pad,pad)), constant_values=0)
    else:
        x_pad = x
    acc = np.zeros((N, C, OH, OW), dtype=np.int64)
    x32 = x_pad.astype(np.int32); w32 = w_int.astype(np.int32)
    for ki in range(kH):
        for kj in range(kW):
            rows = np.arange(OH) * stride + ki
            cols = np.arange(OW) * stride + kj
            acc += (x32[:, :, rows[:, None], cols[None, :]]
                    * w32[:, ki, kj].reshape(1, C, 1, 1))
    acc += b_int.reshape(1, C, 1, 1).astype(np.int64)
    return np.clip(np.round(acc.astype(np.float64) * os_val),
                   -128, 127).astype(np.int8)


def dw_conv_op_perchannel(x, w_int, b_int, per_ch_os, stride, pad):
    """DW conv with per-channel output_scale array."""
    N, C, H, W = x.shape; kH = kW = 3
    OH = (H + 2*pad - kH) // stride + 1
    OW = (W + 2*pad - kW) // stride + 1
    if pad > 0:
        x_pad = np.pad(x, ((0,0),(0,0),(pad,pad),(pad,pad)), constant_values=0)
    else:
        x_pad = x
    acc = np.zeros((N, C, OH, OW), dtype=np.int64)
    x32 = x_pad.astype(np.int32); w32 = w_int.astype(np.int32)
    for ki in range(kH):
        for kj in range(kW):
            rows = np.arange(OH) * stride + ki
            cols = np.arange(OW) * stride + kj
            acc += (x32[:, :, rows[:, None], cols[None, :]]
                    * w32[:, ki, kj].reshape(1, C, 1, 1))
    acc += b_int.reshape(1, C, 1, 1).astype(np.int64)
    return np.clip(
        np.round(acc.astype(np.float64) * per_ch_os.reshape(1, C, 1, 1)),
        -128, 127).astype(np.int8)


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

float_pred = (int(np.argmax(float_logits[1:])) if nc == 1001
              else int(np.argmax(float_logits)))
print(f"Float prediction: {float_pred}")

# ---------------------------------------------------------------------------
# Prepare INT8 input
# ---------------------------------------------------------------------------

img_int8 = (img_rgb.astype(np.int16) - 128).clip(-128, 127).astype(np.int8)
x_in = img_int8.transpose(2, 0, 1)[np.newaxis]


# =========================================================================
# PHASE 1: Dead BN channel analysis
# =========================================================================

def scan_dead_channels(sd, threshold):
    """Scan all layers, return dict of dead-channel info per layer."""
    info = {}
    total_dead = 0
    total_ch = 0
    for gn, pfx, k, s, pad, dw, relu in ALL_LAYERS:
        _, _, _, _, bn_v = get_bn(sd, pfx)
        dead_mask = bn_v < threshold
        n_dead = int(np.sum(dead_mask))
        total_dead += n_dead
        total_ch += len(bn_v)
        info[gn] = {
            "dead_mask": dead_mask,
            "n_dead": n_dead,
            "n_channels": len(bn_v),
            "dead_indices": list(np.where(dead_mask)[0]),
            "min_var": float(np.min(bn_v)),
            "max_var": float(np.max(bn_v)),
        }
    return info, total_dead, total_ch


MAIN_THRESHOLD = 0.01
print("\n" + "=" * 100)
print(f"PHASE 1: Dead BN Channel Analysis (threshold = {MAIN_THRESHOLD})")
print("=" * 100)

dead_info, total_dead, total_ch = scan_dead_channels(sd, MAIN_THRESHOLD)
print(f"\nTotal channels across all {len(ALL_LAYERS)} layers: {total_ch}")
print(f"Total dead channels (var < {MAIN_THRESHOLD}): {total_dead}")
print(f"Dead channel rate: {100.0 * total_dead / total_ch:.2f}%\n")

print(f"  {'Layer':<14s} {'Type':<6s} {'Ch':>5s} {'Dead':>5s}"
      f" {'MinVar':>12s} {'MaxVar':>12s}  Dead Indices")
print("  " + "-" * 94)
for gn, pfx, k, s, pad, dw, relu in ALL_LAYERS:
    di = dead_info[gn]
    ltype = "DW" if dw else "Conv"
    idx_str = str(di["dead_indices"]) if di["n_dead"] > 0 else ""
    print(f"  {gn:<14s} {ltype:<6s} {di['n_channels']:>5d} {di['n_dead']:>5d}"
          f" {di['min_var']:>12.6f} {di['max_var']:>12.2f}  {idx_str}")


# =========================================================================
# PHASE 2: INT8 inference engine
# =========================================================================

def run_int8_pass(threshold, mode="baseline"):
    """Run full 52-layer INT8 inference.

    Modes:
      "baseline"            -- no dead-channel fixes
      "zero_own"            -- zero dead channels at own layer only
      "zero_combined"       -- zero own + preceding layer output (cross-layer)
      "zero_combined+pchdw" -- combined zeroing + per-channel DW quantization
    """
    # Pre-scan dead channels
    dead_masks = {}
    if threshold > 0 and mode != "baseline":
        for gn, pfx, k, s, pad, dw, relu in ALL_LAYERS:
            _, _, _, _, bn_v = get_bn(sd, pfx)
            dead_masks[gn] = bn_v < threshold

    out_int = x_in.copy()
    stored = {}
    x_scale = 128.0 / 127.0
    layer_y_range = {}
    total_killed = 0
    total_own = 0
    total_propagated = 0
    per_layer_killed = {}

    for idx, (gn, pfx, k, s, pad, dw, relu) in enumerate(ALL_LAYERS):
        cw, gamma, beta, bn_m, bn_v = get_bn(sd, pfx)

        if mode == "baseline":
            wf, bf = fold_bn(cw, gamma, beta, bn_m, bn_v)
            n_killed = n_own = n_prop = 0

        else:
            own_dead = dead_masks.get(gn)

            # Cross-layer: if next layer is DW, propagate its dead channels
            output_dead = None
            if mode in ("zero_combined", "zero_combined+pchdw"):
                if idx + 1 < len(ALL_LAYERS):
                    next_gn, _, _, _, _, next_dw, _ = ALL_LAYERS[idx + 1]
                    if next_dw:
                        next_dead = dead_masks.get(next_gn)
                        if (next_dead is not None
                                and len(next_dead) == cw.shape[0]):
                            output_dead = next_dead

            wf, bf, n_killed, n_own, n_prop = fold_bn_combined_dead_zero(
                cw, gamma, beta, bn_m, bn_v,
                own_dead=own_dead, output_dead=output_dead)

        total_killed += n_killed
        total_own += n_own
        total_propagated += n_prop
        per_layer_killed[gn] = (n_killed, n_own, n_prop)

        # Absorb ImageNet normalisation into first layer
        if gn == "conv_1":
            no = (128.0 / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
            for c in range(3):
                bf += wf[:, c, :, :].sum(axis=(1, 2)) * no[c]
            for c in range(3):
                wf[:, c, :, :] /= (255.0 * IMAGENET_STD[c])

        # y_range
        yr = (6.0 if relu
              else max(float(np.max(np.abs(beta) + 6.0 * np.abs(gamma))), 1.0))
        layer_y_range[gn] = yr

        # Quantise & compute
        if dw:
            wg = wf.squeeze(1)  # (C, kH, kW)

            if mode == "zero_combined+pchdw":
                wi, w_scales = qw_perchannel_dw(wg)
                combined_scales = w_scales * x_scale
                bi = qb_perchannel(bf, combined_scales)
                y_scale = yr / 127.0
                per_ch_os = combined_scales / y_scale
                out_int = dw_conv_op_perchannel(
                    out_int, wi, bi, per_ch_os, s, pad)
            else:
                wg_flat = wg.reshape(wg.shape[0], -1)
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

        # Residual addition
        if gn in RESIDUAL_SKIP:
            skip_src = RESIDUAL_SKIP[gn]
            skip = stored[skip_src]
            rs = layer_y_range[skip_src] / yr
            out_int = np.clip(
                np.round(skip.astype(np.float64) * rs
                         + out_int.astype(np.float64)),
                -128, 127).astype(np.int8)
            stored[gn] = out_int

        x_scale = yr / 127.0

    # --- Global avg pool ---
    avg = out_int.astype(np.float64).mean(axis=(2, 3))
    avg_int8 = np.clip(np.round(avg), -128, 127).astype(np.int8)

    float_conv52_r6 = np.clip(float_acts["conv_52"], 0, 6.0)
    float_avg = float_conv52_r6.mean(axis=(2, 3))
    corr_avg = np.corrcoef(float_avg[0],
                           avg_int8[0].astype(np.float64))[0, 1]

    # --- FC layer ---
    fc_w = sd["classifier.weight"].numpy().astype(np.float64)
    fc_b = sd["classifier.bias"].numpy().astype(np.float64)
    if nc == 1001:
        fc_w = fc_w[1:, :]
        fc_b = fc_b[1:]
    fc_wi, fc_ws = qw(fc_w)
    fc_bi = qb(fc_b, fc_ws * x_scale)
    logits = (fc_wi.astype(np.int32) @ avg_int8[0].astype(np.int32)
              + fc_bi.astype(np.int32))
    pred = int(np.argmax(logits))

    # --- Per-layer correlations (ALL layers) ---
    corrs = {}
    for layer_gn, _, _, _, _, _, layer_relu in ALL_LAYERS:
        float_ref = float_acts.get(layer_gn)
        if float_ref is None:
            continue
        if layer_relu:
            float_ref = np.clip(float_ref, 0, 6.0)
        yr_val = layer_y_range.get(layer_gn, 1.0)
        int8_val = stored.get(layer_gn)
        if int8_val is not None and float_ref.shape == int8_val.shape:
            dequant = int8_val.astype(np.float64) * (yr_val / 127.0)
            corrs[layer_gn] = np.corrcoef(
                float_ref.flatten(), dequant.flatten())[0, 1]

    return {
        "pred": pred,
        "corr_avg": corr_avg,
        "corrs": corrs,
        "total_killed": total_killed,
        "total_own": total_own,
        "total_propagated": total_propagated,
        "per_layer_killed": per_layer_killed,
    }


# =========================================================================
# PHASE 2: Run all modes
# =========================================================================

print("\n" + "=" * 100)
print(f"PHASE 2: Full 52-Layer INT8 Pipeline  (threshold = {MAIN_THRESHOLD})")
print("=" * 100)

MODES = [
    ("baseline",              "Baseline (no fixes)"),
    ("zero_own",              "Dead-channel zero (own layer only)"),
    ("zero_combined",         "Dead-channel zero (cross-layer propagation)"),
    ("zero_combined+pchdw",   "Dead-channel zero (cross-layer) + per-ch DW quant"),
]

results = {}
for mode_key, mode_label in MODES:
    print(f"\n  Running: {mode_label} ...")
    r = run_int8_pass(MAIN_THRESHOLD, mode=mode_key)
    results[mode_key] = r
    match = "MATCH" if r["pred"] == float_pred else "MISMATCH"
    print(f"    Prediction: {r['pred']}  ({match})"
          f"  |  Avg-pool corr: {r['corr_avg']:.6f}"
          f"  |  Killed: {r['total_killed']}"
          f" (own={r['total_own']}, propagated={r['total_propagated']})")


# =========================================================================
# PHASE 3: Per-layer correlation comparison (ALL 52 layers)
# =========================================================================

print("\n" + "=" * 100)
print("PHASE 3: Per-Layer Correlation  (all 52 layers)")
print("=" * 100)

col_w = 12
print(f"\n  {'Layer':<14s} {'baseline':>{col_w}s} {'zero_own':>{col_w}s}"
      f" {'zero_comb':>{col_w}s} {'comb+pcDW':>{col_w}s} {'Dead':>5s}"
      f" {'Prop':>5s}")
print("  " + "-" * (14 + 4 * (col_w + 1) + 12))

for gn, pfx, k, s, pad, dw, relu in ALL_LAYERS:
    di = dead_info[gn]
    vals = []
    for mk, _ in MODES:
        c = results[mk]["corrs"].get(gn, float('nan'))
        vals.append(f"{c:>{col_w}.6f}")
    dead_s = f"{di['n_dead']:>5d}" if di['n_dead'] > 0 else f"{'':>5s}"
    # propagated count from the combined mode
    pk = results["zero_combined"]["per_layer_killed"].get(gn, (0, 0, 0))
    prop_s = f"{pk[2]:>5d}" if pk[2] > 0 else f"{'':>5s}"
    print(f"  {gn:<14s} {' '.join(vals)} {dead_s} {prop_s}")

# Avg-pool row
print(f"  {'avg_pool':<14s}", end="")
for mk, _ in MODES:
    print(f" {results[mk]['corr_avg']:>{col_w}.6f}", end="")
print()


# =========================================================================
# PHASE 4: Key-layer focus + predictions summary
# =========================================================================

print("\n" + "=" * 100)
print("PHASE 4: Key Layers + Prediction Summary")
print("=" * 100)

print(f"\n  {'Layer':<14s} {'baseline':>{col_w}s} {'zero_own':>{col_w}s}"
      f" {'zero_comb':>{col_w}s} {'comb+pcDW':>{col_w}s}")
print("  " + "-" * (14 + 4 * (col_w + 1)))

for gn in KEY_LAYERS:
    vals = []
    for mk, _ in MODES:
        c = results[mk]["corrs"].get(gn, float('nan'))
        vals.append(f"{c:>{col_w}.6f}")
    print(f"  {gn:<14s} {' '.join(vals)}")

print(f"  {'avg_pool':<14s}", end="")
for mk, _ in MODES:
    print(f" {results[mk]['corr_avg']:>{col_w}.6f}", end="")
print()

print(f"\n  {'Mode':<52s} {'Pred':>6s} {'Match':>6s}"
      f" {'AvgCorr':>9s} {'Killed':>7s}")
print("  " + "-" * 82)
for mk, ml in MODES:
    r = results[mk]
    m = "YES" if r["pred"] == float_pred else "NO"
    print(f"  {ml:<52s} {r['pred']:>6d} {m:>6s}"
          f" {r['corr_avg']:>9.6f} {r['total_killed']:>7d}")
print(f"\n  Float reference prediction: {float_pred}")


# =========================================================================
# PHASE 5: Threshold sweep for combined + per-channel DW
# =========================================================================

print("\n" + "=" * 100)
print("PHASE 5: Threshold Sweep  (zero_combined + per-channel DW)")
print("=" * 100)

thresholds = [0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
print(f"\n  {'Thr':>8s} {'Kill':>5s} {'Pred':>6s} {'Match':>6s}"
      f" {'AvgCorr':>9s} {'conv_1':>8s} {'dw_2':>8s} {'conv_3':>8s}"
      f" {'conv_52':>8s}")
print("  " + "-" * 75)

for thr in thresholds:
    r = run_int8_pass(thr, mode="zero_combined+pchdw")
    m = "YES" if r["pred"] == float_pred else "NO"
    c1   = r["corrs"].get("conv_1",    float('nan'))
    cdw2 = r["corrs"].get("conv_dw_2", float('nan'))
    c3   = r["corrs"].get("conv_3",    float('nan'))
    c52  = r["corrs"].get("conv_52",   float('nan'))
    print(f"  {thr:>8.3f} {r['total_killed']:>5d} {r['pred']:>6d} {m:>6s}"
          f" {r['corr_avg']:>9.6f} {c1:>8.4f} {cdw2:>8.4f}"
          f" {c3:>8.4f} {c52:>8.4f}")


# =========================================================================
# PHASE 6: Cross-layer propagation diagnostic
# =========================================================================

print("\n" + "=" * 100)
print("PHASE 6: Cross-Layer Propagation Diagnostic")
print("  Shows which layers gained additional zeroed channels from the")
print("  next DW layer's dead channels (channels zeroed ONLY due to")
print("  cross-layer propagation, not from own BN)")
print("=" * 100)

r_comb = results["zero_combined"]
r_own  = results["zero_own"]

print(f"\n  {'Layer':<14s} {'OwnDead':>8s} {'Propagated':>11s}"
      f" {'TotalKill':>10s} {'Corr(own)':>10s} {'Corr(comb)':>11s}"
      f" {'Delta':>8s}")
print("  " + "-" * 80)

for gn, pfx, k, s, pad, dw, relu in ALL_LAYERS:
    pk = r_comb["per_layer_killed"].get(gn, (0, 0, 0))
    n_kill, n_own, n_prop = pk

    corr_own  = r_own["corrs"].get(gn, float('nan'))
    corr_comb = r_comb["corrs"].get(gn, float('nan'))

    delta = corr_comb - corr_own if not (np.isnan(corr_comb)
                                          or np.isnan(corr_own)) else 0.0

    if n_prop > 0 or n_own > 0:
        d_str = f"{delta:>+8.4f}" if abs(delta) > 1e-8 else f"{'':>8s}"
        print(f"  {gn:<14s} {n_own:>8d} {n_prop:>11d}"
              f" {n_kill:>10d} {corr_own:>10.6f} {corr_comb:>11.6f}"
              f" {d_str}")

print(f"\n  Total own-dead zeroed:        {r_comb['total_own']}")
print(f"  Total propagated zeroed:      {r_comb['total_propagated']}")
print(f"  Total combined:               {r_comb['total_killed']}")


# =========================================================================
# FINAL ANALYSIS
# =========================================================================

print("\n" + "=" * 100)
print("FINAL ANALYSIS")
print("=" * 100)

best_mode = None
best_corr = -1.0
for mk, ml in MODES:
    r = results[mk]
    if r["corr_avg"] > best_corr:
        best_corr = r["corr_avg"]
        best_mode = ml

print(f"""
  Float prediction:               {float_pred}
  Best mode (by avg-pool corr):   {best_mode}
                                   (corr = {best_corr:.6f})
  Predictions per mode:""")
for mk, ml in MODES:
    r = results[mk]
    tag = " <-- MATCH" if r["pred"] == float_pred else ""
    print(f"    {ml:<52s}  pred={r['pred']}{tag}")

print(f"""
  Approach overview:
    1. Baseline: standard BN fold + per-tensor INT8. Dead BN channels with
       running_var ~ 0 cause weight explosion via 1/sqrt(var+eps), dominating
       per-tensor w_scale and crushing live channels' dynamic range.

    2. Zero own: each layer's dead BN channels get their folded weights and
       biases set to 0. Removes dead channels from w_scale. But the preceding
       layer may still output nonzero quantization noise on those channels.

    3. Zero combined (cross-layer): same as (2), but also zeros the preceding
       layer's output columns for dead channels in the next DW layer. This
       ensures exactly zero input to dead DW channels, preventing noise
       propagation. Only applied when next layer is DW (channel dims match).
       Key: conv_1 weight [27,32] has columns zeroed for conv_dw_2 dead ch;
       expand_1x1 outputs zeroed for each block's DW dead channels.

    4. Zero combined + per-channel DW: same as (3), but DW conv weights are
       quantized per-channel instead of per-tensor. Each DW channel gets its
       own w_scale, so no single outlier can dominate. Combined with zeroing,
       this should give maximum INT8 accuracy.
""")
