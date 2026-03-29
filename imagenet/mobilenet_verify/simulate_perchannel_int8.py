#!/usr/bin/env python3
"""
Per-channel INT8 simulation for MobileNetV2 ImageNet.

Same model/preprocessing/activation quantization as simulate_mobilenet_imagenet_int8.py,
but uses per-output-channel weight quantization instead of per-tensor.

This shows the accuracy ceiling if Gemmini supported per-channel weight scales.

Usage:
    conda run -n ImageNet python simulate_perchannel_int8.py \
        --imagenet-dir /home/hansa/Downloads/Images200 \
        --labels-file /home/hansa/Downloads/imagenet_val_50000_clipped_labels.txt \
        --pixel-minus-128 --num-images 200 \
        --calibrate-dir /home/hansa/Downloads/Images200 --num-calibrate 200 \
        --check-float
"""

import os
import sys
import argparse
import numpy as np
import torch

try:
    from transformers import MobileNetV2ForImageClassification
except ImportError:
    print("ERROR: pip install transformers")
    sys.exit(1)

MODEL_NAME  = "google/mobilenet_v2_1.0_224"
BN_EPS      = 0.001
IMAGENET_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float64)
IMAGENET_STD  = np.array([0.5, 0.5, 0.5], dtype=np.float64)

# ---------------------------------------------------------------------------
# Layer architecture  (same as existing simulation)
# ---------------------------------------------------------------------------

ALL_LAYERS = [
    ("conv_1",    "mobilenet_v2.conv_stem.first_conv", 3, 2, 1, False, True),
    ("conv_dw_2", "mobilenet_v2.conv_stem.conv_3x3",   3, 1, 1,  True, True),
    ("conv_3",    "mobilenet_v2.conv_stem.reduce_1x1", 1, 1, 0, False, False),
]
_idx = 4
_dw_strides = {0: 2, 2: 2, 5: 2, 12: 2}
for _hi in range(16):
    _pfx = f"mobilenet_v2.layer.{_hi}"
    _ds  = _dw_strides.get(_hi, 1)
    ALL_LAYERS.append((f"conv_{_idx}",    f"{_pfx}.expand_1x1", 1, 1, 0, False, True));  _idx += 1
    ALL_LAYERS.append((f"conv_dw_{_idx}", f"{_pfx}.conv_3x3",   3, _ds, 1, True, True)); _idx += 1
    ALL_LAYERS.append((f"conv_{_idx}",    f"{_pfx}.reduce_1x1", 1, 1, 0, False, False)); _idx += 1
ALL_LAYERS.append(("conv_52", "mobilenet_v2.conv_1x1", 1, 1, 0, False, True))

RESIDUAL_SKIP = {
    "conv_9":  "conv_6",  "conv_15": "conv_12", "conv_18": "conv_15",
    "conv_24": "conv_21", "conv_27": "conv_24",  "conv_30": "conv_27",
    "conv_36": "conv_33", "conv_39": "conv_36",  "conv_45": "conv_42",
    "conv_48": "conv_45",
}

# ---------------------------------------------------------------------------
# BN folding helpers (unchanged from existing simulation)
# ---------------------------------------------------------------------------

def fold_bn(conv_w, gamma, beta, bn_m, bn_v, eps=BN_EPS):
    inv_std = 1.0 / np.sqrt(bn_v + eps)
    scale   = gamma * inv_std
    shape   = [conv_w.shape[0]] + [1] * (conv_w.ndim - 1)
    return conv_w * scale.reshape(shape), beta - gamma * bn_m * inv_std


def get_conv_bn(sd, prefix):
    conv_w = sd[f"{prefix}.convolution.weight"].float().numpy().astype(np.float64)
    gamma  = sd[f"{prefix}.normalization.weight"].float().numpy().astype(np.float64)
    beta   = sd[f"{prefix}.normalization.bias"].float().numpy().astype(np.float64)
    mean   = sd[f"{prefix}.normalization.running_mean"].float().numpy().astype(np.float64)
    var    = sd[f"{prefix}.normalization.running_var"].float().numpy().astype(np.float64)
    return conv_w, gamma, beta, mean, var


def reshape_conv_weight(w):
    """[out_ch, in_ch, kH, kW] -> [patch_size, out_ch]  (Gemmini im2col order)."""
    out_ch = w.shape[0]
    if w.ndim == 4:
        w = w.transpose(0, 2, 3, 1)
    return w.reshape(out_ch, -1).T


def reshape_dw_weight(w):
    assert w.shape[1] == 1
    return w.squeeze(1)  # [C, 3, 3]


# ---------------------------------------------------------------------------
# Per-channel weight quantization  (KEY CHANGE vs per-tensor simulation)
# ---------------------------------------------------------------------------

def qw_perchannel_conv(w_flat):
    """w_flat shape [patch_size, out_ch].
    Returns (w_int8, ws) where ws is [out_ch] per-channel scales.
    """
    out_ch = w_flat.shape[1]
    ws = np.zeros(out_ch, dtype=np.float64)
    for j in range(out_ch):
        mx = float(np.max(np.abs(w_flat[:, j])))
        ws[j] = max(mx / 127.0, 1e-10)
    w_int = np.clip(np.round(w_flat / ws[np.newaxis, :]), -128, 127).astype(np.int8)
    return w_int, ws


def qw_perchannel_dw(w_chw):
    """w_chw shape [C, 3, 3].  Per-filter (= per-channel) scale."""
    C = w_chw.shape[0]
    ws = np.zeros(C, dtype=np.float64)
    for c in range(C):
        mx = float(np.max(np.abs(w_chw[c])))
        ws[c] = max(mx / 127.0, 1e-10)
    w_int = np.clip(np.round(w_chw / ws.reshape(-1, 1, 1)), -128, 127).astype(np.int8)
    return w_int, ws


def qb_perchannel(b, ws_arr, x_scale):
    """Bias quantized with per-channel combined scale."""
    combined = ws_arr * x_scale  # [out_ch]
    return np.clip(np.round(b / combined), -(2**31), 2**31 - 1).astype(np.int32)


def os_perchannel(ws_arr, x_scale, y_range):
    """Per-channel output scale: os[j] = (ws_j * x_scale) / (y_range/127)."""
    y_scale = y_range / 127.0
    raw = (ws_arr * x_scale) / y_scale
    raw = np.where(np.isfinite(raw) & (raw > 0), raw, 1.0)
    return raw.astype(np.float64)


# ---------------------------------------------------------------------------
# Calibration (identical to existing simulation)
# ---------------------------------------------------------------------------

def run_calibration(model, image_dir, num_images=100, percentile=99.99):
    import cv2
    all_abs = {}

    def make_hook(gn, has_relu):
        def hook(module, inp, out):
            x = out.detach().float()
            if has_relu:
                x = torch.clamp(torch.relu(x), max=6.0)
            vals = x.abs().flatten()
            if vals.numel() > 10000:
                idx = torch.randperm(vals.numel())[:10000]
                vals = vals[idx]
            all_abs.setdefault(gn, []).append(vals.numpy())
        return hook

    hooks = []
    for gn, pfx, k, s, p, dw, relu in ALL_LAYERS:
        mod = model
        for part in pfx.split("."):
            mod = getattr(mod, part)
        h = mod.normalization.register_forward_hook(make_hook(gn, relu))
        hooks.append(h)

    files = sorted([f for f in os.listdir(image_dir)
                    if f.lower().endswith((".jpeg", ".jpg", ".png"))])[:num_images]
    print(f"  Running {len(files)} calibration images...")
    with torch.no_grad():
        for i, fname in enumerate(files):
            img = cv2.imread(os.path.join(image_dir, fname))
            if img is None: continue
            img = cv2.resize(img, (224, 224))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
            for c in range(3):
                img[:, :, c] = (img[:, :, c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
            model(torch.tensor(img.transpose(2, 0, 1)[np.newaxis], dtype=torch.float32))
            if (i + 1) % 50 == 0:
                print(f"    [{i+1}/{len(files)}]")

    for h in hooks: h.remove()
    captured = {gn: float(np.percentile(np.concatenate(v), percentile))
                for gn, v in all_abs.items()}
    print(f"  Calibrated {len(captured)} layers (percentile={percentile})")
    return captured


# ---------------------------------------------------------------------------
# Build per-channel INT8 layer dict
# ---------------------------------------------------------------------------

def build_layers_perchannel(sd, num_classes, pixel_minus_128=False, calibrated=None):
    if calibrated is None:
        calibrated = {}

    x_scale = 128.0 / 127.0 if pixel_minus_128 else 1.0
    print(f"  x_scale_0 = {x_scale:.6f}  ({'pixel-128' if pixel_minus_128 else 'normalized'})")

    # Pre-pass: y_range per layer
    layer_y_range = {}
    for gn, pfx, k, s, p, dw, relu in ALL_LAYERS:
        if gn in calibrated:
            layer_y_range[gn] = max(calibrated[gn], 1.0)
        else:
            _, gamma, beta, _, _ = get_conv_bn(sd, pfx)
            if relu:
                layer_y_range[gn] = 6.0
            else:
                layer_y_range[gn] = max(float(np.max(np.abs(beta) + 6.0 * np.abs(gamma))), 1.0)

    layers = {}
    for gn, pfx, k, s, pad, dw, relu in ALL_LAYERS:
        conv_w, gamma, beta, bn_m, bn_v = get_conv_bn(sd, pfx)
        w_f, b_f = fold_bn(conv_w, gamma, beta, bn_m, bn_v)

        # Fold normalization into conv_1 for pixel-128 mode
        if gn == "conv_1" and pixel_minus_128:
            norm_offset = (128.0 / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
            for c in range(3):
                b_f += w_f[:, c, :, :].sum(axis=(1, 2)) * norm_offset[c]
            for c in range(3):
                w_f[:, c, :, :] /= (255.0 * IMAGENET_STD[c])

        # Dead channel fix for DW+ReLU layers
        if dw and relu:
            dead_threshold = 0.01
            n_dead = 0
            for c in range(w_f.shape[0]):
                if bn_v[c] < dead_threshold:
                    w_f[c] = 0.0
                    b_f[c] = float(np.clip(b_f[c], 0.0, 6.0))
                    n_dead += 1
            if n_dead > 0:
                print(f"    [{gn}] {n_dead} dead channels fixed")

        yr = layer_y_range[gn]
        res_scale = 1.0
        if gn in RESIDUAL_SKIP:
            skip_src = RESIDUAL_SKIP[gn]
            res_scale = layer_y_range[skip_src] / yr

        if dw:
            wg = reshape_dw_weight(w_f)     # [C, 3, 3]
            w_int, ws = qw_perchannel_dw(wg)
            b_int = qb_perchannel(b_f, ws, x_scale)
            output_scale = os_perchannel(ws, x_scale, yr)  # [C]

            # Dead channel fix: per-channel scale is 1e-10 for zeroed channels,
            # causing b_int overflow (b_f / (1e-10 * x_scale) >> INT32_MAX) and
            # output_scale ≈ 0 so output → wrong constant.
            # Fix: bypass scale chain; encode constant output directly.
            for c in range(wg.shape[0]):
                if bn_v[c] < 0.01:  # dead channel
                    # b_f[c] = clamp(b_folded, 0, 6) — the constant float output
                    # Encode as: b_int[c] = round(b_f[c] * 127/yr), os[c] = 1.0
                    # Then acc = 0 * inputs + b_int[c]; out = clip(round(acc * 1.0))
                    b_int[c] = int(np.clip(np.round(b_f[c] * 127.0 / yr), -(2**31), 2**31-1))
                    output_scale[c] = 1.0
        else:
            wg = reshape_conv_weight(w_f)   # [patch_size, out_ch]
            w_int, ws = qw_perchannel_conv(wg)
            b_int = qb_perchannel(b_f, ws, x_scale)
            output_scale = os_perchannel(ws, x_scale, yr)  # [out_ch]

        os_mean = float(np.mean(output_scale))
        rs_str = f" rs={res_scale:.4f}" if gn in RESIDUAL_SKIP else ""
        print(f"    {gn:12s}  os_mean={os_mean:.4e}  {'ReLU' if relu else 'Lin '}"
              f"  {'DW' if dw else 'CV'}{rs_str}")

        layers[gn] = dict(w_int=w_int, b_int=b_int, output_scale=output_scale,
                          res_scale=res_scale, kernel=k, stride=s, padding=pad,
                          dw=dw, relu=relu)
        x_scale = yr / 127.0

    # FC layer — per-row (per output class) quantization
    fc_w_f = sd["classifier.weight"].float().numpy().astype(np.float64)  # [cls, 1280]
    fc_b_f = sd["classifier.bias"].float().numpy().astype(np.float64)
    if num_classes == 1001:
        fc_w_f = fc_w_f[1:, :]
        fc_b_f = fc_b_f[1:]

    # Per-channel: each output class gets its own weight scale
    out_cls = fc_w_f.shape[0]
    ws_fc = np.zeros(out_cls, dtype=np.float64)
    for j in range(out_cls):
        mx = float(np.max(np.abs(fc_w_f[j])))
        ws_fc[j] = max(mx / 127.0, 1e-10)
    fc_w_int = np.clip(np.round(fc_w_f / ws_fc[:, np.newaxis]), -128, 127).astype(np.int8)
    fc_b_int = np.clip(np.round(fc_b_f / (ws_fc * x_scale)), -(2**31), 2**31 - 1).astype(np.int32)

    # FC y_range (same formula as existing simulation)
    input_float_range = x_scale * 127.0
    fc_y_range = max(
        float(np.max(np.abs(fc_b_f))
              + np.std(fc_w_f) * np.sqrt(float(fc_w_f.shape[1])) * input_float_range * 3),
        5.0
    )
    fc_output_scale = (ws_fc * x_scale) / (fc_y_range / 127.0)  # [out_cls]
    fc_output_scale = np.where(np.isfinite(fc_output_scale) & (fc_output_scale > 0),
                               fc_output_scale, 1.0)

    layers["fc_53"] = dict(w_int=fc_w_int, b_int=fc_b_int, output_scale=fc_output_scale)
    print(f"    {'fc_53':12s}  os_mean={float(np.mean(fc_output_scale)):.4e}  y_range={fc_y_range:.2f}")

    return layers, 128.0 / 127.0 if pixel_minus_128 else 1.0


# ---------------------------------------------------------------------------
# im2col
# ---------------------------------------------------------------------------

def im2col(x_nchw, kernel, stride, pad):
    N, C, H, W = x_nchw.shape
    kH = kW = kernel
    OH = (H + 2*pad - kH) // stride + 1
    OW = (W + 2*pad - kW) // stride + 1
    if pad > 0:
        x_nchw = np.pad(x_nchw, ((0,0),(0,0),(pad,pad),(pad,pad)), constant_values=0)
    patches = np.zeros((N, OH, OW, kH*kW*C), dtype=x_nchw.dtype)
    for i in range(OH):
        for j in range(OW):
            p = x_nchw[:, :, i*stride:i*stride+kH, j*stride:j*stride+kW]
            patches[:, i, j, :] = p.transpose(0, 2, 3, 1).reshape(N, -1)
    return patches.reshape(N*OH*OW, kH*kW*C), OH, OW


# ---------------------------------------------------------------------------
# Per-channel conv ops
# ---------------------------------------------------------------------------

def conv_perchannel(x_nchw, w_int, b_int, output_scale, kernel, stride, pad):
    """output_scale shape: [out_ch] — per-channel scales."""
    N, C, H, W = x_nchw.shape
    if kernel == 1 and stride == 1:
        x_flat = x_nchw.reshape(N, C, H*W).transpose(0, 2, 1).reshape(N*H*W, C)
        oh = ow = H
    elif kernel == 1:
        xs = x_nchw[:, :, ::stride, ::stride]
        _N, _C, _H, _W = xs.shape
        x_flat = xs.reshape(_N, _C, _H*_W).transpose(0, 2, 1).reshape(_N*_H*_W, _C)
        oh = ow = _H
    else:
        x_flat, oh, ow = im2col(x_nchw, kernel, stride, pad)

    acc = x_flat.astype(np.int32) @ w_int.astype(np.int32)     # [N*oh*ow, out_ch]
    acc += b_int.reshape(1, -1).astype(np.int32)
    # Per-channel scale: output_scale[out_ch]
    y = np.clip(np.round(acc.astype(np.float64) * output_scale[np.newaxis, :]),
                -128, 127).astype(np.int8)
    return y.reshape(N, oh, ow, w_int.shape[1]).transpose(0, 3, 1, 2)


def conv_dw_perchannel(x_nchw, w_int, b_int, output_scale, stride, pad):
    """output_scale shape: [C] — per-channel scales for depthwise."""
    N, C, H, W = x_nchw.shape
    kH = kW = 3
    OH = (H + 2*pad - kH) // stride + 1
    OW = (W + 2*pad - kW) // stride + 1
    x_pad = np.pad(x_nchw, ((0,0),(0,0),(pad,pad),(pad,pad)), constant_values=0) if pad > 0 else x_nchw
    acc = np.zeros((N, C, OH, OW), dtype=np.int64)
    x32 = x_pad.astype(np.int32)
    w32 = w_int.astype(np.int32)
    for ki in range(kH):
        for kj in range(kW):
            rows = np.arange(OH) * stride + ki
            cols = np.arange(OW) * stride + kj
            acc += x32[:, :, rows[:, None], cols[None, :]] * w32[:, ki, kj].reshape(1, C, 1, 1)
    acc += b_int.reshape(1, C, 1, 1).astype(np.int64)
    # Per-channel scale: output_scale[C] -> [1, C, 1, 1]
    out = np.clip(np.round(acc.astype(np.float64) * output_scale.reshape(1, -1, 1, 1)),
                  -128, 127).astype(np.int8)
    return out


def resadd(skip, main, res_scale):
    return np.clip(np.round(skip.astype(np.float64) * res_scale + main.astype(np.float64)),
                   -128, 127).astype(np.int8)


# ---------------------------------------------------------------------------
# Forward pass (per-channel INT8)
# ---------------------------------------------------------------------------

def forward_perchannel(x_nchw, layers):
    stored = {}
    out = x_nchw
    for gn, pfx, k, s, pad, dw, relu in ALL_LAYERS:
        inf = layers[gn]
        if dw:
            out = conv_dw_perchannel(out, inf["w_int"], inf["b_int"],
                                     inf["output_scale"], inf["stride"], inf["padding"])
        else:
            out = conv_perchannel(out, inf["w_int"], inf["b_int"],
                                  inf["output_scale"], inf["kernel"], inf["stride"], inf["padding"])
        if relu:
            out = np.maximum(out, np.int8(0))
        stored[gn] = out
        if gn in RESIDUAL_SKIP:
            skip = stored[RESIDUAL_SKIP[gn]]
            out = resadd(skip, out, inf["res_scale"])
            stored[gn] = out

    avg = out.astype(np.float64).mean(axis=(2, 3))
    avg_int8 = np.clip(np.round(avg), -128, 127).astype(np.int8)

    fc = layers["fc_53"]
    fc_acc = fc["w_int"].astype(np.int32) @ avg_int8[0].astype(np.int32)
    fc_acc += fc["b_int"].astype(np.int32)
    return np.clip(np.round(fc_acc.astype(np.float64) * fc["output_scale"]), -128, 127)


# ---------------------------------------------------------------------------
# Float reference (unchanged from existing simulation)
# ---------------------------------------------------------------------------

def forward_float(img_bgr, model, num_classes):
    import cv2
    img_rgb = cv2.cvtColor(cv2.resize(img_bgr, (224, 224)), cv2.COLOR_BGR2RGB)
    img_f = img_rgb.astype(np.float32) / 127.5 - 1.0
    x = torch.tensor(img_f.transpose(2, 0, 1)).unsqueeze(0)
    with torch.no_grad():
        logits = model(x).logits[0].float().numpy()
    if num_classes == 1001:
        logits = logits[1:]
    return logits


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Per-channel INT8 MobileNetV2 simulation")
    parser.add_argument("--imagenet-dir", required=True)
    parser.add_argument("--labels-file",  required=True)
    parser.add_argument("--num-images",   type=int, default=200)
    parser.add_argument("--pixel-minus-128", action="store_true")
    parser.add_argument("--calibrate-dir",   type=str, default=None)
    parser.add_argument("--num-calibrate",   type=int, default=200)
    parser.add_argument("--check-float",     action="store_true")
    args = parser.parse_args()

    import cv2

    print("=" * 70)
    print("Per-Channel INT8 Simulation — MobileNetV2 ImageNet (Gemmini-sim)")
    print("=" * 70)

    print(f"\nLoading {MODEL_NAME}...")
    model = MobileNetV2ForImageClassification.from_pretrained(MODEL_NAME)
    model.eval()
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    num_classes = sd["classifier.weight"].shape[0]
    print(f"Model has {num_classes} output classes")

    print("\nQuantizing layers (per-channel weights)...")
    calibrated = {}
    if args.calibrate_dir:
        print("Running calibration pass...")
        calibrated = run_calibration(model, args.calibrate_dir, args.num_calibrate)
    layers, _ = build_layers_perchannel(sd, num_classes,
                                        pixel_minus_128=args.pixel_minus_128,
                                        calibrated=calibrated)

    # Load labels
    files = sorted([f for f in os.listdir(args.imagenet_dir)
                    if f.lower().endswith((".jpeg", ".jpg", ".png"))])
    with open(args.labels_file) as f:
        labels = [int(l.strip()) for l in f if l.strip().lstrip('-').isdigit()]

    N = min(len(files), len(labels), args.num_images)
    print(f"\nLoaded {N} images and labels")

    print(f"\nRunning per-channel INT8 inference on {N} images...")
    top1 = top5 = float_top1 = 0

    for i in range(N):
        img_bgr = cv2.imread(os.path.join(args.imagenet_dir, files[i]))
        if img_bgr is None: continue
        label = labels[i]

        img = cv2.resize(img_bgr, (224, 224))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if args.pixel_minus_128:
            x_int8 = (img_rgb.astype(np.int16) - 128).clip(-128, 127).astype(np.int8)
        else:
            img_f = img_rgb.astype(np.float64) / 255.0
            for c in range(3):
                img_f[:, :, c] = (img_f[:, :, c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
            fmax = max(abs(-IMAGENET_MEAN[c]/IMAGENET_STD[c]) for c in range(3))
            fmax = max(fmax, max(abs((1-IMAGENET_MEAN[c])/IMAGENET_STD[c]) for c in range(3)))
            x_int8 = np.clip(np.round(img_f / fmax * 127), -128, 127).astype(np.int8)

        logits = forward_perchannel(x_int8.transpose(2, 0, 1)[np.newaxis], layers)
        pred = int(np.argmax(logits))
        top5_set = set(int(x) for x in np.argsort(logits)[-5:])

        if pred == label:  top1 += 1
        if label in top5_set: top5 += 1

        if args.check_float:
            lf = forward_float(img_bgr, model, num_classes)
            if int(np.argmax(lf)) == label:
                float_top1 += 1

        if i < 5:
            match = "OK" if pred == label else "WRONG"
            print(f"  [{i+1:4d}] label={label:4d}  pred={pred:4d}  "
                  f"logit_range=[{logits.min():.0f},{logits.max():.0f}]  {match}")

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{N}]  Top-1: {top1}/{i+1} ({100*top1/(i+1):.1f}%)")

    print(f"\n{'='*70}")
    print(f"Per-Channel INT8 Results ({N} images):")
    print(f"  Top-1: {top1}/{N} ({100*top1/max(N,1):.1f}%)")
    print(f"  Top-5: {top5}/{N} ({100*top5/max(N,1):.1f}%)")
    if args.check_float:
        print(f"\nFloat Reference:")
        print(f"  Top-1: {float_top1}/{N} ({100*float_top1/max(N,1):.1f}%)")
        print(f"  Per-channel INT8 vs Float gap: {(float_top1-top1)/max(N,1)*100:.1f}%")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
