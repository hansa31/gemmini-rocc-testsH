#!/usr/bin/env python3
"""
Simulated INT8 inference for MobileNetV2 ImageNet on Gemmini.

Exactly mimics what the Gemmini hardware does:
  1. Regular conv (im2col + matmul): acc = w_int8.T @ x_int8 + b_int32 (INT32)
  2. Depthwise conv: per-channel 3x3 convolution
  3. Scale + requantize: y_int8 = clip(round(acc * output_scale), -128, 127)
  4. ResAdd: result = clip(round(skip * res_scale + main), -128, 127)
     * NO ReLU after resadd in MobileNetV2 (all resadds have relu=false)
  5. Global average pooling -> FC (1280 -> 1000)

Architecture: MobileNetV2
  - conv_1: 3x3, stride 2, ReLU (3->32)
  - conv_dw_2: 3x3 depthwise, stride 1, ReLU (32)
  - conv_3: 1x1, NO_ACTIVATION (32->16)
  - 16 inverted residual blocks (conv_4..conv_51): expand+dw+reduce
  - conv_52: 1x1, ReLU (320->1280)
  - Global avg pool + FC (1280->1000)

Usage:
    cd imagenet/mobilenet_verify
    conda run -n ImageNet python simulate_mobilenet_imagenet_int8.py \\
        --imagenet-dir /path/to/ILSVRC2012_img_val \\
        --labels-file /path/to/val.txt \\
        --num-images 200

    # With pixel-128 quantised binary:
    conda run -n ImageNet python simulate_mobilenet_imagenet_int8.py \\
        --images-bin /path/to/imagenet_val_pixel128.bin \\
        --labels-file /path/to/labels.txt \\
        --pixel-minus-128 --num-images 200
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
NUM_CLASSES = 1000
BN_EPS = 0.001  # MobileNetV2 uses eps=0.001

# MobileNetV2 uses [-1, 1] normalization (mean=0.5, std=0.5),
# NOT the standard ImageNet mean/std used by ResNet.
# AutoImageProcessor: image_mean=[0.5,0.5,0.5], image_std=[0.5,0.5,0.5]
IMAGENET_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float64)
IMAGENET_STD  = np.array([0.5, 0.5, 0.5], dtype=np.float64)

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess_imagenet_fmax(img_bgr, fmax):
    """Resize + normalize + quantize using FMAX-based scaling."""
    import cv2
    img = cv2.resize(img_bgr, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_f = img.astype(np.float64) / 255.0
    img_norm = (img_f - IMAGENET_MEAN) / IMAGENET_STD
    img_clipped = np.clip(img_norm / fmax * 127, -128, 127)
    img_int8 = np.round(img_clipped).astype(np.int8)
    return img_int8.transpose(2, 0, 1)  # CHW


def preprocess_pixel_minus_128(img_bgr):
    """Simple pixel-128 preprocessing: resize + BGR2RGB + subtract 128."""
    import cv2
    img = cv2.resize(img_bgr, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_int8 = (img.astype(np.int16) - 128).clip(-128, 127).astype(np.int8)
    return img_int8.transpose(2, 0, 1)  # CHW


# ---------------------------------------------------------------------------
# BN folding & quantization helpers
# ---------------------------------------------------------------------------

def fold_bn(conv_weight, bn_weight, bn_bias, bn_mean, bn_var, eps=BN_EPS):
    inv_std = 1.0 / np.sqrt(bn_var + eps)
    scale   = bn_weight * inv_std
    shape   = [conv_weight.shape[0]] + [1] * (conv_weight.ndim - 1)
    w_folded = conv_weight * scale.reshape(shape)
    b_folded = bn_bias - bn_weight * bn_mean * inv_std
    return w_folded, b_folded


def get_conv_bn(sd, prefix):
    """Load conv+BN from HuggingFace MobileNetV2 state dict."""
    conv_w = sd[f"{prefix}.convolution.weight"].float().numpy().astype(np.float64)
    gamma  = sd[f"{prefix}.normalization.weight"].float().numpy().astype(np.float64)
    beta   = sd[f"{prefix}.normalization.bias"].float().numpy().astype(np.float64)
    mean   = sd[f"{prefix}.normalization.running_mean"].float().numpy().astype(np.float64)
    var    = sd[f"{prefix}.normalization.running_var"].float().numpy().astype(np.float64)
    return conv_w, gamma, beta, mean, var


def reshape_conv_weight(w):
    """Regular conv: [out_ch, in_ch, kH, kW] -> [patch_size, out_ch].
    Gemmini expects patch dimension in (kH, kW, in_ch) order.
    """
    out_ch = w.shape[0]
    if w.ndim == 4:
        w = w.transpose(0, 2, 3, 1)  # [out_ch, kH, kW, in_ch]
    return w.reshape(out_ch, -1).T


def reshape_dw_weight(w):
    """Depthwise conv: [channels, 1, kH, kW] -> [channels, kH, kW]."""
    assert w.shape[1] == 1
    return w.squeeze(1)


def quantize_weight_int8(w_float):
    w_abs_max = np.max(np.abs(w_float))
    if w_abs_max < 1e-10:
        return np.zeros_like(w_float, dtype=np.int8), 1e-10
    w_scale = float(w_abs_max) / 127.0
    w_int   = np.clip(np.round(w_float / w_scale), -128, 127).astype(np.int8)
    return w_int, w_scale


def quantize_bias_int32(b_float, combined_scale):
    if combined_scale < 1e-10:
        return np.zeros_like(b_float, dtype=np.int32)
    return np.clip(np.round(b_float / combined_scale), -(2**31), 2**31 - 1).astype(np.int32)


def estimate_activation_range(bn_gamma, bn_beta, has_relu):
    """Estimate activation range from BN statistics.

    For ReLU6 layers (MobileNetV2 uses ReLU6 for all relu activations):
      - y_range = 6.0 (the ReLU6 clamp ensures output is in [0, 6])
      - In INT8, clipping to 127 with y_scale = 6/127 implements ReLU6

    For linear (no activation) layers (reduce/project):
      - Use per-channel max of |beta| + 6*|gamma| (6-sigma for wider coverage)
    """
    if has_relu:
        y_range = 6.0  # ReLU6 clamp
    else:
        y_range = max(float(np.max(np.abs(bn_beta) + 6.0 * np.abs(bn_gamma))), 1.0)
    return y_range


def run_calibration(model, image_dir, num_images=100, percentile=99.99):
    """Run calibration images through the float model to measure actual
    per-layer activation ranges using a percentile.
    """
    import cv2

    # Build lookup for which layers have relu
    layer_relu = {gn: relu for gn, pfx, k, s, p, dw, relu in ALL_LAYERS}

    all_abs = {}

    def make_hook(gemmini_name, has_relu):
        def hook(module, inp, out):
            x = out.detach().float()
            if has_relu:
                x = torch.clamp(torch.relu(x), max=6.0)  # ReLU6
            vals = x.abs().flatten()
            if vals.numel() > 10000:
                idx = torch.randperm(vals.numel())[:10000]
                vals = vals[idx]
            if gemmini_name not in all_abs:
                all_abs[gemmini_name] = []
            all_abs[gemmini_name].append(vals.numpy())
        return hook

    hooks = []
    for gn, pfx, k, s, p, dw, relu in ALL_LAYERS:
        parts = pfx.split(".")
        mod = model
        for part in parts:
            mod = getattr(mod, part)
        bn_mod = mod.normalization
        h = bn_mod.register_forward_hook(make_hook(gn, relu))
        hooks.append(h)

    files = sorted([
        f for f in os.listdir(image_dir)
        if f.lower().endswith((".jpeg", ".jpg", ".png"))
    ])[:num_images]
    print(f"  Running {len(files)} calibration images...")

    mean_arr = IMAGENET_MEAN
    std_arr = IMAGENET_STD

    with torch.no_grad():
        for i, fname in enumerate(files):
            img_bgr = cv2.imread(os.path.join(image_dir, fname))
            if img_bgr is None:
                continue
            img = cv2.resize(img_bgr, (224, 224))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_f = img_rgb.astype(np.float64) / 255.0
            for c in range(3):
                img_f[:, :, c] = (img_f[:, :, c] - mean_arr[c]) / std_arr[c]
            x = torch.tensor(img_f.transpose(2, 0, 1)[np.newaxis], dtype=torch.float32)
            model(x)
            if (i + 1) % 50 == 0:
                print(f"    [{i+1}/{len(files)}]")

    for h in hooks:
        h.remove()

    captured = {}
    for gn in all_abs:
        combined = np.concatenate(all_abs[gn])
        captured[gn] = float(np.percentile(combined, percentile))

    print(f"  Calibrated {len(captured)} layers (percentile={percentile})")
    return captured


def compute_output_scale(w_scale, x_scale, y_range):
    y_scale = y_range / 127.0
    raw     = (w_scale * x_scale) / y_scale
    if raw <= 0 or not np.isfinite(raw):
        raw = 1.0
    return float(raw)


# ---------------------------------------------------------------------------
# Layer architecture
# (gemmini_name, hf_prefix, kernel, stride, padding, is_dw, has_relu)
# ---------------------------------------------------------------------------

ALL_LAYERS = [
    ("conv_1",     "mobilenet_v2.conv_stem.first_conv", 3, 2, 1, False, True),
    ("conv_dw_2",  "mobilenet_v2.conv_stem.conv_3x3",   3, 1, 1,  True, True),
    ("conv_3",     "mobilenet_v2.conv_stem.reduce_1x1", 1, 1, 0, False, False),
]

# Add 16 inverted residual blocks
_idx = 4
for _hf_idx in range(16):
    _pfx = f"mobilenet_v2.layer.{_hf_idx}"
    # Stride for depthwise layer — lookup from architecture
    _dw_strides = {0: 2, 2: 2, 5: 2, 12: 2}  # hf_layer_idx -> dw_stride
    _dw_stride = _dw_strides.get(_hf_idx, 1)
    ALL_LAYERS.append((f"conv_{_idx}",     f"{_pfx}.expand_1x1",  1, 1, 0, False, True))
    _idx += 1
    ALL_LAYERS.append((f"conv_dw_{_idx}",  f"{_pfx}.conv_3x3",    3, _dw_stride, 1,  True, True))
    _idx += 1
    ALL_LAYERS.append((f"conv_{_idx}",     f"{_pfx}.reduce_1x1",  1, 1, 0, False, False))
    _idx += 1

ALL_LAYERS.append(("conv_52", "mobilenet_v2.conv_1x1", 1, 1, 0, False, True))


# Residual skip connections (target -> source)
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
# im2col
# ---------------------------------------------------------------------------

def im2col(input_nchw, kernel_size, stride, padding):
    N, C, H, W = input_nchw.shape
    kH = kW = kernel_size
    OH = (H + 2 * padding - kH) // stride + 1
    OW = (W + 2 * padding - kW) // stride + 1
    if padding > 0:
        input_nchw = np.pad(input_nchw,
                            ((0,0),(0,0),(padding,padding),(padding,padding)),
                            mode='constant', constant_values=0)
    patches = np.zeros((N, OH, OW, kH * kW * C), dtype=input_nchw.dtype)
    for i in range(OH):
        for j in range(OW):
            patch = input_nchw[:, :, i*stride:i*stride+kH, j*stride:j*stride+kW]
            patches[:, i, j, :] = patch.transpose(0, 2, 3, 1).reshape(N, -1)
    return patches.reshape(N * OH * OW, kH * kW * C), OH, OW


# ---------------------------------------------------------------------------
# Gemmini-equivalent ops
# ---------------------------------------------------------------------------

def gemmini_conv(x_nchw, w_int, b_int, output_scale, kernel, stride, padding):
    """Regular conv via im2col + matmul."""
    N, C, H, W = x_nchw.shape
    if kernel == 1 and stride == 1:
        x_flat = x_nchw.reshape(N, C, H*W).transpose(0, 2, 1).reshape(N*H*W, C)
        out_h = out_w = H
    elif kernel == 1 and stride > 1:
        x_s = x_nchw[:, :, ::stride, ::stride]
        _N, _C, _H, _W = x_s.shape
        x_flat = x_s.reshape(_N, _C, _H*_W).transpose(0, 2, 1).reshape(_N*_H*_W, _C)
        out_h = out_w = _H
    else:
        x_flat, out_h, out_w = im2col(x_nchw, kernel, stride, padding)

    # INT32 accumulation: x_flat[N*OH*OW, patch_size] @ w_int[patch_size, out_ch]
    acc = x_flat.astype(np.int32) @ w_int.astype(np.int32)
    acc += b_int.reshape(1, -1).astype(np.int32)
    y_flat = np.clip(np.round(acc.astype(np.float64) * output_scale), -128, 127).astype(np.int8)

    out_ch = w_int.shape[1]
    return y_flat.reshape(N, out_h, out_w, out_ch).transpose(0, 3, 1, 2)


def gemmini_conv_dw(x_nchw, w_int, b_int, output_scale, stride, padding):
    """Depthwise conv: per-channel 3x3 convolution (vectorized).
    w_int: [channels, 3, 3] (int8)
    b_int: [channels] (int32)
    """
    N, C, H, W = x_nchw.shape
    assert w_int.shape[0] == C, f"DW weight channels {w_int.shape[0]} != input {C}"
    kH = kW = 3

    OH = (H + 2 * padding - kH) // stride + 1
    OW = (W + 2 * padding - kW) // stride + 1

    if padding > 0:
        x_pad = np.pad(x_nchw, ((0,0),(0,0),(padding,padding),(padding,padding)),
                        mode='constant', constant_values=0)
    else:
        x_pad = x_nchw

    # Vectorized: extract all patches at once
    # acc shape: [N, C, OH, OW] in int32
    acc = np.zeros((N, C, OH, OW), dtype=np.int64)
    x32 = x_pad.astype(np.int32)
    w32 = w_int.astype(np.int32)  # [C, 3, 3]

    for ki in range(kH):
        for kj in range(kW):
            # x32[:, :, ki::stride, kj::stride] doesn't work for arbitrary offsets
            # Instead extract at each output position
            rows = np.arange(OH) * stride + ki
            cols = np.arange(OW) * stride + kj
            patch = x32[:, :, rows[:, None], cols[None, :]]  # [N, C, OH, OW]
            acc += patch * w32[:, ki, kj].reshape(1, C, 1, 1)

    acc += b_int.reshape(1, C, 1, 1).astype(np.int64)
    out = np.clip(np.round(acc.astype(np.float64) * output_scale), -128, 127).astype(np.int8)
    return out


def gemmini_resadd(skip_int8, main_int8, res_scale):
    """ResAdd: result = clip(round(skip * res_scale + main), -128, 127).
    NO ReLU applied (MobileNetV2 resadds have relu=false).
    """
    result = skip_int8.astype(np.float64) * res_scale + main_int8.astype(np.float64)
    return np.clip(np.round(result), -128, 127).astype(np.int8)


def cpu_relu(x_int8):
    return np.maximum(x_int8, np.int8(0))


# ---------------------------------------------------------------------------
# Load and quantize all layers
# ---------------------------------------------------------------------------

def build_int8_layers(sd, num_classes, pixel_minus_128=False, calibrated=None, dead_zero=True):
    """Build quantized layer dict from state dict. Returns (layers, x_scale_0)."""
    if calibrated is None:
        calibrated = {}
    if pixel_minus_128:
        x_scale_0 = 128.0 / 127.0
        print(f"  Input x_scale_0 = 128/127 = {x_scale_0:.6f}  (pixel-128 mode)")
    else:
        maxvals = []
        for i in range(3):
            maxvals.append(abs(-IMAGENET_MEAN[i] / IMAGENET_STD[i]))
            maxvals.append(abs((1.0 - IMAGENET_MEAN[i]) / IMAGENET_STD[i]))
        fmax = max(maxvals)
        x_scale_0 = fmax / 127.0
        print(f"  Input x_scale_0 = FMAX/127 = {x_scale_0:.6f}  (FMAX={fmax:.4f})")

    # Pre-pass: collect y_range for all conv layers
    layer_y_range = {}
    for gn, pfx, k, s, p, dw, relu in ALL_LAYERS:
        if gn in calibrated:
            layer_y_range[gn] = max(calibrated[gn], 1.0)
        else:
            _, gamma, beta, _, _ = get_conv_bn(sd, pfx)
            layer_y_range[gn] = estimate_activation_range(gamma, beta, relu)

    # Main quantization pass
    layers = {}
    x_scale = x_scale_0

    for gn, pfx, k, s, pad, dw, relu in ALL_LAYERS:
        conv_w, gamma, beta, bn_m, bn_v = get_conv_bn(sd, pfx)
        w_folded, b_folded = fold_bn(conv_w, gamma, beta, bn_m, bn_v)

        # For pixel-128 mode: fold ImageNet normalization into conv_1
        if gn == "conv_1" and pixel_minus_128:
            mean_arr = np.array(IMAGENET_MEAN, dtype=np.float64)
            std_arr  = np.array(IMAGENET_STD, dtype=np.float64)
            norm_offset = (128.0 / 255.0 - mean_arr) / std_arr
            for c in range(3):
                b_folded += w_folded[:, c, :, :].sum(axis=(1, 2)) * norm_offset[c]
            for c in range(3):
                w_folded[:, c, :, :] /= (255.0 * std_arr[c])
            print(f"    [conv_1] Folded ImageNet normalization (pixel-128 mode)")

        # For DW layers with ReLU: fix dead channels (var ≈ 0).
        # These channels have huge BN-folded weights that dominate per-tensor scale.
        # Zero their weights to fix the scale, but clamp bias to ReLU6 output
        # so the dead channel's constant contribution is preserved (not lost).
        if dw and relu and dead_zero:
            dead_threshold = 0.01  # channels with var < threshold
            n_dead = 0
            for c in range(w_folded.shape[0]):
                if bn_v[c] < dead_threshold:
                    w_folded[c] = 0.0
                    b_folded[c] = float(np.clip(b_folded[c], 0.0, 6.0))
                    n_dead += 1
            if n_dead > 0:
                print(f"    [{gn}] Fixed {n_dead} dead channels (zero weights, clamp bias to ReLU6)")

        if dw:
            w_g = reshape_dw_weight(w_folded)  # [ch, 3, 3]
        else:
            w_g = reshape_conv_weight(w_folded)  # [patch_size, out_ch]

        w_int, w_scale = quantize_weight_int8(w_g)
        b_int = quantize_bias_int32(b_folded, w_scale * x_scale)

        y_range = layer_y_range[gn]
        output_scale = compute_output_scale(w_scale, x_scale, y_range)

        res_scale = 1.0
        if gn in RESIDUAL_SKIP:
            skip_src = RESIDUAL_SKIP[gn]
            res_scale = layer_y_range[skip_src] / y_range

        layers[gn] = dict(w_int=w_int, b_int=b_int,
                          output_scale=output_scale, res_scale=res_scale,
                          kernel=k, stride=s, padding=pad, dw=dw, relu=relu)

        rs_str = f" rs={res_scale:.4f}" if gn in RESIDUAL_SKIP else ""
        print(f"    {gn:12s}  os={output_scale:.4e}  {'ReLU' if relu else 'Lin '}"
              f"  {'DW' if dw else 'CV'}{rs_str}")

        # Propagate x_scale
        x_scale = y_range / 127.0

    # FC layer
    fc_w_f = sd["classifier.weight"].float().numpy().astype(np.float64)  # [num_classes, 1280]
    fc_b_f = sd["classifier.bias"].float().numpy().astype(np.float64)    # [num_classes]

    if num_classes == 1001:
        fc_w_f = fc_w_f[1:, :]  # skip background class 0
        fc_b_f = fc_b_f[1:]

    # fc_53_w[1000][1280] — MobileNetV2 FC is NOT transposed (same as PyTorch layout)
    w_int, w_scale = quantize_weight_int8(fc_w_f)
    b_int = quantize_bias_int32(fc_b_f, w_scale * x_scale)

    # FC y_range: use float input magnitude (x_scale * 127 = previous layer's y_range),
    # not the per-INT8-unit x_scale, to avoid output saturation.
    input_float_range = x_scale * 127.0
    fc_y_range = max(
        float(np.max(np.abs(fc_b_f))
              + np.std(fc_w_f) * np.sqrt(float(fc_w_f.shape[1])) * input_float_range * 3),
        5.0
    )
    fc_output_scale = compute_output_scale(w_scale, x_scale, fc_y_range)

    layers["fc_53"] = dict(w_int=w_int, b_int=b_int, output_scale=fc_output_scale)
    print(f"    {'fc_53':12s}  w_scale={w_scale:.6f}  x_scale={x_scale:.6f}"
          f"  y_range={fc_y_range:.2f}  output_scale={fc_output_scale:.4e}")

    return layers, x_scale_0


# ---------------------------------------------------------------------------
# INT8 forward pass
# ---------------------------------------------------------------------------

def forward_int8(x_nchw, layers):
    """Run full INT8 MobileNetV2 forward pass. Returns INT8 logits [1000] (matches hardware)."""
    L = layers
    stored = {}

    # Process all conv layers sequentially
    out = x_nchw
    for gn, pfx, k, s, pad, dw, relu in ALL_LAYERS:
        inf = L[gn]
        if dw:
            out = gemmini_conv_dw(out, inf["w_int"], inf["b_int"],
                                   inf["output_scale"], inf["stride"], inf["padding"])
        else:
            out = gemmini_conv(out, inf["w_int"], inf["b_int"],
                               inf["output_scale"], inf["kernel"], inf["stride"], inf["padding"])

        if relu:
            out = cpu_relu(out)

        stored[gn] = out

        # ResAdd: add skip connection AFTER the reduce layer
        if gn in RESIDUAL_SKIP:
            skip_src = RESIDUAL_SKIP[gn]
            skip = stored[skip_src]
            out = gemmini_resadd(skip, out, inf["res_scale"])
            # NO ReLU after resadd in MobileNetV2
            stored[gn] = out  # update with resadd result

    # Global average pooling: [N=1, C=1280, H=7, W=7] -> [1280]
    # C code: average[1280][4] with avg per channel
    avg = out.astype(np.float64).mean(axis=(2, 3))  # [N, 1280]
    avg_int8 = np.clip(np.round(avg), -128, 127).astype(np.int8)  # [N, 1280]

    # FC: fc_53_w[1000][1280]  @  avg[1280] -> logits[1000]
    # Hardware applies output_scale and clips to INT8 (NO_ACTIVATION but still requantized).
    fc = L["fc_53"]
    fc_acc = fc["w_int"].astype(np.int32) @ avg_int8[0].astype(np.int32)  # [1000]
    fc_acc += fc["b_int"].astype(np.int32)
    # Apply output_scale and clip to INT8 — matches Gemmini hardware exactly
    fc_int8 = np.clip(np.round(fc_acc.astype(np.float64) * fc["output_scale"]), -128, 127)
    return fc_int8


# ---------------------------------------------------------------------------
# Float reference forward pass
# ---------------------------------------------------------------------------

def forward_float(img_bgr, model, num_classes):
    """Run float model inference. Returns logits numpy array [1000].
    Preprocessing matches FPGA binary: direct resize to 224x224 (no crop),
    normalized with MobileNetV2 mean=[0.5,0.5,0.5] std=[0.5,0.5,0.5].
    """
    import cv2

    img_rgb = cv2.cvtColor(cv2.resize(img_bgr, (224, 224)), cv2.COLOR_BGR2RGB)
    # Normalize: (pixel/255 - 0.5) / 0.5  =  pixel/127.5 - 1.0  (range [-1, 1])
    img_f = img_rgb.astype(np.float32) / 127.5 - 1.0
    # [H, W, C] -> [1, C, H, W]
    x = torch.tensor(img_f.transpose(2, 0, 1)).unsqueeze(0)
    with torch.no_grad():
        logits = model(x).logits[0].float().numpy()

    if num_classes == 1001:
        logits = logits[1:]  # skip background class

    return logits


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

IMAGE_BYTES = 224 * 224 * 3


def load_labels_plain(labels_file, num_images):
    labels = []
    with open(labels_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            labels.append(int(line))
            if len(labels) >= num_images:
                break
    return labels


def load_labels_imagenet(labels_file, num_images):
    paths, labels = [], []
    with open(labels_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            paths.append(parts[0])
            labels.append(int(parts[1]))
            if len(paths) >= num_images:
                break
    return paths, labels


def read_image_from_bin(f):
    raw = f.read(IMAGE_BYTES)
    if len(raw) < IMAGE_BYTES:
        return None
    img_hwc = np.frombuffer(raw, dtype=np.int8).reshape(224, 224, 3).copy()
    return img_hwc.transpose(2, 0, 1)  # CHW


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Simulate INT8 MobileNetV2 ImageNet inference")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--imagenet-dir",
                       help="Directory containing validation JPEG images")
    group.add_argument("--images-bin",
                       help="Pre-processed binary file (int8 HWC, 224*224*3 per image)")
    parser.add_argument("--labels-file", required=True,
                        help="Labels file: one integer per line OR 'filename label' per line")
    parser.add_argument("--num-images", type=int, default=100)
    parser.add_argument("--check-float", action="store_true",
                        help="Also run float model and compare")
    parser.add_argument("--pixel-minus-128", action="store_true",
                        help="Use pixel-128 input mode (fold normalization into conv_1)")
    parser.add_argument("--calibrate-dir", type=str, default=None,
                        help="Directory of calibration JPEG images for activation range measurement")
    parser.add_argument("--num-calibrate", type=int, default=100,
                        help="Number of calibration images to use (default: 100)")
    parser.add_argument("--no-dead-zero", action="store_true",
                        help="Disable dead channel zeroing for DW layers")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.check_float and args.images_bin:
        print("WARNING: --check-float requires --imagenet-dir, ignoring")
        args.check_float = False

    print("=" * 70)
    print("Simulated INT8 Inference for MobileNetV2 ImageNet (Gemmini)")
    print("=" * 70)
    if args.pixel_minus_128:
        print(f"\nPreprocessing: pixel-128 (normalization folded into conv_1)")
        print(f"x_scale_0 = 128/127 = {128/127:.6f}")
    else:
        maxvals = []
        for i in range(3):
            maxvals.append(abs(-IMAGENET_MEAN[i] / IMAGENET_STD[i]))
            maxvals.append(abs((1.0 - IMAGENET_MEAN[i]) / IMAGENET_STD[i]))
        fmax = max(maxvals)
        print(f"\nPreprocessing: (pixel/255 - mean)/std / FMAX * 127 -> INT8")
        print(f"FMAX={fmax:.4f}, x_scale_0 = {fmax/127:.6f}")

    # Load model
    print(f"\nLoading {MODEL_NAME}...")
    model = MobileNetV2ForImageClassification.from_pretrained(MODEL_NAME)
    model.eval()
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    num_classes = sd["classifier.weight"].shape[0]
    print(f"Model has {num_classes} output classes")

    # Quantize all layers
    print("\nQuantizing layers...")
    calibrated = {}
    if args.calibrate_dir:
        print("Running calibration pass...")
        calibrated = run_calibration(model, args.calibrate_dir, args.num_calibrate)
    layers, x_scale_0 = build_int8_layers(sd, num_classes,
                                           pixel_minus_128=args.pixel_minus_128,
                                           calibrated=calibrated,
                                           dead_zero=not args.no_dead_zero)

    # Load labels
    print(f"\nLoading labels from {args.labels_file}...")
    img_paths = None
    with open(args.labels_file, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split()
                is_imagenet_fmt = len(parts) >= 2 and not parts[0].lstrip('-').isdigit()
                break

    if is_imagenet_fmt and args.imagenet_dir:
        img_paths_rel, img_labels = load_labels_imagenet(args.labels_file, args.num_images)
        img_paths = [os.path.join(args.imagenet_dir, p) for p in img_paths_rel]
    else:
        img_labels = load_labels_plain(args.labels_file, args.num_images)
        if args.imagenet_dir:
            image_files = sorted([
                f for f in os.listdir(args.imagenet_dir)
                if f.lower().endswith((".jpeg", ".jpg", ".png"))
            ])
            img_paths = [os.path.join(args.imagenet_dir, f) for f in image_files]

    N = min(len(img_labels), args.num_images)
    img_labels = img_labels[:N]
    print(f"Loaded {N} labels")

    # Evaluate
    print(f"\nRunning INT8 inference on {N} images...")
    top1_correct = top5_correct = 0
    float_top1_correct = 0

    bin_f = open(args.images_bin, "rb") if args.images_bin else None

    for i, label in enumerate(img_labels):
        if bin_f is not None:
            img_int8 = read_image_from_bin(bin_f)
            if img_int8 is None:
                print(f"  WARNING: binary file ended at image {i}, stopping")
                break
            img_bgr = None
        else:
            import cv2
            path = img_paths[i] if img_paths else None
            if path is None or not os.path.exists(path):
                continue
            img_bgr = cv2.imread(path)
            if img_bgr is None:
                continue
            if args.pixel_minus_128:
                img_int8 = preprocess_pixel_minus_128(img_bgr)
            else:
                maxvals = []
                for ci in range(3):
                    maxvals.append(abs(-IMAGENET_MEAN[ci] / IMAGENET_STD[ci]))
                    maxvals.append(abs((1.0 - IMAGENET_MEAN[ci]) / IMAGENET_STD[ci]))
                fmax = max(maxvals)
                img_int8 = preprocess_imagenet_fmax(img_bgr, fmax)

        x_in = img_int8[np.newaxis, :, :, :]  # [1, 3, 224, 224]

        logits_int8 = forward_int8(x_in, layers)
        pred = int(np.argmax(logits_int8))
        top5 = set(np.argsort(logits_int8)[-5:].tolist())

        if pred == label:
            top1_correct += 1
        if label in top5:
            top5_correct += 1

        if args.check_float and img_bgr is not None:
            logits_f = forward_float(img_bgr, model, num_classes)
            pred_f = int(np.argmax(logits_f))
            if pred_f == label:
                float_top1_correct += 1

        if args.verbose or i < 5:
            match = "OK" if pred == label else "WRONG"
            print(f"  [{i+1:4d}] label={label:4d}  int8_pred={pred:4d}  "
                  f"logit_range=[{logits_int8.min():.0f},{logits_int8.max():.0f}]  {match}")

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{N}]  Top-1: {top1_correct}/{i+1} "
                  f"({100*top1_correct/(i+1):.1f}%)")

    if bin_f is not None:
        bin_f.close()

    done = min(i + 1, N) if N > 0 else 0
    print(f"\n{'='*70}")
    print(f"INT8 Simulation Results ({done} images):")
    print(f"  Top-1: {top1_correct}/{done} ({100*top1_correct/max(done,1):.1f}%)")
    print(f"  Top-5: {top5_correct}/{done} ({100*top5_correct/max(done,1):.1f}%)")
    if args.check_float:
        print(f"\nFloat Reference Results:")
        print(f"  Top-1: {float_top1_correct}/{done} ({100*float_top1_correct/max(done,1):.1f}%)")
        print(f"  INT8 vs Float gap: {(float_top1_correct-top1_correct)/max(done,1)*100:.1f}%")
    print(f"{'='*70}")

    if done > 0 and top1_correct / done < 0.3:
        print("\n*** TOP-1 < 30% — quantization pipeline may be broken ***")


if __name__ == "__main__":
    main()
