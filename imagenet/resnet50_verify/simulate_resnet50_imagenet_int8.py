#!/usr/bin/env python3
"""
Simulated INT8 inference for ResNet-50 ImageNet on Gemmini.

Exactly mimics what the Gemmini hardware does:
  1. INT8 matmul: acc = w_int8.T @ x_int8 + b_int32  (INT32 accumulator)
  2. Scale + requantize: y_int8 = clip(round(acc * output_scale), -128, 127)
  3. ResAdd: result = clip(round(skip * res_scale + main), -128, 127)
  4. ReLU: result = max(0, result) applied in CPU after resadd
  5. MaxPool after stem: 3x3 stride 2 pad 1

Input preprocessing (CORRECT - matches ImageNet training pipeline):
  x_float = (pixel/255 - mean) / std
  x_int8  = clip(round(x_float / FMAX * 127), -128, 127)
  x_scale_0 = FMAX / 127

  ImageNet mean = [0.485, 0.456, 0.406]  (RGB)
  ImageNet std  = [0.229, 0.224, 0.225]  (RGB)
  FMAX = 2.75  (clips all but ~0.01% of ImageNet pixels)

Usage:
    conda run -n ImageNet_stable python simulate_resnet50_imagenet_int8.py \\
        --imagenet-dir /path/to/ILSVRC2012_img_val \\
        --labels-file /path/to/val.txt \\
        --num-images 100
"""

import os
import sys
import argparse
import numpy as np
import torch

try:
    from transformers import ResNetForImageClassification
except ImportError:
    print("ERROR: pip install transformers")
    sys.exit(1)

MODEL_NAME  = "microsoft/resnet-50"
NUM_CLASSES = 1000

# ImageNet standard normalization
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float64)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float64)
FMAX = 2.75  # clip range for INT8 quantization (3-sigma covers >99.7%)

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess_imagenet_correct(img_bgr):
    """Resize + normalize + quantize to INT8 for Gemmini.

    Returns int8 array of shape (3, 224, 224) in CHW order (NCHW for batching).
    x_float = (pixel/255 - mean) / std  -> clip to FMAX -> INT8
    """
    import cv2
    img = cv2.resize(img_bgr, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_f = img.astype(np.float64) / 255.0
    img_norm = (img_f - IMAGENET_MEAN) / IMAGENET_STD        # HWC, float
    img_clipped = np.clip(img_norm / FMAX * 127, -128, 127)  # scale to INT8 range
    img_int8 = np.round(img_clipped).astype(np.int8)         # HWC int8
    return img_int8.transpose(2, 0, 1)                        # CHW int8


def preprocess_pixel_minus_128(img_bgr):
    """Simple pixel-128 preprocessing: resize + BGR2RGB + subtract 128.

    Returns int8 array of shape (3, 224, 224) in CHW order.
    """
    import cv2
    img = cv2.resize(img_bgr, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_int8 = (img.astype(np.int16) - 128).clip(-128, 127).astype(np.int8)
    return img_int8.transpose(2, 0, 1)                        # CHW int8


# ---------------------------------------------------------------------------
# BN folding & quantization helpers
# ---------------------------------------------------------------------------

def fold_bn(conv_weight, bn_weight, bn_bias, bn_mean, bn_var, eps=1e-5):
    inv_std = 1.0 / np.sqrt(bn_var + eps)
    scale   = bn_weight * inv_std
    shape   = [conv_weight.shape[0]] + [1] * (conv_weight.ndim - 1)
    w_folded = conv_weight * scale.reshape(shape)
    b_folded = bn_bias - bn_weight * bn_mean * inv_std
    return w_folded, b_folded


def get_conv_bn(sd, prefix):
    """Load conv+BN for HuggingFace microsoft/resnet-50 state dict."""
    conv_w = sd[f"{prefix}.convolution.weight"].float().numpy()
    gamma  = sd[f"{prefix}.normalization.weight"].float().numpy()
    beta   = sd[f"{prefix}.normalization.bias"].float().numpy()
    mean   = sd[f"{prefix}.normalization.running_mean"].float().numpy()
    var    = sd[f"{prefix}.normalization.running_var"].float().numpy()
    return conv_w, gamma, beta, mean, var


def reshape_conv_weight(w):
    """PyTorch [out_ch, C, kH, kW] -> Gemmini [kH*kW*C, out_ch]"""
    out_ch = w.shape[0]
    if w.ndim == 4:
        w = w.transpose(0, 2, 3, 1)  # [out_ch, kH, kW, C]
    return w.reshape(out_ch, -1).T   # [kH*kW*C, out_ch]


def quantize_weight_int8(w_float):
    w_abs_max = np.max(np.abs(w_float))
    if w_abs_max < 1e-10:
        return np.zeros_like(w_float, dtype=np.int8), 1e-10
    w_scale = w_abs_max / 127.0
    w_int   = np.clip(np.round(w_float / w_scale), -128, 127).astype(np.int8)
    return w_int, w_scale


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


def run_calibration(model, image_dir, num_images=100, percentile=99.99):
    """Run calibration images thru float model to get actual per-layer activation ranges.

    Uses the given percentile (default 99.99) instead of max-abs to avoid
    outliers dominating the range.  max-abs often produces ranges 2-5x larger
    than needed, wasting INT8 precision and tanking accuracy.
    """
    import torchvision.transforms as T
    from PIL import Image

    transform = T.Compose([
        T.Resize((224, 224)), T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN.tolist(), std=IMAGENET_STD.tolist()),
    ])
    # Collect per-layer absolute-value histograms (sampled)
    all_abs = {}  # gemmini_name -> list of sampled abs values

    def make_hook(gn, has_relu):
        def hook(module, inp, out):
            x = out.detach().float()
            if has_relu:
                x = torch.relu(x)
            vals = x.abs().flatten()
            # Subsample to keep memory bounded (~10K samples per image per layer)
            if vals.numel() > 10000:
                idx = torch.randperm(vals.numel())[:10000]
                vals = vals[idx]
            if gn not in all_abs:
                all_abs[gn] = []
            all_abs[gn].append(vals.numpy())
        return hook

    hooks = []
    for gn, pfx, k, s, p, relu in ALL_LAYERS:
        parts = pfx.split(".")
        mod = model
        for part in parts:
            mod = getattr(mod, part)
        h = mod.normalization.register_forward_hook(make_hook(gn, relu))
        hooks.append(h)

    files = sorted([f for f in os.listdir(image_dir)
                    if f.lower().endswith((".jpeg", ".jpg", ".png"))])[:num_images]
    with torch.no_grad():
        for fname in files:
            img = Image.open(os.path.join(image_dir, fname)).convert("RGB")
            model(transform(img).unsqueeze(0))
    for h in hooks:
        h.remove()

    # Compute percentile-based ranges
    captured = {}
    for gn in all_abs:
        combined = np.concatenate(all_abs[gn])
        captured[gn] = float(np.percentile(combined, percentile))
    return captured


def compute_output_scale(w_scale, x_scale, y_range):
    y_scale = y_range / 127.0
    raw     = (w_scale * x_scale) / y_scale
    if raw <= 0 or not np.isfinite(raw):
        raw = 1.0
    return raw


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


def maxpool2d_3x3_s2_p1(x_nchw):
    """3x3 MaxPool with stride 2 and padding 1 (matches CUDA default)."""
    N, C, H, W = x_nchw.shape
    x_pad = np.pad(x_nchw, ((0,0),(0,0),(1,1),(1,1)),
                   mode='constant', constant_values=-128)
    OH = (H + 2*1 - 3) // 2 + 1
    OW = (W + 2*1 - 3) // 2 + 1
    out = np.zeros((N, C, OH, OW), dtype=x_nchw.dtype)
    for i in range(OH):
        for j in range(OW):
            patch = x_pad[:, :, i*2:i*2+3, j*2:j*2+3]
            out[:, :, i, j] = patch.max(axis=(2, 3))
    return out


# ---------------------------------------------------------------------------
# Gemmini-equivalent ops
# ---------------------------------------------------------------------------

def gemmini_conv(x_nchw, w_int, b_int, output_scale, kernel, stride, padding):
    """Full im2col matmul for conv layers."""
    N, C, H, W = x_nchw.shape
    if kernel == 1 and stride == 1:
        # 1x1 conv without striding: reshape directly
        x_flat = x_nchw.reshape(N, C, H*W).transpose(0, 2, 1).reshape(N*H*W, C)
        out_h = out_w = H  # assume square
    elif kernel == 1 and stride > 1:
        x_s = x_nchw[:, :, ::stride, ::stride]
        _N, _C, _H, _W = x_s.shape
        x_flat = x_s.reshape(_N, _C, _H*_W).transpose(0, 2, 1).reshape(_N*_H*_W, _C)
        out_h = out_w = _H
    else:
        x_flat, out_h, out_w = im2col(x_nchw, kernel, stride, padding)

    # INT32 accumulation
    acc    = x_flat.astype(np.int32) @ w_int.astype(np.int32)
    acc   += b_int.reshape(1, -1).astype(np.int32)
    y_flat = np.clip(np.round(acc.astype(np.float64) * output_scale), -128, 127).astype(np.int8)

    out_ch = w_int.shape[1]
    return y_flat.reshape(N, out_h, out_w, out_ch).transpose(0, 3, 1, 2)


def gemmini_resadd(skip_int8, main_int8, res_scale):
    result = skip_int8.astype(np.float64) * res_scale + main_int8.astype(np.float64)
    return np.clip(np.round(result), -128, 127).astype(np.int8)


def cpu_relu(x_int8):
    return np.maximum(x_int8, np.int8(0))


# ---------------------------------------------------------------------------
# Layer mapping (gemmini_name, hf_prefix, kernel, stride, padding, relu)
# ---------------------------------------------------------------------------

ALL_LAYERS = [
    ("conv_1",  "resnet.embedder.embedder",                             7, 2, 3, True),

    ("conv_2",  "resnet.encoder.stages.0.layers.0.layer.0",            1, 1, 0, True),
    ("conv_3",  "resnet.encoder.stages.0.layers.0.layer.1",            3, 1, 1, True),
    ("conv_4",  "resnet.encoder.stages.0.layers.0.layer.2",            1, 1, 0, False),
    ("conv_5",  "resnet.encoder.stages.0.layers.0.shortcut",           1, 1, 0, False),

    ("conv_6",  "resnet.encoder.stages.0.layers.1.layer.0",            1, 1, 0, True),
    ("conv_7",  "resnet.encoder.stages.0.layers.1.layer.1",            3, 1, 1, True),
    ("conv_8",  "resnet.encoder.stages.0.layers.1.layer.2",            1, 1, 0, False),

    ("conv_9",  "resnet.encoder.stages.0.layers.2.layer.0",            1, 1, 0, True),
    ("conv_10", "resnet.encoder.stages.0.layers.2.layer.1",            3, 1, 1, True),
    ("conv_11", "resnet.encoder.stages.0.layers.2.layer.2",            1, 1, 0, False),

    ("conv_12", "resnet.encoder.stages.1.layers.0.layer.0",            1, 1, 0, True),
    ("conv_13", "resnet.encoder.stages.1.layers.0.layer.1",            3, 2, 1, True),
    ("conv_14", "resnet.encoder.stages.1.layers.0.layer.2",            1, 1, 0, False),
    ("conv_15", "resnet.encoder.stages.1.layers.0.shortcut",           1, 2, 0, False),

    ("conv_16", "resnet.encoder.stages.1.layers.1.layer.0",            1, 1, 0, True),
    ("conv_17", "resnet.encoder.stages.1.layers.1.layer.1",            3, 1, 1, True),
    ("conv_18", "resnet.encoder.stages.1.layers.1.layer.2",            1, 1, 0, False),

    ("conv_19", "resnet.encoder.stages.1.layers.2.layer.0",            1, 1, 0, True),
    ("conv_20", "resnet.encoder.stages.1.layers.2.layer.1",            3, 1, 1, True),
    ("conv_21", "resnet.encoder.stages.1.layers.2.layer.2",            1, 1, 0, False),

    ("conv_22", "resnet.encoder.stages.1.layers.3.layer.0",            1, 1, 0, True),
    ("conv_23", "resnet.encoder.stages.1.layers.3.layer.1",            3, 1, 1, True),
    ("conv_24", "resnet.encoder.stages.1.layers.3.layer.2",            1, 1, 0, False),

    ("conv_25", "resnet.encoder.stages.2.layers.0.layer.0",            1, 1, 0, True),
    ("conv_26", "resnet.encoder.stages.2.layers.0.layer.1",            3, 2, 1, True),
    ("conv_27", "resnet.encoder.stages.2.layers.0.layer.2",            1, 1, 0, False),
    ("conv_28", "resnet.encoder.stages.2.layers.0.shortcut",           1, 2, 0, False),

    ("conv_29", "resnet.encoder.stages.2.layers.1.layer.0",            1, 1, 0, True),
    ("conv_30", "resnet.encoder.stages.2.layers.1.layer.1",            3, 1, 1, True),
    ("conv_31", "resnet.encoder.stages.2.layers.1.layer.2",            1, 1, 0, False),

    ("conv_32", "resnet.encoder.stages.2.layers.2.layer.0",            1, 1, 0, True),
    ("conv_33", "resnet.encoder.stages.2.layers.2.layer.1",            3, 1, 1, True),
    ("conv_34", "resnet.encoder.stages.2.layers.2.layer.2",            1, 1, 0, False),

    ("conv_35", "resnet.encoder.stages.2.layers.3.layer.0",            1, 1, 0, True),
    ("conv_36", "resnet.encoder.stages.2.layers.3.layer.1",            3, 1, 1, True),
    ("conv_37", "resnet.encoder.stages.2.layers.3.layer.2",            1, 1, 0, False),

    ("conv_38", "resnet.encoder.stages.2.layers.4.layer.0",            1, 1, 0, True),
    ("conv_39", "resnet.encoder.stages.2.layers.4.layer.1",            3, 1, 1, True),
    ("conv_40", "resnet.encoder.stages.2.layers.4.layer.2",            1, 1, 0, False),

    ("conv_41", "resnet.encoder.stages.2.layers.5.layer.0",            1, 1, 0, True),
    ("conv_42", "resnet.encoder.stages.2.layers.5.layer.1",            3, 1, 1, True),
    ("conv_43", "resnet.encoder.stages.2.layers.5.layer.2",            1, 1, 0, False),

    ("conv_44", "resnet.encoder.stages.3.layers.0.layer.0",            1, 1, 0, True),
    ("conv_45", "resnet.encoder.stages.3.layers.0.layer.1",            3, 2, 1, True),
    ("conv_46", "resnet.encoder.stages.3.layers.0.layer.2",            1, 1, 0, False),
    ("conv_47", "resnet.encoder.stages.3.layers.0.shortcut",           1, 2, 0, False),

    ("conv_48", "resnet.encoder.stages.3.layers.1.layer.0",            1, 1, 0, True),
    ("conv_49", "resnet.encoder.stages.3.layers.1.layer.1",            3, 1, 1, True),
    ("conv_50", "resnet.encoder.stages.3.layers.1.layer.2",            1, 1, 0, False),

    ("conv_51", "resnet.encoder.stages.3.layers.2.layer.0",            1, 1, 0, True),
    ("conv_52", "resnet.encoder.stages.3.layers.2.layer.1",            3, 1, 1, True),
    ("conv_53", "resnet.encoder.stages.3.layers.2.layer.2",            1, 1, 0, False),
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

BLOCKS = [
    # Stage 0
    {"main": ["conv_2", "conv_3", "conv_4"], "proj": "conv_5",  "skip_src": "conv_5"},
    {"main": ["conv_6", "conv_7", "conv_8"], "proj": None,      "skip_src": "conv_4"},
    {"main": ["conv_9", "conv_10","conv_11"],"proj": None,      "skip_src": "conv_8"},
    # Stage 1
    {"main": ["conv_12","conv_13","conv_14"],"proj": "conv_15", "skip_src": "conv_15"},
    {"main": ["conv_16","conv_17","conv_18"],"proj": None,      "skip_src": "conv_14"},
    {"main": ["conv_19","conv_20","conv_21"],"proj": None,      "skip_src": "conv_18"},
    {"main": ["conv_22","conv_23","conv_24"],"proj": None,      "skip_src": "conv_21"},
    # Stage 2
    {"main": ["conv_25","conv_26","conv_27"],"proj": "conv_28", "skip_src": "conv_28"},
    {"main": ["conv_29","conv_30","conv_31"],"proj": None,      "skip_src": "conv_27"},
    {"main": ["conv_32","conv_33","conv_34"],"proj": None,      "skip_src": "conv_31"},
    {"main": ["conv_35","conv_36","conv_37"],"proj": None,      "skip_src": "conv_34"},
    {"main": ["conv_38","conv_39","conv_40"],"proj": None,      "skip_src": "conv_37"},
    {"main": ["conv_41","conv_42","conv_43"],"proj": None,      "skip_src": "conv_40"},
    # Stage 3
    {"main": ["conv_44","conv_45","conv_46"],"proj": "conv_47", "skip_src": "conv_47"},
    {"main": ["conv_48","conv_49","conv_50"],"proj": None,      "skip_src": "conv_46"},
    {"main": ["conv_51","conv_52","conv_53"],"proj": None,      "skip_src": "conv_50"},
]

SAVE_XSCALE_BEFORE = {
    "conv_2":  "conv_5",
    "conv_12": "conv_15",
    "conv_25": "conv_28",
    "conv_44": "conv_47",
}


# ---------------------------------------------------------------------------
# Load and quantize all layers
# ---------------------------------------------------------------------------

def build_int8_layers(sd, calibrated_ranges=None, pixel_minus_128=False):
    """Build quantized layer dict from state dict. Returns (layers, x_scale_0).

    If calibrated_ranges is provided (dict: gemmini_name -> max_abs_float),
    those values are used as y_range instead of BN 3-sigma estimates.

    If pixel_minus_128 is True, fold ImageNet normalization into conv_1 weights
    so that raw pixel-128 INT8 input works without repreprocessing.
    """
    if pixel_minus_128:
        x_scale_0 = 128.0 / 127.0
        print(f"  Input x_scale_0 = 128/127 = {x_scale_0:.6f}  (pixel-128 mode)")
    else:
        x_scale_0 = FMAX / 127.0
        print(f"  Input x_scale_0 = FMAX/127 = {x_scale_0:.6f}  (FMAX={FMAX})")

    # Pre-pass: collect y_range for all layers
    layer_y_range = {}
    for gn, pfx, k, s, p, relu in ALL_LAYERS:
        if calibrated_ranges and gn in calibrated_ranges:
            layer_y_range[gn] = max(calibrated_ranges[gn], 1.0)
        else:
            _, gamma, beta, _, _ = get_conv_bn(sd, pfx)
            layer_y_range[gn] = estimate_activation_range(gamma, beta, relu)

    # Main quantization pass
    layers        = {}
    x_scale       = x_scale_0
    shortcut_xs   = {}

    for gn, pfx, k, s, pad, relu in ALL_LAYERS:
        if gn in SAVE_XSCALE_BEFORE:
            shortcut_xs[SAVE_XSCALE_BEFORE[gn]] = x_scale
        eff_x_scale = shortcut_xs.pop(gn, x_scale)

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

        w_g    = reshape_conv_weight(w_folded)
        w_int, w_scale = quantize_weight_int8(w_g)
        b_int  = quantize_bias_int32(b_folded, w_scale * eff_x_scale)

        y_range      = layer_y_range[gn]
        output_scale = compute_output_scale(w_scale, eff_x_scale, y_range)

        res_scale = 1.0
        if gn in RESIDUAL_SKIP:
            skip_src  = RESIDUAL_SKIP[gn]
            res_scale = layer_y_range[skip_src] / y_range

        layers[gn] = dict(w_int=w_int, b_int=b_int,
                          output_scale=output_scale, res_scale=res_scale,
                          kernel=k, stride=s, padding=pad, relu=relu)

        print(f"    {gn:10s}  os={output_scale:.4e}  rs={res_scale:.4f}  "
              f"{'ReLU' if relu else 'Lin'}")
        # Don't propagate x_scale from shortcut/projection layers - the resadd
        # output is quantized at the main path's scale, not the shortcut's.
        is_shortcut = gn in set(SAVE_XSCALE_BEFORE.values())
        if not is_shortcut:
            x_scale = y_range / 127.0

    # FC
    fc_w_f  = sd["classifier.1.weight"].float().numpy()   # [1000, 2048]
    fc_b_f  = sd["classifier.1.bias"].float().numpy()     # [1000]
    fc_w_g  = fc_w_f.T                                    # [2048, 1000]
    fc_w_int, fc_w_scale = quantize_weight_int8(fc_w_g)
    fc_b_int = quantize_bias_int32(fc_b_f, fc_w_scale * x_scale)
    layers["fc_54"] = dict(w_int=fc_w_int, b_int=fc_b_int)
    print(f"    {'fc_54':10s}  w_scale={fc_w_scale:.6f}  x_scale={x_scale:.6f}")

    return layers, x_scale_0


# ---------------------------------------------------------------------------
# INT8 forward pass
# ---------------------------------------------------------------------------

def forward_int8(x_nchw, layers):
    """Run full INT8 ResNet-50 forward pass. Returns raw int32 logits."""
    L = layers

    # conv_1 + ReLU + MaxPool
    out = gemmini_conv(x_nchw, L["conv_1"]["w_int"], L["conv_1"]["b_int"],
                       L["conv_1"]["output_scale"], 7, 2, 3)
    out = cpu_relu(out)
    out = maxpool2d_3x3_s2_p1(out)   # [N, 64, 56, 56]

    stored     = {"conv_1": out}
    block_input = out

    for block in BLOCKS:
        x = block_input
        for ln in block["main"]:
            inf = L[ln]
            x   = gemmini_conv(x, inf["w_int"], inf["b_int"],
                               inf["output_scale"], inf["kernel"], inf["stride"], inf["padding"])
            if inf["relu"]:
                x = cpu_relu(x)
            stored[ln] = x

        main_out = stored[block["main"][-1]]

        if block["proj"] is not None:
            pn  = block["proj"]
            inf = L[pn]
            skip = gemmini_conv(block_input, inf["w_int"], inf["b_int"],
                                inf["output_scale"], inf["kernel"], inf["stride"], inf["padding"])
            if inf["relu"]:
                skip = cpu_relu(skip)
            stored[pn] = skip
        else:
            skip = stored[block["skip_src"]]

        main_last = block["main"][-1]
        result    = gemmini_resadd(skip, main_out, L[main_last]["res_scale"])
        result    = cpu_relu(result)
        stored[main_last] = result
        block_input = result

    # Global avg pool
    avg = block_input.astype(np.float64).mean(axis=(2, 3))   # [N, 2048]
    avg_int8 = np.clip(np.round(avg), -128, 127).astype(np.int8)

    # FC (no output_scale — raw INT32 logits)
    fc_acc  = avg_int8.astype(np.int32) @ L["fc_54"]["w_int"].astype(np.int32)
    fc_acc += L["fc_54"]["b_int"].reshape(1, -1).astype(np.int32)
    return fc_acc[0].astype(np.float64)


# ---------------------------------------------------------------------------
# Float reference forward pass
# ---------------------------------------------------------------------------

def forward_float(img_bgr, model):
    """Run original float model inference. Returns logits numpy array."""
    import cv2
    import torchvision.transforms as T
    img_rgb = cv2.cvtColor(cv2.resize(img_bgr, (224, 224)), cv2.COLOR_BGR2RGB)
    from PIL import Image
    pil_img = Image.fromarray(img_rgb)
    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # Use simple resize + normalize to match our preprocessing
    transform_simple = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    x = transform_simple(pil_img).unsqueeze(0)
    with torch.no_grad():
        logits = model(x).logits[0].float().numpy()
    return logits


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

IMAGE_BYTES = 224 * 224 * 3  # HWC int8, one image


def load_labels_plain(labels_file, num_images):
    """Load labels from a plain text file (one integer per line)."""
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
    """Load from 'filename label'-per-line format. Returns (paths, labels)."""
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
    """Read one HWC int8 image from binary file, return CHW int8."""
    raw = f.read(IMAGE_BYTES)
    if len(raw) < IMAGE_BYTES:
        return None
    img_hwc = np.frombuffer(raw, dtype=np.int8).reshape(224, 224, 3).copy()
    return img_hwc.transpose(2, 0, 1)  # CHW


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Simulate INT8 ResNet-50 ImageNet inference")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--imagenet-dir",
                       help="Directory containing validation JPEG images "
                            "(applies new normalization: (pixel/255-mean)/std)")
    group.add_argument("--images-bin",
                       help="Pre-processed binary file (int8 HWC, 224*224*3 per image). "
                            "Must be generated with the CURRENT prepare_imagenet.py "
                            "(FMAX=2.75 normalization). Old pixel-128 binaries give wrong results.")
    parser.add_argument("--labels-file", required=True,
                        help="Labels file: one integer per line (plain format) OR "
                             "'filename label' per line (imagenet format)")
    parser.add_argument("--num-images", type=int, default=100,
                        help="Number of images to evaluate (default: 100)")
    parser.add_argument("--check-float", action="store_true",
                        help="Also run float model and compare predictions "
                             "(only available with --imagenet-dir, needs original JPEGs)")
    parser.add_argument("--calibrate-dir", type=str, default=None,
                        help="Directory of JPEG images for calibration-based "
                             "activation ranges (highly recommended for accuracy)")
    parser.add_argument("--num-calibrate", type=int, default=100,
                        help="Number of calibration images (default 100)")
    parser.add_argument("--pixel-minus-128", action="store_true",
                        help="Use pixel-128 input mode (fold normalization into "
                             "conv_1 weights instead of expecting FMAX-normalized input)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-image predictions")
    args = parser.parse_args()

    if args.check_float and args.images_bin:
        print("WARNING: --check-float requires original JPEG files (--imagenet-dir), ignoring")
        args.check_float = False

    print("=" * 70)
    print("Simulated INT8 Inference for ResNet-50 ImageNet (Gemmini)")
    print("=" * 70)
    if args.pixel_minus_128:
        print(f"\nPreprocessing: pixel-128 (normalization folded into conv_1)")
        print(f"x_scale_0 = 128/127 = {128/127:.6f}")
    else:
        print(f"\nPreprocessing: (pixel/255 - mean)/std / FMAX * 127  ->  INT8")
        print(f"FMAX={FMAX},  ImageNet mean={IMAGENET_MEAN.tolist()},  std={IMAGENET_STD.tolist()}")
        print(f"x_scale_0 = FMAX/127 = {FMAX/127:.6f}")

    # Load model
    print(f"\nLoading {MODEL_NAME}...")
    model = ResNetForImageClassification.from_pretrained(MODEL_NAME)
    model.eval()
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    # Quantize all layers
    calibrated_ranges = None
    if args.calibrate_dir:
        print(f"\nCalibration pass: measuring actual activation ranges...")
        calibrated_ranges = run_calibration(model, args.calibrate_dir, args.num_calibrate)
        print(f"  Calibrated {len(calibrated_ranges)} layers")
    else:
        print("\n  NOTE: Using BN-estimated ranges. Use --calibrate-dir for better accuracy.")
    print("\nQuantizing layers...")
    layers, x_scale_0 = build_int8_layers(sd, calibrated_ranges, pixel_minus_128=args.pixel_minus_128)

    # Load labels
    print(f"\nLoading labels from {args.labels_file}...")
    # Auto-detect format: if first non-empty line splits into 2+ parts, it's imagenet format
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
        # If --imagenet-dir provided with plain labels, build sorted paths from directory
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
            # Read pre-processed int8 from binary file
            img_int8 = read_image_from_bin(bin_f)
            if img_int8 is None:
                print(f"  WARNING: binary file ended at image {i}, stopping")
                break
            img_bgr = None
        else:
            # Load JPEG and apply current preprocessing
            import cv2
            path = img_paths[i] if img_paths else None
            if path is None:
                print(f"  WARNING: no path for image {i}, skipping")
                continue
            img_bgr = cv2.imread(path)
            if img_bgr is None:
                print(f"  WARNING: cannot read {path}, skipping")
                continue
            if args.pixel_minus_128:
                img_int8 = preprocess_pixel_minus_128(img_bgr)
            else:
                img_int8 = preprocess_imagenet_correct(img_bgr)  # [3, 224, 224]

        x_in = img_int8[np.newaxis, :, :, :]  # [1, 3, 224, 224]

        # INT8 forward
        logits_int8 = forward_int8(x_in, layers)
        pred         = int(np.argmax(logits_int8))
        top5         = set(np.argsort(logits_int8)[-5:].tolist())

        if pred == label:
            top1_correct += 1
        if label in top5:
            top5_correct += 1

        # Float reference (JPEG mode only)
        if args.check_float and img_bgr is not None:
            logits_f = forward_float(img_bgr, model)
            pred_f   = int(np.argmax(logits_f))
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

    done = min(i + 1, N)
    print(f"\n{'='*70}")
    print(f"INT8 Simulation Results ({done} images):")
    print(f"  Top-1: {top1_correct}/{done} ({100*top1_correct/done:.1f}%)")
    print(f"  Top-5: {top5_correct}/{done} ({100*top5_correct/done:.1f}%)")
    if args.check_float:
        print(f"\nFloat Reference Results:")
        print(f"  Top-1: {float_top1_correct}/{done} ({100*float_top1_correct/done:.1f}%)")
        print(f"  INT8 vs Float gap: {(float_top1_correct-top1_correct)/done*100:.1f}%")
    print(f"{'='*70}")

    if top1_correct / done < 0.5:
        print("\n*** TOP-1 < 50% — quantization pipeline may be broken ***")
        if args.images_bin:
            print("NOTE: If this binary was created with the OLD 'pixel-128' preprocessing,")
            print("      regenerate it with the updated prepare_imagenet.py (FMAX=2.75 normalization).")
        print("Possible causes:")
        print("  1. Preprocessing mismatch (old binary vs new params)")
        print("  2. Wrong FMAX or x_scale_0")
        print("  3. BN folding error")
        print("  4. Residual skip scale error")
    else:
        print(f"\n*** OK — INT8 accuracy {100*top1_correct/done:.1f}% ***")
        print("If FPGA accuracy is different, the problem is in C code / hardware.")


if __name__ == "__main__":
    main()
