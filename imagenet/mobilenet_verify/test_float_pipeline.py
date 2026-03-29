#!/usr/bin/env python3
"""
Float-precision forward pass through our MobileNetV2 layer topology.

Purpose: verify that our forward pass topology (layer ordering, residual
connections, ReLU vs linear, depthwise handling, global avg pool, FC) is
correct by running ALL computations in float64 -- no INT8 quantization at
all.  If the float-through simulation matches the HuggingFace model
predictions, the topology is correct and any accuracy loss is purely from
INT8 quantization.

Architecture (mirrors the INT8 simulation in simulate_mobilenet_imagenet_int8.py):
  - conv_1:    3x3, stride 2, ReLU     (3  -> 32)
  - conv_dw_2: 3x3 depthwise, stride 1, ReLU (32)
  - conv_3:    1x1, NO_ACTIVATION       (32 -> 16)
  - 16 inverted residual blocks (conv_4 .. conv_51): expand + dw + reduce
  - conv_52:   1x1, ReLU                (320 -> 1280)
  - Global avg pool + FC                (1280 -> 1000)

Key design choices matching the Gemmini C code:
  - Regular convs: im2col with (kH, kW, in_ch) patch order + matmul
  - Depthwise convs: per-channel 3x3 convolution
  - ReLU layers: clamp to [0, inf)  (NOT ReLU6 -- the Gemmini C code uses RELU)
  - Linear layers (reduce/project): no activation at all
  - ResAdd: skip + main  (float, no scaling, no clipping)
  - Global avg pool: mean over spatial dims
  - FC: matmul + bias

Usage:
    conda run -n ImageNet python test_float_pipeline.py
"""

import sys
import numpy as np
import torch

try:
    from transformers import MobileNetV2ForImageClassification
except ImportError:
    print("ERROR: pip install transformers")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_NAME  = "google/mobilenet_v2_1.0_224"
BN_EPS      = 0.001   # MobileNetV2 uses eps=0.001
IMAGE_PATH  = "/home/hansa/Downloads/Images200/ILSVRC2012_val_00000001.JPEG"
TRUE_LABEL  = 65

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float64)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float64)

# ---------------------------------------------------------------------------
# Layer architecture  (identical to simulate_mobilenet_imagenet_int8.py)
# (gemmini_name, hf_prefix, kernel, stride, padding, is_dw, has_relu)
# ---------------------------------------------------------------------------

ALL_LAYERS = [
    ("conv_1",     "mobilenet_v2.conv_stem.first_conv", 3, 2, 1, False, True),
    ("conv_dw_2",  "mobilenet_v2.conv_stem.conv_3x3",   3, 1, 1,  True, True),
    ("conv_3",     "mobilenet_v2.conv_stem.reduce_1x1", 1, 1, 0, False, False),
]

_idx = 4
for _hf_idx in range(16):
    _pfx = f"mobilenet_v2.layer.{_hf_idx}"
    _dw_strides = {0: 2, 2: 2, 5: 2, 12: 2}
    _dw_stride = _dw_strides.get(_hf_idx, 1)
    ALL_LAYERS.append((f"conv_{_idx}",    f"{_pfx}.expand_1x1", 1, 1, 0, False, True))
    _idx += 1
    ALL_LAYERS.append((f"conv_dw_{_idx}", f"{_pfx}.conv_3x3",   3, _dw_stride, 1, True, True))
    _idx += 1
    ALL_LAYERS.append((f"conv_{_idx}",    f"{_pfx}.reduce_1x1", 1, 1, 0, False, False))
    _idx += 1

ALL_LAYERS.append(("conv_52", "mobilenet_v2.conv_1x1", 1, 1, 0, False, True))

# Residual skip connections:  target_layer -> source_layer
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

# Layers at which we compare with HuggingFace intermediate activations
CHECKPOINT_LAYERS = {"conv_1", "conv_dw_2", "conv_3", "conv_52"}


# ---------------------------------------------------------------------------
# BN folding
# ---------------------------------------------------------------------------

def fold_bn(conv_weight, bn_weight, bn_bias, bn_mean, bn_var, eps=BN_EPS):
    """Fold batch-norm into conv weight/bias.  All float64."""
    inv_std  = 1.0 / np.sqrt(bn_var + eps)
    scale    = bn_weight * inv_std
    shape    = [conv_weight.shape[0]] + [1] * (conv_weight.ndim - 1)
    w_folded = conv_weight * scale.reshape(shape)
    b_folded = bn_bias - bn_weight * bn_mean * inv_std
    return w_folded, b_folded


def get_conv_bn(sd, prefix):
    """Extract conv + BN params from HuggingFace state dict as float64."""
    conv_w = sd[f"{prefix}.convolution.weight"].float().numpy().astype(np.float64)
    gamma  = sd[f"{prefix}.normalization.weight"].float().numpy().astype(np.float64)
    beta   = sd[f"{prefix}.normalization.bias"].float().numpy().astype(np.float64)
    mean   = sd[f"{prefix}.normalization.running_mean"].float().numpy().astype(np.float64)
    var    = sd[f"{prefix}.normalization.running_var"].float().numpy().astype(np.float64)
    return conv_w, gamma, beta, mean, var


# ---------------------------------------------------------------------------
# Weight reshaping  (must match im2col patch order)
# ---------------------------------------------------------------------------

def reshape_conv_weight(w):
    """Regular conv: [out_ch, in_ch, kH, kW] -> [patch_size, out_ch].

    Gemmini im2col produces patches in (kH, kW, in_ch) order, so we must
    transpose to (out_ch, kH, kW, in_ch) before flattening.
    """
    out_ch = w.shape[0]
    if w.ndim == 4:
        w = w.transpose(0, 2, 3, 1)   # [out_ch, kH, kW, in_ch]
    return w.reshape(out_ch, -1).T     # [kH*kW*in_ch, out_ch]


def reshape_dw_weight(w):
    """Depthwise conv: [channels, 1, kH, kW] -> [channels, kH, kW]."""
    assert w.shape[1] == 1, f"Depthwise weight dim-1 must be 1, got {w.shape[1]}"
    return w.squeeze(1)


# ---------------------------------------------------------------------------
# im2col  (float64 version)
# ---------------------------------------------------------------------------

def im2col_float(input_nchw, kernel_size, stride, padding):
    """im2col with patch order (kH, kW, in_ch) -- matches Gemmini layout."""
    N, C, H, W = input_nchw.shape
    kH = kW = kernel_size
    OH = (H + 2 * padding - kH) // stride + 1
    OW = (W + 2 * padding - kW) // stride + 1

    if padding > 0:
        input_nchw = np.pad(
            input_nchw,
            ((0, 0), (0, 0), (padding, padding), (padding, padding)),
            mode='constant', constant_values=0.0,
        )

    patches = np.zeros((N, OH, OW, kH * kW * C), dtype=np.float64)
    for i in range(OH):
        for j in range(OW):
            patch = input_nchw[:, :, i*stride:i*stride+kH, j*stride:j*stride+kW]
            # transpose to (N, kH, kW, C) then flatten spatial+channel
            patches[:, i, j, :] = patch.transpose(0, 2, 3, 1).reshape(N, -1)

    return patches.reshape(N * OH * OW, kH * kW * C), OH, OW


# ---------------------------------------------------------------------------
# Float convolution ops
# ---------------------------------------------------------------------------

def float_conv(x_nchw, w_mat, bias, kernel, stride, padding):
    """Regular convolution via im2col + matmul, all float64.

    Args:
        x_nchw: [N, C, H, W] float64
        w_mat:  [patch_size, out_ch] float64  (already reshaped)
        bias:   [out_ch] float64
        kernel, stride, padding: int

    Returns:
        [N, out_ch, OH, OW] float64
    """
    N, C, H, W = x_nchw.shape

    if kernel == 1 and stride == 1:
        # Fast path for 1x1 stride-1: no im2col needed
        x_flat = x_nchw.reshape(N, C, H * W).transpose(0, 2, 1).reshape(N * H * W, C)
        out_h = out_w = H
    elif kernel == 1 and stride > 1:
        # 1x1 with stride: subsample then flatten
        x_s = x_nchw[:, :, ::stride, ::stride]
        _N, _C, _H, _W = x_s.shape
        x_flat = x_s.reshape(_N, _C, _H * _W).transpose(0, 2, 1).reshape(_N * _H * _W, _C)
        out_h = out_w = _H
    else:
        x_flat, out_h, out_w = im2col_float(x_nchw, kernel, stride, padding)

    # matmul:  [N*OH*OW, patch_size] @ [patch_size, out_ch] -> [N*OH*OW, out_ch]
    acc = x_flat @ w_mat
    acc += bias.reshape(1, -1)

    out_ch = w_mat.shape[1]
    return acc.reshape(N, out_h, out_w, out_ch).transpose(0, 3, 1, 2)


def float_conv_dw(x_nchw, w_dw, bias, stride, padding):
    """Depthwise 3x3 convolution, all float64.

    Args:
        x_nchw: [N, C, H, W] float64
        w_dw:   [C, 3, 3] float64
        bias:   [C] float64

    Returns:
        [N, C, OH, OW] float64
    """
    N, C, H, W = x_nchw.shape
    kH = kW = 3
    OH = (H + 2 * padding - kH) // stride + 1
    OW = (W + 2 * padding - kW) // stride + 1

    if padding > 0:
        x_pad = np.pad(
            x_nchw,
            ((0, 0), (0, 0), (padding, padding), (padding, padding)),
            mode='constant', constant_values=0.0,
        )
    else:
        x_pad = x_nchw

    acc = np.zeros((N, C, OH, OW), dtype=np.float64)
    for ki in range(kH):
        for kj in range(kW):
            rows = np.arange(OH) * stride + ki
            cols = np.arange(OW) * stride + kj
            patch = x_pad[:, :, rows[:, None], cols[None, :]]   # [N, C, OH, OW]
            acc += patch * w_dw[:, ki, kj].reshape(1, C, 1, 1)

    acc += bias.reshape(1, C, 1, 1)
    return acc


# ---------------------------------------------------------------------------
# Capture HuggingFace intermediate activations
# ---------------------------------------------------------------------------

def register_hf_hooks(model):
    """Register forward hooks on BN layers to capture post-BN activations.

    HuggingFace MobileNetV2 applies ReLU6 AFTER BN inside each block, but
    the BN hook fires BEFORE the activation.  We record the raw post-BN
    output; activations are applied separately for comparison.
    """
    captured = {}
    handles  = []

    for gn, pfx, _k, _s, _p, _dw, _relu in ALL_LAYERS:
        parts = pfx.split(".")
        mod = model
        for pt in parts:
            mod = getattr(mod, pt)
        bn_mod = mod.normalization

        def _make_hook(name):
            def hook(_module, _inp, out):
                captured[name] = out.detach().float().numpy().astype(np.float64)
            return hook

        h = bn_mod.register_forward_hook(_make_hook(gn))
        handles.append(h)

    return captured, handles


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import cv2

    print("=" * 78)
    print("Float-Through Forward Pass  --  MobileNetV2 Topology Verification")
    print("=" * 78)

    # ------------------------------------------------------------------
    # 1. Load model
    # ------------------------------------------------------------------
    print(f"\nLoading {MODEL_NAME} ...")
    model = MobileNetV2ForImageClassification.from_pretrained(MODEL_NAME)
    model.eval()
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    num_classes = sd["classifier.weight"].shape[0]
    print(f"  Model output classes: {num_classes}")

    # ------------------------------------------------------------------
    # 2. Load and preprocess image  (standard ImageNet normalisation)
    # ------------------------------------------------------------------
    print(f"\nLoading image: {IMAGE_PATH}")
    img_bgr = cv2.imread(IMAGE_PATH)
    if img_bgr is None:
        print(f"ERROR: cannot read {IMAGE_PATH}")
        sys.exit(1)

    img_resized = cv2.resize(img_bgr, (224, 224))
    img_rgb     = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

    # Float normalisation: (pixel / 255 - mean) / std
    img_f = img_rgb.astype(np.float64) / 255.0
    for c in range(3):
        img_f[:, :, c] = (img_f[:, :, c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]

    x_input = img_f.transpose(2, 0, 1)[np.newaxis]   # [1, 3, 224, 224]  float64
    print(f"  Input shape: {x_input.shape}  dtype: {x_input.dtype}")
    print(f"  Input range: [{x_input.min():.4f}, {x_input.max():.4f}]")

    # ------------------------------------------------------------------
    # 3. Run HuggingFace reference  (with intermediate hooks)
    # ------------------------------------------------------------------
    print("\n--- HuggingFace Reference Forward Pass ---")
    hf_captured, hf_handles = register_hf_hooks(model)

    x_torch = torch.tensor(x_input, dtype=torch.float32)
    with torch.no_grad():
        hf_out = model(x_torch)
        hf_logits = hf_out.logits[0].float().numpy().astype(np.float64)

    for h in hf_handles:
        h.remove()

    # Handle 1001-class models (skip background class 0)
    if num_classes == 1001:
        hf_logits = hf_logits[1:]

    hf_pred  = int(np.argmax(hf_logits))
    hf_top5  = set(np.argsort(hf_logits)[-5:].tolist())
    print(f"  HuggingFace prediction: {hf_pred}  (true label: {TRUE_LABEL})")
    print(f"  HuggingFace top-5:      {sorted(hf_top5)}")
    print(f"  Correct: {'YES' if hf_pred == TRUE_LABEL else 'NO'}")

    # ------------------------------------------------------------------
    # 4. Build BN-folded float weights for all conv layers
    # ------------------------------------------------------------------
    print("\n--- Building BN-folded float weights ---")
    layer_weights = {}

    for gn, pfx, k, s, pad, dw, relu in ALL_LAYERS:
        conv_w, gamma, beta, bn_m, bn_v = get_conv_bn(sd, pfx)
        w_folded, b_folded = fold_bn(conv_w, gamma, beta, bn_m, bn_v)

        if dw:
            w_mat = reshape_dw_weight(w_folded)   # [C, 3, 3]
        else:
            w_mat = reshape_conv_weight(w_folded)  # [patch_size, out_ch]

        layer_weights[gn] = dict(w=w_mat, b=b_folded, kernel=k, stride=s,
                                  padding=pad, dw=dw, relu=relu)

    # FC layer
    fc_w = sd["classifier.weight"].float().numpy().astype(np.float64)  # [num_classes, 1280]
    fc_b = sd["classifier.bias"].float().numpy().astype(np.float64)    # [num_classes]
    if num_classes == 1001:
        fc_w = fc_w[1:, :]
        fc_b = fc_b[1:]

    print(f"  Built weights for {len(layer_weights)} conv layers + FC")

    # ------------------------------------------------------------------
    # 5. Run our manual float forward pass
    # ------------------------------------------------------------------
    print("\n--- Manual Float Forward Pass (our topology) ---")
    print(f"{'Layer':14s}  {'Shape':22s}  {'Min':>10s}  {'Max':>10s}  {'Act':>6s}")
    print("-" * 78)

    out = x_input.copy()   # [1, 3, 224, 224]  float64
    stored = {}             # store outputs for residual connections
    our_activations = {}    # store activations at checkpoint layers

    for gn, pfx, k, s, pad, dw, relu in ALL_LAYERS:
        lw = layer_weights[gn]

        if dw:
            out = float_conv_dw(out, lw["w"], lw["b"], lw["stride"], lw["padding"])
        else:
            out = float_conv(out, lw["w"], lw["b"], lw["kernel"], lw["stride"], lw["padding"])

        # Activation: ReLU (NOT ReLU6) for relu layers, nothing for linear
        act_str = "---"
        if relu:
            out = np.maximum(out, 0.0)
            act_str = "ReLU"

        # Store for potential residual connections
        stored[gn] = out.copy()

        # ResAdd: applied AFTER the reduce/project layer
        if gn in RESIDUAL_SKIP:
            skip_src = RESIDUAL_SKIP[gn]
            skip = stored[skip_src]
            out = skip + out       # float add -- no scaling, no clipping
            stored[gn] = out.copy()  # update with resadd result
            act_str += "+res"

        # Record checkpoint
        if gn in CHECKPOINT_LAYERS:
            our_activations[gn] = out.copy()

        print(f"  {gn:12s}  {str(out.shape):22s}  {out.min():10.4f}  "
              f"{out.max():10.4f}  {act_str:>6s}")

    # ------------------------------------------------------------------
    # 6. Global average pooling  + FC
    # ------------------------------------------------------------------
    print("-" * 78)
    # out shape: [1, 1280, 7, 7]
    avg_pool = out.mean(axis=(2, 3))   # [1, 1280]
    our_activations["avg_pool"] = avg_pool.copy()
    print(f"  {'avg_pool':12s}  {str(avg_pool.shape):22s}  {avg_pool.min():10.4f}  "
          f"{avg_pool.max():10.4f}")

    # FC:  logits = avg_pool @ fc_w.T + fc_b
    # fc_w is [1000, 1280],  avg_pool is [1, 1280]
    our_logits = avg_pool @ fc_w.T + fc_b.reshape(1, -1)   # [1, 1000]
    our_logits = our_logits[0]   # [1000]
    print(f"  {'FC':12s}  {str(our_logits.shape):22s}  {our_logits.min():10.4f}  "
          f"{our_logits.max():10.4f}")

    our_pred = int(np.argmax(our_logits))
    our_top5 = set(np.argsort(our_logits)[-5:].tolist())

    # ------------------------------------------------------------------
    # 7. Comparison: our float pipeline vs HuggingFace
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("RESULTS COMPARISON")
    print("=" * 78)

    print(f"\n  {'':30s}  {'Ours (float)':>14s}  {'HuggingFace':>14s}")
    print(f"  {'-'*30}  {'-'*14}  {'-'*14}")
    print(f"  {'Top-1 prediction':30s}  {our_pred:>14d}  {hf_pred:>14d}")
    print(f"  {'True label':30s}  {TRUE_LABEL:>14d}  {TRUE_LABEL:>14d}")
    match_str_ours = "CORRECT" if our_pred == TRUE_LABEL else "WRONG"
    match_str_hf   = "CORRECT" if hf_pred  == TRUE_LABEL else "WRONG"
    print(f"  {'Correct?':30s}  {match_str_ours:>14s}  {match_str_hf:>14s}")
    print(f"  {'Top-5':30s}  {str(sorted(our_top5)):>14s}")
    print(f"  {'':30s}  {str(sorted(hf_top5)):>14s}")

    # Logit correlation
    logit_corr = np.corrcoef(our_logits, hf_logits)[0, 1]
    logit_mse  = float(np.mean((our_logits - hf_logits) ** 2))
    logit_max_diff = float(np.max(np.abs(our_logits - hf_logits)))
    print(f"\n  Logit correlation:  {logit_corr:.10f}")
    print(f"  Logit MSE:          {logit_mse:.6e}")
    print(f"  Logit max |diff|:   {logit_max_diff:.6e}")

    # ------------------------------------------------------------------
    # 8. Layer-by-layer comparison at checkpoints
    # ------------------------------------------------------------------
    print("\n" + "-" * 78)
    print("Layer-by-layer comparison at checkpoints")
    print(f"  NOTE: HuggingFace applies ReLU6 internally.  Our pipeline uses")
    print(f"        plain ReLU.  We compare our output vs HuggingFace post-BN")
    print(f"        (before HF applies its own activation), then also compare")
    print(f"        after applying the SAME activation on both sides.")
    print("-" * 78)

    print(f"\n  {'Layer':14s}  {'Corr(raw BN)':>14s}  {'Corr(+ReLU)':>14s}  "
          f"{'Corr(+ReLU6)':>14s}  {'MaxDiff(raw)':>14s}")
    print(f"  {'-'*14}  {'-'*14}  {'-'*14}  {'-'*14}  {'-'*14}")

    for gn in list(CHECKPOINT_LAYERS) + ["avg_pool"]:
        ours = our_activations.get(gn)
        if ours is None:
            continue

        if gn == "avg_pool":
            # No HF hook for avg_pool -- compare logits instead
            continue

        hf_raw = hf_captured.get(gn)
        if hf_raw is None:
            print(f"  {gn:14s}  {'(no HF data)':>14s}")
            continue

        # Look up whether this layer has relu
        has_relu = False
        for _gn, _pfx, _k, _s, _p, _dw, _relu in ALL_LAYERS:
            if _gn == gn:
                has_relu = _relu
                break

        # Raw post-BN comparison (before any activation)
        # Our output already has activation applied; HF hook captures post-BN
        # before HF applies its activation.  For fair raw comparison, we
        # need the pre-activation version of our output.  But since we
        # stored post-activation, we can only compare post-activation.
        #
        # Strategy: apply the same activation on HF side and compare.

        if ours.shape != hf_raw.shape:
            print(f"  {gn:14s}  SHAPE MISMATCH: ours={ours.shape} hf={hf_raw.shape}")
            continue

        # HF raw (post-BN, no activation yet)
        # Our stored value has ReLU applied (if relu layer) or no act (if linear)
        # For linear layers: both are raw, so compare directly
        if has_relu:
            hf_relu  = np.maximum(hf_raw, 0.0)        # plain ReLU
            hf_relu6 = np.clip(hf_raw, 0.0, 6.0)      # ReLU6

            corr_raw   = np.corrcoef(ours.flatten(), hf_raw.flatten())[0, 1]
            corr_relu  = np.corrcoef(ours.flatten(), hf_relu.flatten())[0, 1]
            corr_relu6 = np.corrcoef(ours.flatten(), hf_relu6.flatten())[0, 1]
            max_diff   = float(np.max(np.abs(ours - hf_raw)))

            print(f"  {gn:14s}  {corr_raw:14.10f}  {corr_relu:14.10f}  "
                  f"{corr_relu6:14.10f}  {max_diff:14.6e}")
        else:
            # Linear layer: our output == post-BN (no activation)
            corr_raw = np.corrcoef(ours.flatten(), hf_raw.flatten())[0, 1]
            max_diff = float(np.max(np.abs(ours - hf_raw)))
            print(f"  {gn:14s}  {corr_raw:14.10f}  {'(linear)':>14s}  "
                  f"{'(linear)':>14s}  {max_diff:14.6e}")

    # ------------------------------------------------------------------
    # 9. Detailed per-checkpoint statistics
    # ------------------------------------------------------------------
    print("\n" + "-" * 78)
    print("Detailed per-checkpoint statistics")
    print("-" * 78)

    for gn in ["conv_1", "conv_dw_2", "conv_3", "conv_52"]:
        ours   = our_activations.get(gn)
        hf_raw = hf_captured.get(gn)
        if ours is None or hf_raw is None:
            continue

        has_relu = False
        for _gn, _pfx, _k, _s, _p, _dw, _relu in ALL_LAYERS:
            if _gn == gn:
                has_relu = _relu
                break

        print(f"\n  {gn}:")
        print(f"    Our shape:       {ours.shape}")
        print(f"    Our range:       [{ours.min():.6f}, {ours.max():.6f}]")
        print(f"    HF post-BN range:[{hf_raw.min():.6f}, {hf_raw.max():.6f}]")

        if has_relu:
            hf_relu  = np.maximum(hf_raw, 0.0)
            hf_relu6 = np.clip(hf_raw, 0.0, 6.0)
            diff_relu  = np.abs(ours - hf_relu)
            diff_relu6 = np.abs(ours - hf_relu6)

            print(f"    HF+ReLU range:   [{hf_relu.min():.6f}, {hf_relu.max():.6f}]")
            print(f"    HF+ReLU6 range:  [{hf_relu6.min():.6f}, {hf_relu6.max():.6f}]")
            print(f"    |ours - HF+ReLU|:  mean={diff_relu.mean():.6e}  "
                  f"max={diff_relu.max():.6e}")
            print(f"    |ours - HF+ReLU6|: mean={diff_relu6.mean():.6e}  "
                  f"max={diff_relu6.max():.6e}")

            # Check if any HF values exceed 6 (would be clipped by ReLU6 but not our ReLU)
            n_above_6 = int(np.sum(hf_raw > 6.0))
            if n_above_6 > 0:
                print(f"    ** {n_above_6} HF values > 6.0 (ReLU6 would clip, our ReLU does not)")
        else:
            diff = np.abs(ours - hf_raw)
            print(f"    |ours - HF|:     mean={diff.mean():.6e}  max={diff.max():.6e}")

    # ------------------------------------------------------------------
    # 10. Final verdict
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    predictions_match = (our_pred == hf_pred)
    topology_correct  = predictions_match and logit_corr > 0.999

    if topology_correct:
        print("VERDICT: TOPOLOGY IS CORRECT")
        print(f"  Both predict class {our_pred} (true label {TRUE_LABEL}).")
        print(f"  Logit correlation = {logit_corr:.10f}")
        print(f"  Any accuracy loss in INT8 mode is purely from quantization.")
    elif predictions_match:
        print("VERDICT: PREDICTIONS MATCH (but logit correlation is not perfect)")
        print(f"  Both predict class {our_pred} (true label {TRUE_LABEL}).")
        print(f"  Logit correlation = {logit_corr:.10f}")
        print(f"  There may be minor topology differences (e.g., ReLU vs ReLU6).")
        print(f"  Check the layer-by-layer comparison above for details.")
    else:
        print("VERDICT: PREDICTIONS DIFFER -- TOPOLOGY MAY HAVE AN ISSUE")
        print(f"  Our prediction:  {our_pred}")
        print(f"  HF prediction:   {hf_pred}")
        print(f"  True label:      {TRUE_LABEL}")
        print(f"  Logit correlation = {logit_corr:.10f}")
        print(f"  Check the layer-by-layer comparison above to find where divergence occurs.")

        # Identify divergence point
        print("\n  Checking where divergence starts:")
        for gn in ["conv_1", "conv_dw_2", "conv_3", "conv_52"]:
            ours   = our_activations.get(gn)
            hf_raw = hf_captured.get(gn)
            if ours is None or hf_raw is None:
                continue
            has_relu = False
            for _gn, _pfx, _k, _s, _p, _dw, _relu in ALL_LAYERS:
                if _gn == gn:
                    has_relu = _relu
                    break
            if has_relu:
                hf_act = np.maximum(hf_raw, 0.0)
            else:
                hf_act = hf_raw
            corr = np.corrcoef(ours.flatten(), hf_act.flatten())[0, 1]
            status = "OK" if corr > 0.999 else "** DIVERGED **"
            print(f"    {gn:14s}  corr={corr:.10f}  {status}")

    print("=" * 78)


if __name__ == "__main__":
    main()
