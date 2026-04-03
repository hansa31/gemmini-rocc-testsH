#!/usr/bin/env python3
"""
Verification script for MobileNetV2 CIFAR-10 float pipeline.

Compares three pipelines:
  1) HuggingFace reference (ReLU6, float32)
  2) Our manual pipeline with ReLU6 (float64, matching HF activations)
  3) Our manual pipeline with plain ReLU (float64, matching Gemmini hardware)

Uses the actual CIFAR-10 model: jialicheng/cifar10_mobilenet-v2
Tests on CIFAR-10 test images (native 32x32 or resized to 224x224 based on model).

Usage:
    conda run -n ImageNet python test_cifar10_float.py
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from transformers import MobileNetV2ForImageClassification
from torchvision.datasets import CIFAR10

MODEL_NAME = "jialicheng/cifar10_mobilenet-v2"
INPUT_DIM = 224  # HF MobileNetV2 expects 224x224
CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]

# ── Architecture (must match extract script) ──────────────────────────────
LAYER_ARCH = [
    # (name, kernel, in_ch, out_ch, stride, padding, depthwise, activation)
    ("conv_1",     3,   3,   32, 2, 1, False, "relu6"),
    ("conv_dw_2",  3,  32,   32, 1, 1,  True, "relu6"),
    ("conv_3",     1,  32,   16, 1, 0, False, "none"),
    ("conv_4",     1,  16,   96, 1, 0, False, "relu6"),
    ("conv_dw_5",  3,  96,   96, 2, 1,  True, "relu6"),
    ("conv_6",     1,  96,   24, 1, 0, False, "none"),
    ("conv_7",     1,  24,  144, 1, 0, False, "relu6"),
    ("conv_dw_8",  3, 144,  144, 1, 1,  True, "relu6"),
    ("conv_9",     1, 144,   24, 1, 0, False, "none"),
    ("conv_10",    1,  24,  144, 1, 0, False, "relu6"),
    ("conv_dw_11", 3, 144,  144, 2, 1,  True, "relu6"),
    ("conv_12",    1, 144,   32, 1, 0, False, "none"),
    ("conv_13",    1,  32,  192, 1, 0, False, "relu6"),
    ("conv_dw_14", 3, 192,  192, 1, 1,  True, "relu6"),
    ("conv_15",    1, 192,   32, 1, 0, False, "none"),
    ("conv_16",    1,  32,  192, 1, 0, False, "relu6"),
    ("conv_dw_17", 3, 192,  192, 1, 1,  True, "relu6"),
    ("conv_18",    1, 192,   32, 1, 0, False, "none"),
    ("conv_19",    1,  32,  192, 1, 0, False, "relu6"),
    ("conv_dw_20", 3, 192,  192, 2, 1,  True, "relu6"),
    ("conv_21",    1, 192,   64, 1, 0, False, "none"),
    ("conv_22",    1,  64,  384, 1, 0, False, "relu6"),
    ("conv_dw_23", 3, 384,  384, 1, 1,  True, "relu6"),
    ("conv_24",    1, 384,   64, 1, 0, False, "none"),
    ("conv_25",    1,  64,  384, 1, 0, False, "relu6"),
    ("conv_dw_26", 3, 384,  384, 1, 1,  True, "relu6"),
    ("conv_27",    1, 384,   64, 1, 0, False, "none"),
    ("conv_28",    1,  64,  384, 1, 0, False, "relu6"),
    ("conv_dw_29", 3, 384,  384, 1, 1,  True, "relu6"),
    ("conv_30",    1, 384,   64, 1, 0, False, "none"),
    ("conv_31",    1,  64,  384, 1, 0, False, "relu6"),
    ("conv_dw_32", 3, 384,  384, 1, 1,  True, "relu6"),
    ("conv_33",    1, 384,   96, 1, 0, False, "none"),
    ("conv_34",    1,  96,  576, 1, 0, False, "relu6"),
    ("conv_dw_35", 3, 576,  576, 1, 1,  True, "relu6"),
    ("conv_36",    1, 576,   96, 1, 0, False, "none"),
    ("conv_37",    1,  96,  576, 1, 0, False, "relu6"),
    ("conv_dw_38", 3, 576,  576, 1, 1,  True, "relu6"),
    ("conv_39",    1, 576,   96, 1, 0, False, "none"),
    ("conv_40",    1,  96,  576, 1, 0, False, "relu6"),
    ("conv_dw_41", 3, 576,  576, 2, 1,  True, "relu6"),
    ("conv_42",    1, 576,  160, 1, 0, False, "none"),
    ("conv_43",    1, 160,  960, 1, 0, False, "relu6"),
    ("conv_dw_44", 3, 960,  960, 1, 1,  True, "relu6"),
    ("conv_45",    1, 960,  160, 1, 0, False, "none"),
    ("conv_46",    1, 160,  960, 1, 0, False, "relu6"),
    ("conv_dw_47", 3, 960,  960, 1, 1,  True, "relu6"),
    ("conv_48",    1, 960,  160, 1, 0, False, "none"),
    ("conv_49",    1, 160,  960, 1, 0, False, "relu6"),
    ("conv_dw_50", 3, 960,  960, 1, 1,  True, "relu6"),
    ("conv_51",    1, 960,  320, 1, 0, False, "none"),
    ("conv_52",    1, 320, 1280, 1, 0, False, "relu6"),
]

# ── HuggingFace → Gemmini layer mapping ───────────────────────────────────

def build_mapping():
    mapping = []
    mapping.append(("conv_1",    "mobilenet_v2.conv_stem.first_conv"))
    mapping.append(("conv_dw_2", "mobilenet_v2.conv_stem.conv_3x3"))
    mapping.append(("conv_3",    "mobilenet_v2.conv_stem.reduce_1x1"))
    gemmini_idx = 4
    for hf_idx in range(0, 16):
        prefix = f"mobilenet_v2.layer.{hf_idx}"
        mapping.append((f"conv_{gemmini_idx}",    f"{prefix}.expand_1x1"))
        gemmini_idx += 1
        mapping.append((f"conv_dw_{gemmini_idx}", f"{prefix}.conv_3x3"))
        gemmini_idx += 1
        mapping.append((f"conv_{gemmini_idx}",    f"{prefix}.reduce_1x1"))
        gemmini_idx += 1
    mapping.append(("conv_52", "mobilenet_v2.conv_1x1"))
    return mapping


# ── BN folding ────────────────────────────────────────────────────────────

def fold_bn(conv_w, bn_w, bn_b, bn_mean, bn_var, eps=1e-5):
    inv_std = 1.0 / np.sqrt(bn_var + eps)
    scale = bn_w * inv_std
    shape = [conv_w.shape[0]] + [1] * (conv_w.ndim - 1)
    w = conv_w * scale.reshape(shape)
    b = bn_b - bn_w * bn_mean * inv_std
    return w, b


def get_conv_bn(sd, prefix):
    conv_w = sd[f"{prefix}.convolution.weight"].numpy().astype(np.float64)
    bn_w   = sd[f"{prefix}.normalization.weight"].numpy().astype(np.float64)
    bn_b   = sd[f"{prefix}.normalization.bias"].numpy().astype(np.float64)
    bn_m   = sd[f"{prefix}.normalization.running_mean"].numpy().astype(np.float64)
    bn_v   = sd[f"{prefix}.normalization.running_var"].numpy().astype(np.float64)
    return fold_bn(conv_w, bn_w, bn_b, bn_m, bn_v)


# ── im2col + conv ────────────────────────────────────────────────────────

def im2col(x, kH, kW, stride, padding):
    """x: [B, C, H, W] -> patches [B*OH*OW, kH*kW*C]  (Gemmini patch order)."""
    B, C, H, W = x.shape
    if padding > 0:
        x = np.pad(x, ((0,0),(0,0),(padding,padding),(padding,padding)))
    _, _, pH, pW = x.shape
    OH = (pH - kH) // stride + 1
    OW = (pW - kW) // stride + 1
    patches = np.zeros((B * OH * OW, kH * kW * C), dtype=np.float64)
    idx = 0
    for b in range(B):
        for oh in range(OH):
            for ow in range(OW):
                rs = oh * stride
                cs = ow * stride
                # Gemmini patch order: (kH, kW, C) — spatial outer, channel inner
                patch = x[b, :, rs:rs+kH, cs:cs+kW]  # [C, kH, kW]
                patch = patch.transpose(1, 2, 0).reshape(-1)   # -> [kH, kW, C] -> flat
                patches[idx] = patch
                idx += 1
    return patches, OH, OW


def conv_forward(x, w_folded, b_folded, kH, stride, padding, activation, depthwise=False):
    """Run one conv layer.
    x: [B, C, H, W]
    Returns [B, out_ch, OH, OW]
    """
    B, C, H, W = x.shape
    out_ch = w_folded.shape[0]

    if depthwise:
        assert out_ch == C
        # dw: each filter is [1, kH, kW] applied per channel
        kW = kH
        if padding > 0:
            xp = np.pad(x, ((0,0),(0,0),(padding,padding),(padding,padding)))
        else:
            xp = x
        _, _, pH, pW = xp.shape
        OH = (pH - kH) // stride + 1
        OW = (pW - kW) // stride + 1
        out = np.zeros((B, out_ch, OH, OW), dtype=np.float64)
        # w_folded: [out_ch, 1, kH, kW]
        for b in range(B):
            for c in range(out_ch):
                for oh in range(OH):
                    for ow in range(OW):
                        rs = oh * stride
                        cs = ow * stride
                        patch = xp[b, c, rs:rs+kH, cs:cs+kW]
                        out[b, c, oh, ow] = np.sum(patch * w_folded[c, 0]) + b_folded[c]
        return apply_activation(out, activation)

    # Standard conv: use im2col
    kW = kH
    patches, OH, OW = im2col(x, kH, kW, stride, padding)
    # Gemmini weight layout: [patch_size, out_ch]
    # patches: [B*OH*OW, patch_size]
    # w_folded from PyTorch: [out_ch, C, kH, kW]
    # reshape to [out_ch, patch_size] then transpose for Gemmini matmul
    w_gemmini = w_folded.reshape(out_ch, -1)
    # Need to reorder to match im2col's (kH, kW, C) patch order
    # PyTorch weight: [out_ch, C, kH, kW] -> for each filter: [C, kH, kW]
    # Gemmini: im2col produces [kH, kW, C] patches
    # So we transpose each filter from [C, kH, kW] to [kH, kW, C] then flatten
    w_reordered = np.zeros_like(w_gemmini)
    for oc in range(out_ch):
        filt = w_folded[oc]  # [C, kH, kW]
        filt_reord = filt.transpose(1, 2, 0).reshape(-1)  # [kH, kW, C]
        w_reordered[oc] = filt_reord

    # matmul: patches [N, K] @ w_reordered.T [K, out_ch] + bias
    result = patches @ w_reordered.T + b_folded.reshape(1, -1)  # [N, out_ch]
    return apply_activation(result.reshape(B, OH, OW, out_ch).transpose(0, 3, 1, 2), activation)


def apply_activation(x, act):
    if act == "relu6":
        return np.clip(x, 0, 6)
    elif act == "relu":
        return np.maximum(x, 0)
    return x  # "none"


# ── Residual connections ──────────────────────────────────────────────────

RESIDUAL_PAIRS = {
    "conv_9":  "conv_6",   # block 1, stride=1
    "conv_15": "conv_12",  # block 3
    "conv_18": "conv_15",  # block 4
    "conv_24": "conv_21",  # block 6
    "conv_27": "conv_24",  # block 7
    "conv_30": "conv_27",  # block 8
    "conv_33": "conv_30",  # block 9 (no! different dims: 64->96)
    "conv_36": "conv_33",  # block 10 (96->96, stride=1)
    "conv_39": "conv_36",  # block 11
    "conv_45": "conv_42",  # block 13
    "conv_48": "conv_45",  # block 14
}

def should_add_residual(name, outputs):
    """Check if this layer's output should have a residual added."""
    if name not in RESIDUAL_PAIRS:
        return False
    src = RESIDUAL_PAIRS[name]
    if src not in outputs:
        return False
    # Shapes must match
    return outputs[src].shape == outputs.get(name, np.zeros(0)).shape


# ── Main ──────────────────────────────────────────────────────────────────

def run_pipeline(sd, image_batch, use_relu6=True):
    """Run our manual float pipeline.
    image_batch: [B, 3, H, W] float64 in [0, 1]
    Returns logits [B, 10]
    """
    x = image_batch.copy()
    outputs = {}
    layer_stats = {}

    for name, kernel, in_ch, out_ch, stride, padding, dw, _act_str in LAYER_ARCH:
        # Get weights from mapping
        mapping = build_mapping()
        hf_prefix = None
        for gn, hp in mapping:
            if gn == name:
                hf_prefix = hp
                break
        assert hf_prefix is not None, f"No mapping for {name}"

        w, b = get_conv_bn(sd, hf_prefix)

        act = "relu6" if use_relu6 else "relu"
        if _act_str == "none":
            act = "none"

        x = conv_forward(x, w, b, kernel, stride, padding, act, depthwise=dw)
        outputs[name] = x.copy()

        # Add residual connection if applicable
        if name in RESIDUAL_PAIRS:
            src = RESIDUAL_PAIRS[name]
            if src in outputs and outputs[src].shape == x.shape:
                x = x + outputs[src]
                outputs[name] = x.copy()

        layer_stats[name] = {
            "mean": float(np.mean(x)),
            "std": float(np.std(x)),
            "min": float(np.min(x)),
            "max": float(np.max(x)),
            "shape": x.shape,
        }

    # Global average pool
    x = np.mean(x, axis=(2, 3))  # [B, 1280]

    # FC layer
    fc_w = sd["classifier.weight"].numpy().astype(np.float64)  # [10, 1280]
    fc_b = sd["classifier.bias"].numpy().astype(np.float64)    # [10]
    logits = x @ fc_w.T + fc_b.reshape(1, -1)  # [B, 10]

    return logits, layer_stats


def run_huggingface(model, images_tensor):
    """Run HuggingFace model for reference."""
    with torch.no_grad():
        outputs = model(images_tensor)
    return outputs.logits.numpy()


def main():
    print(f"Loading model: {MODEL_NAME}")
    model = MobileNetV2ForImageClassification.from_pretrained(MODEL_NAME)
    model.eval()
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    # Load CIFAR-10 test images
    print("Loading CIFAR-10 test set...")
    dataset = CIFAR10(root="/tmp/cifar10_data", train=False, download=True)

    # Test on first 4 images (matching BATCH_SIZE=4 in C code)
    test_indices = [0, 1, 2, 3]
    images = []
    labels = []
    for idx in test_indices:
        img_pil, label = dataset[idx]
        # Resize to 224x224 (matching what the model expects)
        from PIL import Image as PILImage
        img_pil = img_pil.resize((INPUT_DIM, INPUT_DIM), PILImage.BILINEAR)
        img_np = np.array(img_pil, dtype=np.float64) / 255.0  # [H, W, 3] in [0,1]
        images.append(img_np)
        labels.append(label)

    print(f"Test images: indices {test_indices}")
    print(f"True labels: {labels} ({[CIFAR10_CLASSES[l] for l in labels]})")

    # Prepare batches
    # For HF: [B, 3, H, W] float32 tensor in [0,1]
    images_np = np.stack(images)  # [B, H, W, 3]
    images_chw = images_np.transpose(0, 3, 1, 2)  # [B, 3, H, W]
    images_tensor = torch.from_numpy(images_chw).float()

    # ── 1) HuggingFace reference ──
    print("\n" + "="*60)
    print("1) HuggingFace reference (ReLU6, float32)")
    print("="*60)
    hf_logits = run_huggingface(model, images_tensor)
    hf_preds = np.argmax(hf_logits, axis=1)
    for i, idx in enumerate(test_indices):
        correct = "OK" if hf_preds[i] == labels[i] else "WRONG"
        print(f"  Image {idx}: pred={hf_preds[i]} ({CIFAR10_CLASSES[hf_preds[i]]}), "
              f"true={labels[i]} ({CIFAR10_CLASSES[labels[i]]}) [{correct}]")
    hf_acc = np.mean(hf_preds == np.array(labels))
    print(f"  Accuracy: {hf_acc*100:.0f}% ({int(hf_acc*len(labels))}/{len(labels)})")

    # ── 2) Our pipeline with ReLU6 ──
    print("\n" + "="*60)
    print("2) Manual pipeline with ReLU6 (float64)")
    print("="*60)
    relu6_logits, relu6_stats = run_pipeline(sd, images_chw.astype(np.float64), use_relu6=True)
    relu6_preds = np.argmax(relu6_logits, axis=1)
    for i, idx in enumerate(test_indices):
        correct = "OK" if relu6_preds[i] == labels[i] else "WRONG"
        print(f"  Image {idx}: pred={relu6_preds[i]} ({CIFAR10_CLASSES[relu6_preds[i]]}), "
              f"true={labels[i]} ({CIFAR10_CLASSES[labels[i]]}) [{correct}]")
    relu6_acc = np.mean(relu6_preds == np.array(labels))
    print(f"  Accuracy: {relu6_acc*100:.0f}% ({int(relu6_acc*len(labels))}/{len(labels)})")

    # Correlation with HF
    for i in range(len(test_indices)):
        corr = np.corrcoef(hf_logits[i], relu6_logits[i])[0, 1]
        print(f"  Image {test_indices[i]}: logit correlation with HF = {corr:.4f}")

    # ── 3) Our pipeline with plain ReLU ──
    print("\n" + "="*60)
    print("3) Manual pipeline with plain ReLU (float64) — Gemmini hardware")
    print("="*60)
    relu_logits, relu_stats = run_pipeline(sd, images_chw.astype(np.float64), use_relu6=False)
    relu_preds = np.argmax(relu_logits, axis=1)
    for i, idx in enumerate(test_indices):
        correct = "OK" if relu_preds[i] == labels[i] else "WRONG"
        print(f"  Image {idx}: pred={relu_preds[i]} ({CIFAR10_CLASSES[relu_preds[i]]}), "
              f"true={labels[i]} ({CIFAR10_CLASSES[labels[i]]}) [{correct}]")
    relu_acc = np.mean(relu_preds == np.array(labels))
    print(f"  Accuracy: {relu_acc*100:.0f}% ({int(relu_acc*len(labels))}/{len(labels)})")

    for i in range(len(test_indices)):
        corr = np.corrcoef(hf_logits[i], relu_logits[i])[0, 1]
        print(f"  Image {test_indices[i]}: logit correlation with HF = {corr:.4f}")

    # ── Layer-by-layer divergence analysis ──
    print("\n" + "="*60)
    print("Layer-by-layer activation comparison (ReLU6 vs ReLU)")
    print("="*60)
    print(f"  {'Layer':15s}  {'ReLU6 max':>10s}  {'ReLU max':>10s}  {'# > 6.0 (ReLU)':>15s}")
    for name in relu6_stats:
        r6 = relu6_stats[name]
        r = relu_stats[name]
        n_over_6 = int(np.sum(np.array(0)))  # placeholder, need actual tensor
        print(f"  {name:15s}  {r6['max']:10.3f}  {r['max']:10.3f}  {'-':>15s}")

    # ── Summary ──
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"  HuggingFace (ReLU6):   {hf_acc*100:.0f}% accuracy")
    print(f"  Manual (ReLU6):        {relu6_acc*100:.0f}% accuracy")
    print(f"  Manual (ReLU/Gemmini): {relu_acc*100:.0f}% accuracy")
    print()
    if relu_acc < relu6_acc:
        diff = relu6_acc - relu_acc
        print(f"  -> ReLU vs ReLU6 costs {diff*100:.0f}% accuracy on this batch.")
        print(f"  -> Gemmini hardware only supports plain ReLU, so software ReLU6")
        print(f"     clamping after each conv layer would be needed for full accuracy.")
    elif relu_acc == relu6_acc:
        print(f"  -> ReLU and ReLU6 give same accuracy on this small batch.")
        print(f"     Larger evaluation needed for conclusive comparison.")


if __name__ == "__main__":
    main()
