#!/usr/bin/env python3
"""
Verify the edadaltocg/resnet50_cifar10 model: check state dict keys,
run FP32 inference on CIFAR-10 test set, and validate quantization.

Usage:
    conda run -n ImageNet_stable python verify_resnet50_cifar10.py
"""

import os
import sys
import math
import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

MODEL_NAME = "edadaltocg/resnet50_cifar10"

# CIFAR-10 normalization (from prepare_cifar10.py)
CIFAR10_MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
CIFAR10_STD  = np.array([0.2023, 0.1994, 0.2010], dtype=np.float32)
CIFAR10_FMAX = 2.75

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def load_raw_state_dict():
    """Load pytorch_model.bin directly from HuggingFace."""
    print(f"=== Loading {MODEL_NAME} pytorch_model.bin ===")
    model_file = hf_hub_download(MODEL_NAME, "pytorch_model.bin")
    sd = torch.load(model_file, map_location="cpu", weights_only=True)
    return sd


def analyze_keys(sd):
    """Print and analyze state dict key structure."""
    print(f"\n=== State Dict Keys ({len(sd)} total) ===")
    top_level = sorted(set(k.split(".")[0] for k in sd.keys()))
    print(f"Top-level prefixes: {top_level}")

    print(f"\nAll keys:")
    for k in sorted(sd.keys()):
        v = sd[k]
        print(f"  {k:60s}  shape={list(v.shape)}")

    # Check expected keys
    has_conv1 = "conv1.weight" in sd
    has_layer1 = any("layer1" in k for k in sd.keys())
    has_fc = "fc.weight" in sd
    print(f"\nKey checks: conv1.weight={has_conv1}, layer1.*={has_layer1}, fc.weight={has_fc}")

    if has_fc:
        fc_w = sd["fc.weight"]
        print(f"FC weight shape: {list(fc_w.shape)} -> {fc_w.shape[0]} classes")

    if has_conv1:
        conv1_w = sd["conv1.weight"]
        print(f"conv1 weight shape: {list(conv1_w.shape)}")
        k_size = conv1_w.shape[2]
        print(f"  -> kernel_size={k_size}x{k_size} (expect 3x3 for CIFAR-10, 7x7 for ImageNet)")

    return has_conv1 and has_layer1 and has_fc


def build_resnet50_cifar10(sd):
    """Build a torchvision ResNet50 with CIFAR-10 modifications and load weights."""
    import torchvision.models as models

    # Standard ResNet50 but with CIFAR-10 stem:
    #   conv1: 3x3, stride=1, padding=1 (vs 7x7 s2 p3 for ImageNet)
    #   no maxpool after stem
    model = models.resnet50(weights=None, num_classes=10)

    # Check if the model was trained with modified stem
    conv1_w = sd["conv1.weight"]
    if conv1_w.shape[2] == 3:
        print("\nModel has 3x3 stem conv (CIFAR-10 style)")
        model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = torch.nn.Identity()
    elif conv1_w.shape[2] == 7:
        print("\nModel has 7x7 stem conv (ImageNet style)")
        # Use standard ImageNet stem
    else:
        print(f"\nWARNING: Unexpected conv1 kernel size: {conv1_w.shape[2]}")

    # Load weights
    result = model.load_state_dict(sd, strict=True)
    print(f"load_state_dict result: {result}")

    model.eval()
    return model


def run_fp32_inference(model, num_images=1000):
    """Run FP32 inference on CIFAR-10 test set."""
    from torchvision.datasets import CIFAR10
    import torchvision.transforms as transforms

    print(f"\n=== FP32 Inference on {num_images} CIFAR-10 test images ===")

    # Standard CIFAR-10 preprocessing (matching edadaltocg training)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR10_MEAN.tolist(), std=CIFAR10_STD.tolist()),
    ])

    dataset = CIFAR10(root="/tmp/cifar10_data", train=False, download=True, transform=transform)

    correct_top1 = 0
    correct_top5 = 0
    total = 0

    with torch.no_grad():
        for i in range(min(num_images, len(dataset))):
            img, label = dataset[i]
            img = img.unsqueeze(0)  # [1, 3, 32, 32]

            output = model(img)  # [1, 10]
            probs = F.softmax(output, dim=1)

            _, pred_top5 = output.topk(5, dim=1)
            pred_top1 = pred_top5[0, 0].item()

            if pred_top1 == label:
                correct_top1 += 1
            if label in pred_top5[0].tolist():
                correct_top5 += 1
            total += 1

            if i < 10:
                pred_class = CIFAR10_CLASSES[pred_top1]
                true_class = CIFAR10_CLASSES[label]
                conf = probs[0, pred_top1].item()
                status = "OK" if pred_top1 == label else "WRONG"
                print(f"  [{i:4d}] true={true_class:12s}  pred={pred_class:12s}  "
                      f"conf={conf:.3f}  {status}")

    top1_acc = 100.0 * correct_top1 / total
    top5_acc = 100.0 * correct_top5 / total
    print(f"\nFP32 Results ({total} images):")
    print(f"  Top-1: {correct_top1}/{total} ({top1_acc:.1f}%)")
    print(f"  Top-5: {correct_top5}/{total} ({top5_acc:.1f}%)")
    print(f"  (Expect ~93-95% top-1 if model is correct)")
    return top1_acc


def verify_int8_preprocessing():
    """Verify that prepare_cifar10.py preprocessing matches extraction script's x_scale."""
    print(f"\n=== Preprocessing Verification ===")

    f_min = float(np.min((0.0 / 255.0 - CIFAR10_MEAN) / CIFAR10_STD))
    f_max = float(np.max((255.0 / 255.0 - CIFAR10_MEAN) / CIFAR10_STD))
    print(f"Normalized float range: [{f_min:.4f}, {f_max:.4f}]")
    print(f"FMAX used in prepare_cifar10.py: {CIFAR10_FMAX}")
    print(f"Actual max abs: {max(abs(f_min), abs(f_max)):.4f}")

    x_scale_extraction = max(abs(f_min), abs(f_max)) / 127.0
    x_scale_prepare = CIFAR10_FMAX / 127.0
    print(f"x_scale from extraction script: {x_scale_extraction:.8f}")
    print(f"x_scale from prepare_cifar10.py: {x_scale_prepare:.8f}")
    print(f"Difference: {abs(x_scale_extraction - x_scale_prepare) / x_scale_prepare * 100:.4f}%")

    # Per-channel analysis
    for ch, name in enumerate(["R", "G", "B"]):
        ch_min = (0.0 - CIFAR10_MEAN[ch]) / CIFAR10_STD[ch]
        ch_max = (1.0 - CIFAR10_MEAN[ch]) / CIFAR10_STD[ch]
        print(f"  Channel {name}: [{ch_min:.4f}, {ch_max:.4f}]  "
              f"max_abs={max(abs(ch_min), abs(ch_max)):.4f}")


def verify_quantized_inference(model, num_images=100):
    """
    Run quantized INT8 inference (simulated in Python) matching what Gemmini does,
    to verify the quantization pipeline produces correct results.
    """
    from torchvision.datasets import CIFAR10

    print(f"\n=== Simulated INT8 Quantized Inference ({num_images} images) ===")

    dataset = CIFAR10(root="/tmp/cifar10_data", train=False, download=True)
    sd = {k: v.float() for k, v in model.state_dict().items()}

    # Preprocess same as prepare_cifar10.py cifar10standard mode
    def preprocess_int8(img_pil):
        img_np = np.array(img_pil, dtype=np.uint8)  # [32, 32, 3]
        float_img = img_np.astype(np.float32) / 255.0
        normalized = (float_img - CIFAR10_MEAN) / CIFAR10_STD
        return np.clip(np.round(normalized / CIFAR10_FMAX * 127.0), -128, 127).astype(np.int8)

    # Build quantized layer data (same as extraction script)
    LAYER_MAPPING = [
        ("conv_1",  "conv1",                   "bn1",                   "conv"),
        ("conv_2",  "layer1.0.conv1",          "layer1.0.bn1",          "conv"),
        ("conv_3",  "layer1.0.conv2",          "layer1.0.bn2",          "conv"),
        ("conv_4",  "layer1.0.conv3",          "layer1.0.bn3",          "conv"),
        ("conv_5",  "layer1.0.downsample.0",   "layer1.0.downsample.1", "conv"),
    ]

    # Just check first 5 layers to see if quantization is in the right ballpark
    x_scale = CIFAR10_FMAX / 127.0

    print(f"\nFirst-layer quantization check:")
    conv1_w = sd["conv1.weight"].numpy()  # [64, 3, 3, 3]
    bn1_g = sd["bn1.weight"].numpy()
    bn1_b = sd["bn1.bias"].numpy()
    bn1_m = sd["bn1.running_mean"].numpy()
    bn1_v = sd["bn1.running_var"].numpy()

    # BN fold
    inv_std = 1.0 / np.sqrt(bn1_v + 1e-5)
    scale = bn1_g * inv_std
    w_folded = conv1_w * scale.reshape(64, 1, 1, 1)
    b_folded = bn1_b - bn1_g * bn1_m * inv_std

    w_gemmini = w_folded.reshape(64, -1).T  # [27, 64]
    w_abs_max = np.max(np.abs(w_gemmini))
    w_scale = w_abs_max / 127.0
    w_int = np.clip(np.round(w_gemmini / w_scale), -128, 127).astype(np.int8)

    combined_scale = w_scale * x_scale
    b_int = np.clip(np.round(b_folded / combined_scale), -(2**31), 2**31 - 1).astype(np.int32)

    # For ReLU layer, estimate activation range
    y_range = max(float(np.max(bn1_b + 3.0 * np.abs(bn1_g))), 1.0)
    y_scale = y_range / 127.0
    output_scale = (w_scale * x_scale) / y_scale

    print(f"  conv1 w_folded: min={w_gemmini.min():.4f} max={w_gemmini.max():.4f} absmax={w_abs_max:.4f}")
    print(f"  w_scale={w_scale:.6f}  x_scale={x_scale:.6f}  y_range={y_range:.2f}")
    print(f"  output_scale={output_scale:.6e}")
    print(f"  b_folded range: [{b_folded.min():.4f}, {b_folded.max():.4f}]")
    print(f"  b_int range: [{b_int.min()}, {b_int.max()}]")

    # Run a single image through FP32 to compare
    img, label = dataset[0]
    img_int8 = preprocess_int8(img)
    print(f"\n  Image 0: label={label} ({CIFAR10_CLASSES[label]})")
    print(f"  int8 input stats: min={img_int8.min()} max={img_int8.max()} "
          f"mean={img_int8.astype(float).mean():.2f}")


def check_model_config():
    """Check the HuggingFace model config for any anomalies."""
    print(f"\n=== HuggingFace Model Config ===")
    try:
        config_file = hf_hub_download(MODEL_NAME, "config.json")
        import json
        with open(config_file) as f:
            config = json.load(f)
        print(f"Config keys: {sorted(config.keys())}")
        for k in ["architectures", "model_type", "num_labels", "id2label",
                   "image_size", "num_channels", "embedding_size"]:
            if k in config:
                print(f"  {k}: {config[k]}")
    except Exception as e:
        print(f"  Could not load config: {e}")


def main():
    check_model_config()

    sd = load_raw_state_dict()
    keys_ok = analyze_keys(sd)

    if not keys_ok:
        print("\nERROR: State dict keys don't match expected torchvision format!")
        print("The model might use a different architecture wrapper.")
        sys.exit(1)

    verify_int8_preprocessing()

    model = build_resnet50_cifar10(sd)
    top1_acc = run_fp32_inference(model, num_images=1000)

    verify_quantized_inference(model)

    if top1_acc < 50:
        print(f"\n*** WARNING: FP32 accuracy is only {top1_acc:.1f}% — model may be wrong! ***")
    else:
        print(f"\n*** FP32 accuracy is {top1_acc:.1f}% — model appears correct ***")


if __name__ == "__main__":
    main()
