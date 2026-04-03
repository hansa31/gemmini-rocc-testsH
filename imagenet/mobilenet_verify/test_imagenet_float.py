#!/usr/bin/env python3
"""
Verify ImageNet MobileNetV2 (google/mobilenet_v2_1.0_224) float pipeline.

Compares:
  1. HuggingFace reference model inference (gold standard)
  2. Manual pipeline with BN-folded weights + ReLU6 (what the Gemmini code does)
  3. Manual pipeline with BN-folded weights + plain ReLU (to show ReLU6 is needed)

Uses 4 ImageNet validation images (or synthetic if unavailable).
"""

import numpy as np
import torch
import torch.nn.functional as F
from transformers import MobileNetV2ForImageClassification, AutoImageProcessor

MODEL_NAME = "google/mobilenet_v2_1.0_224"
BN_EPS = 1e-3  # Actual MobileNetV2 BN epsilon (model uses eps=0.001)
BN_DEAD_VAR_THRESHOLD = 1e-3


def fold_bn(conv_w, bn_w, bn_b, bn_mean, bn_var, eps=BN_EPS):
    inv_std = 1.0 / np.sqrt(bn_var + eps)
    scale = bn_w * inv_std
    shape = [conv_w.shape[0]] + [1] * (conv_w.ndim - 1)
    w = conv_w * scale.reshape(shape)
    b = bn_b - bn_w * bn_mean * inv_std
    dead = bn_var < BN_DEAD_VAR_THRESHOLD
    if np.any(dead):
        w[dead] = 0.0
        b[dead] = 0.0
    return w, b


def get_bn_params(sd, prefix):
    return (
        sd[f"{prefix}.convolution.weight"].numpy(),
        sd[f"{prefix}.normalization.weight"].numpy(),
        sd[f"{prefix}.normalization.bias"].numpy(),
        sd[f"{prefix}.normalization.running_mean"].numpy(),
        sd[f"{prefix}.normalization.running_var"].numpy(),
    )


def conv2d(x, w, b, stride=1, padding=0, groups=1):
    x_t = torch.from_numpy(x).float()
    w_t = torch.from_numpy(w).float()
    b_t = torch.from_numpy(b).float()
    out = F.conv2d(x_t, w_t, b_t, stride=stride, padding=padding, groups=groups)
    return out.numpy()


def manual_inference(sd, img_np, use_relu6=True):
    """Run MobileNetV2 manually with BN-folded weights.
    img_np: [1, 3, H, W] numpy array, already normalized.
    """
    act_fn = (lambda x: np.clip(x, 0, 6)) if use_relu6 else (lambda x: np.maximum(x, 0))
    act_name = "relu6" if use_relu6 else "relu"

    # Track activations
    stats = {}

    # Stem
    w, b = fold_bn(*get_bn_params(sd, "mobilenet_v2.conv_stem.first_conv"))
    x = conv2d(img_np, w, b, stride=2, padding=1)
    x = act_fn(x)
    stats["conv_1"] = float(np.max(np.abs(x)))

    w, b = fold_bn(*get_bn_params(sd, "mobilenet_v2.conv_stem.conv_3x3"))
    x = conv2d(x, w, b, stride=1, padding=1, groups=w.shape[0])
    x = act_fn(x)
    stats["conv_dw_2"] = float(np.max(np.abs(x)))

    w, b = fold_bn(*get_bn_params(sd, "mobilenet_v2.conv_stem.reduce_1x1"))
    x = conv2d(x, w, b)
    stats["conv_3"] = float(np.max(np.abs(x)))

    # Inverted residual blocks
    BLOCK_CONFIG = [
        # (expand_ratio, out_ch, repeats, stride)
        (6, 24, 2, 2),   # layer 0-1
        (6, 32, 3, 2),   # layer 2-4
        (6, 64, 4, 2),   # layer 5-8
        (6, 96, 3, 1),   # layer 9-11
        (6, 160, 3, 2),  # layer 12-14
        (6, 320, 1, 1),  # layer 15
    ]

    hf_layer_idx = 0
    gemmini_idx = 4

    for t, out_c, repeats, first_stride in BLOCK_CONFIG:
        for rep in range(repeats):
            stride = first_stride if rep == 0 else 1
            residual = x if (stride == 1 and x.shape[1] == out_c) else None

            prefix = f"mobilenet_v2.layer.{hf_layer_idx}"

            # Expand 1x1
            w, b = fold_bn(*get_bn_params(sd, f"{prefix}.expand_1x1"))
            x = conv2d(x, w, b)
            x = act_fn(x)
            stats[f"conv_{gemmini_idx}"] = float(np.max(np.abs(x)))
            gemmini_idx += 1

            # Depthwise 3x3
            w, b = fold_bn(*get_bn_params(sd, f"{prefix}.conv_3x3"))
            x = conv2d(x, w, b, stride=stride, padding=1, groups=w.shape[0])
            x = act_fn(x)
            stats[f"conv_dw_{gemmini_idx}"] = float(np.max(np.abs(x)))
            gemmini_idx += 1

            # Project 1x1 (no activation)
            w, b = fold_bn(*get_bn_params(sd, f"{prefix}.reduce_1x1"))
            x = conv2d(x, w, b)
            stats[f"conv_{gemmini_idx}"] = float(np.max(np.abs(x)))

            if residual is not None:
                x = x + residual

            gemmini_idx += 1
            hf_layer_idx += 1

    # Final 1x1
    w, b = fold_bn(*get_bn_params(sd, "mobilenet_v2.conv_1x1"))
    x = conv2d(x, w, b)
    x = act_fn(x)
    stats["conv_52"] = float(np.max(np.abs(x)))

    # Global average pool
    x = np.mean(x, axis=(2, 3), keepdims=True)  # [1, 1280, 1, 1]

    # FC
    fc_w = sd["classifier.weight"].numpy()
    fc_b = sd["classifier.bias"].numpy()
    num_classes = fc_w.shape[0]
    if num_classes == 1001:
        fc_w = fc_w[1:, :]
        fc_b = fc_b[1:]
    logits = x.reshape(1, -1) @ fc_w.T + fc_b  # [1, 1000]

    return logits[0], stats


def main():
    print(f"Loading {MODEL_NAME}...")
    model = MobileNetV2ForImageClassification.from_pretrained(MODEL_NAME)
    model.eval()
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    print(f"image_mean={processor.image_mean}, image_std={processor.image_std}")

    # Generate test images
    rng = np.random.RandomState(42)
    from PIL import Image
    test_images = []
    for _ in range(4):
        arr = rng.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        test_images.append(Image.fromarray(arr))

    print(f"\n{'='*70}")
    print("Running verification with 4 synthetic test images")
    print(f"{'='*70}")

    for img_idx, img in enumerate(test_images):
        print(f"\n--- Image {img_idx} ---")

        # HuggingFace reference
        inputs = processor(images=img, return_tensors="pt")
        with torch.no_grad():
            hf_out = model(**inputs)
        hf_logits = hf_out.logits[0].numpy()
        if hf_logits.shape[0] == 1001:
            hf_logits = hf_logits[1:]
        hf_pred = int(np.argmax(hf_logits))

        # Manual with ReLU6
        img_np = np.array(img, dtype=np.float32) / 255.0
        mean = np.array(processor.image_mean).reshape(1, 3, 1, 1)
        std = np.array(processor.image_std).reshape(1, 3, 1, 1)
        img_norm = (img_np.transpose(2, 0, 1)[np.newaxis] - mean) / std

        relu6_logits, relu6_stats = manual_inference(sd, img_norm, use_relu6=True)
        relu6_pred = int(np.argmax(relu6_logits))

        relu_logits, relu_stats = manual_inference(sd, img_norm, use_relu6=False)
        relu_pred = int(np.argmax(relu_logits))

        # Compare
        if hf_logits.shape == relu6_logits.shape:
            corr_relu6 = float(np.corrcoef(hf_logits, relu6_logits)[0, 1])
            corr_relu = float(np.corrcoef(hf_logits, relu_logits)[0, 1])
        else:
            corr_relu6 = corr_relu = float('nan')

        print(f"  HF reference:   pred={hf_pred}")
        print(f"  Manual (ReLU6): pred={relu6_pred}, logit_corr={corr_relu6:.4f}")
        print(f"  Manual (ReLU):  pred={relu_pred},  logit_corr={corr_relu:.4f}")

        # Show activation stats
        print(f"  Activation stats (ReLU6 vs ReLU):")
        for layer in ["conv_1", "conv_dw_2", "conv_3", "conv_52"]:
            r6 = relu6_stats.get(layer, 0)
            r = relu_stats.get(layer, 0)
            print(f"    {layer:15s}: relu6_max={r6:.1f}, relu_max={r:.1f}")

    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")
    print("If Manual(ReLU6) predictions match HF reference and correlations")
    print("are > 0.9, the architecture and weights are correct.")
    print("If Manual(ReLU) diverges, it confirms ReLU6 clamping is essential.")


if __name__ == "__main__":
    main()
