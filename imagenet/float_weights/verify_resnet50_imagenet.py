#!/usr/bin/env python3
"""
ResNet50 ImageNet INT8 Verification — analogous to mobilenet_verify/

Three independent checks:
  1. ARCHITECTURE   — every LAYER_MAPPING entry exists in microsoft/resnet-50
                      with the correct tensor shape
  2. PREPROCESSING  — run the float model on the 4 images.h reference images
                      with THREE different input treatments:
                        A) x_float = x_int8              (current: x_scale_0 = 1.0)
                        B) x_float = (x_int8+128)/255    (raw [0,1], no normalization)
                        C) standard ImageNet normalization (correct for this model)
                      Check which method predicts the expected labels {75,900,125,897}
  3. WEIGHT COMPARE — parse current resnet50_params.h and compare weights/biases/
                      output_scales with a fresh re-extraction from HuggingFace;
                      also checks whether x_scale_0=1.0 or x_scale_0=128/127
                      gives a closer match

Usage:
    cd imagenet/float_weights
    conda run -n ImageNet python verify_resnet50_imagenet.py

Output:
    resnet50_verify_report.txt  — detailed report saved to float_weights/
"""

import os, sys, re
import numpy as np
import torch
from transformers import ResNetForImageClassification

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
IMAGENET_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
IMAGES_H     = os.path.join(IMAGENET_DIR, "images.h")
PARAMS_H     = os.path.join(IMAGENET_DIR, "resnet50_params.h")
REPORT_PATH  = os.path.join(SCRIPT_DIR,   "resnet50_verify_report.txt")

MODEL_NAME   = "microsoft/resnet-50"
BATCH_SIZE   = 4
NUM_CLASSES  = 1000

# Expected predictions for the 4 images in images.h (from A/B test printout)
EXPECTED_LABELS = [75, 900, 125, 897]

# ---------------------------------------------------------------------------
# LAYER_MAPPING (must match extract_resnet50_imagenet_int8.py exactly)
# ---------------------------------------------------------------------------
LAYER_MAPPING = [
    ("conv_1",  "resnet.embedder.embedder",                         "conv"),
    ("conv_2",  "resnet.encoder.stages.0.layers.0.layer.0",         "conv"),
    ("conv_3",  "resnet.encoder.stages.0.layers.0.layer.1",         "conv"),
    ("conv_4",  "resnet.encoder.stages.0.layers.0.layer.2",         "conv"),
    ("conv_5",  "resnet.encoder.stages.0.layers.0.shortcut",        "conv"),
    ("conv_6",  "resnet.encoder.stages.0.layers.1.layer.0",         "conv"),
    ("conv_7",  "resnet.encoder.stages.0.layers.1.layer.1",         "conv"),
    ("conv_8",  "resnet.encoder.stages.0.layers.1.layer.2",         "conv"),
    ("conv_9",  "resnet.encoder.stages.0.layers.2.layer.0",         "conv"),
    ("conv_10", "resnet.encoder.stages.0.layers.2.layer.1",         "conv"),
    ("conv_11", "resnet.encoder.stages.0.layers.2.layer.2",         "conv"),
    ("conv_12", "resnet.encoder.stages.1.layers.0.layer.0",         "conv"),
    ("conv_13", "resnet.encoder.stages.1.layers.0.layer.1",         "conv"),
    ("conv_14", "resnet.encoder.stages.1.layers.0.layer.2",         "conv"),
    ("conv_15", "resnet.encoder.stages.1.layers.0.shortcut",        "conv"),
    ("conv_16", "resnet.encoder.stages.1.layers.1.layer.0",         "conv"),
    ("conv_17", "resnet.encoder.stages.1.layers.1.layer.1",         "conv"),
    ("conv_18", "resnet.encoder.stages.1.layers.1.layer.2",         "conv"),
    ("conv_19", "resnet.encoder.stages.1.layers.2.layer.0",         "conv"),
    ("conv_20", "resnet.encoder.stages.1.layers.2.layer.1",         "conv"),
    ("conv_21", "resnet.encoder.stages.1.layers.2.layer.2",         "conv"),
    ("conv_22", "resnet.encoder.stages.1.layers.3.layer.0",         "conv"),
    ("conv_23", "resnet.encoder.stages.1.layers.3.layer.1",         "conv"),
    ("conv_24", "resnet.encoder.stages.1.layers.3.layer.2",         "conv"),
    ("conv_25", "resnet.encoder.stages.2.layers.0.layer.0",         "conv"),
    ("conv_26", "resnet.encoder.stages.2.layers.0.layer.1",         "conv"),
    ("conv_27", "resnet.encoder.stages.2.layers.0.layer.2",         "conv"),
    ("conv_28", "resnet.encoder.stages.2.layers.0.shortcut",        "conv"),
    ("conv_29", "resnet.encoder.stages.2.layers.1.layer.0",         "conv"),
    ("conv_30", "resnet.encoder.stages.2.layers.1.layer.1",         "conv"),
    ("conv_31", "resnet.encoder.stages.2.layers.1.layer.2",         "conv"),
    ("conv_32", "resnet.encoder.stages.2.layers.2.layer.0",         "conv"),
    ("conv_33", "resnet.encoder.stages.2.layers.2.layer.1",         "conv"),
    ("conv_34", "resnet.encoder.stages.2.layers.2.layer.2",         "conv"),
    ("conv_35", "resnet.encoder.stages.2.layers.3.layer.0",         "conv"),
    ("conv_36", "resnet.encoder.stages.2.layers.3.layer.1",         "conv"),
    ("conv_37", "resnet.encoder.stages.2.layers.3.layer.2",         "conv"),
    ("conv_38", "resnet.encoder.stages.2.layers.4.layer.0",         "conv"),
    ("conv_39", "resnet.encoder.stages.2.layers.4.layer.1",         "conv"),
    ("conv_40", "resnet.encoder.stages.2.layers.4.layer.2",         "conv"),
    ("conv_41", "resnet.encoder.stages.2.layers.5.layer.0",         "conv"),
    ("conv_42", "resnet.encoder.stages.2.layers.5.layer.1",         "conv"),
    ("conv_43", "resnet.encoder.stages.2.layers.5.layer.2",         "conv"),
    ("conv_44", "resnet.encoder.stages.3.layers.0.layer.0",         "conv"),
    ("conv_45", "resnet.encoder.stages.3.layers.0.layer.1",         "conv"),
    ("conv_46", "resnet.encoder.stages.3.layers.0.layer.2",         "conv"),
    ("conv_47", "resnet.encoder.stages.3.layers.0.shortcut",        "conv"),
    ("conv_48", "resnet.encoder.stages.3.layers.1.layer.0",         "conv"),
    ("conv_49", "resnet.encoder.stages.3.layers.1.layer.1",         "conv"),
    ("conv_50", "resnet.encoder.stages.3.layers.1.layer.2",         "conv"),
    ("conv_51", "resnet.encoder.stages.3.layers.2.layer.0",         "conv"),
    ("conv_52", "resnet.encoder.stages.3.layers.2.layer.1",         "conv"),
    ("conv_53", "resnet.encoder.stages.3.layers.2.layer.2",         "conv"),
    ("fc_54",   "classifier",                                        "fc"),
]

# ImageNet channel mean/std (for standard preprocessing)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------------------------------------------------------------------------
# Utility: parse images.h -> [4, 224, 224, 3] int8
# ---------------------------------------------------------------------------
def parse_images_h(path):
    """Extract the 4 reference images from images.h as int8 ndarray [4,224,224,3]."""
    print(f"  Parsing {path} ...", end=" ", flush=True)
    with open(path, "r") as f:
        text = f.read()
    # Find the array initialiser
    m = re.search(r'static const elem_t images\[4\]\[224\]\[224\]\[3\].*?=\s*\\?\s*(\{)', text, re.DOTALL)
    if not m:
        raise RuntimeError("Could not find images[] array in images.h")
    start = m.start(1)
    # Extract all integers (positive and negative)
    vals = np.array([int(x) for x in re.findall(r'-?\d+', text[start:start + 10_000_000])],
                    dtype=np.int8)
    # Should be 4 * 224 * 224 * 3 = 602112
    expected_n = 4 * 224 * 224 * 3
    if len(vals) < expected_n:
        raise RuntimeError(f"Expected {expected_n} values, got {len(vals)}")
    images = vals[:expected_n].reshape(4, 224, 224, 3)
    print(f"OK — shape {images.shape}, dtype {images.dtype}")
    return images  # HWC int8 layout, pixel-128 encoding


# ---------------------------------------------------------------------------
# Utility: parse resnet50_params.h -> {layer: {w, b, output_scale}}
# ---------------------------------------------------------------------------
def parse_params_h(path):
    """Parse resnet50_params.h into a dict of layer dicts with w, b, output_scale."""
    print(f"  Parsing {path} ...", end=" ", flush=True)
    with open(path, "r") as f:
        text = f.read()

    layers = {}

    # Weight arrays: elem_t NAME_w[...] = {...};
    for m in re.finditer(
            r'static const elem_t (\w+)_w\[.*?\] row_align\(1\) = (\{[^;]+\});',
            text, re.DOTALL):
        name, raw = m.group(1), m.group(2)
        vals = [int(x) for x in re.findall(r'-?\d+', raw)]
        layers.setdefault(name, {})["w"] = vals

    # Bias arrays: acc_t NAME_b[...] = {...};
    for m in re.finditer(
            r'static const acc_t (\w+)_b\[.*?\] row_align_acc\(1\) = (\{[^;]+\});',
            text, re.DOTALL):
        name, raw = m.group(1), m.group(2)
        vals = [int(x) for x in re.findall(r'-?\d+', raw)]
        layers.setdefault(name, {})["b"] = vals

    # Output scale from ConvParams / FcParams
    for m in re.finditer(r'(\w+?)_params\s*=\s*\{[^}]*\.output_scale=([^,}]+)[,}]', text):
        name, os_str = m.group(1), m.group(2).strip()
        try:
            scale = float(os_str.rstrip("f"))
        except Exception:
            scale = None
        layers.setdefault(name, {})["output_scale"] = scale

    print(f"OK — {len(layers)} layers found")
    return layers


# ---------------------------------------------------------------------------
# Utility: INT8 quantise weight tensor
# ---------------------------------------------------------------------------
def quantize_weight_int8(w_float):
    w_max = np.max(np.abs(w_float))
    if w_max < 1e-10:
        return np.zeros_like(w_float, dtype=np.int8), 1.0
    w_scale = w_max / 127.0
    w_int = np.clip(np.round(w_float / w_scale), -128, 127).astype(np.int8)
    return w_int, float(w_scale)


def quantize_bias_int32(b_float, scale):
    if scale < 1e-30:
        return np.zeros_like(b_float, dtype=np.int32)
    return np.clip(np.round(b_float / scale), -(2**31), 2**31 - 1).astype(np.int32)


def compute_output_scale(w_scale, x_scale, y_range):
    y_scale = y_range / 127.0
    os = (w_scale * x_scale) / y_scale
    return os, f"{os:.8e}f"


# ---------------------------------------------------------------------------
# BN folding (same as extract script)
# ---------------------------------------------------------------------------
def fold_bn(conv_weight, bn_weight, bn_bias, bn_mean, bn_var, eps=1e-5):
    inv_std = 1.0 / np.sqrt(bn_var + eps)
    scale = bn_weight * inv_std
    shape = [conv_weight.shape[0]] + [1] * (conv_weight.ndim - 1)
    w_folded = conv_weight * scale.reshape(shape)
    b_folded = bn_bias - bn_weight * bn_mean * inv_std
    return w_folded, b_folded


def reshape_conv_weight(w):
    out_ch = w.shape[0]
    if w.ndim == 4:
        w = w.transpose(0, 2, 3, 1)
    return w.reshape(out_ch, -1).T   # [patch_size, out_ch]


def get_conv_bn_params(state_dict, prefix):
    conv_w = state_dict[f"{prefix}.convolution.weight"].float().numpy()
    gamma  = state_dict[f"{prefix}.normalization.weight"].float().numpy()
    beta   = state_dict[f"{prefix}.normalization.bias"].float().numpy()
    mean   = state_dict[f"{prefix}.normalization.running_mean"].float().numpy()
    var    = state_dict[f"{prefix}.normalization.running_var"].float().numpy()
    return conv_w, gamma, beta, mean, var


def extract_and_fold(state_dict, hf_prefix):
    conv_w, gamma, beta, mean, var = get_conv_bn_params(state_dict, hf_prefix)
    w_folded, b_folded = fold_bn(conv_w, gamma, beta, mean, var)
    return reshape_conv_weight(w_folded), b_folded, gamma, beta


def estimate_activation_range(bn_gamma, bn_beta, has_relu):
    if has_relu:
        return float(max(np.max(bn_beta + 3 * np.abs(bn_gamma)), 0.001))
    else:
        return float(max(np.max(np.abs(bn_beta)) + 3 * np.max(np.abs(bn_gamma)), 0.001))


# ---------------------------------------------------------------------------
# CHECK 1: Architecture verification
# ---------------------------------------------------------------------------
def check_architecture(state_dict, report):
    report.append("\n" + "="*70)
    report.append("CHECK 1: Architecture (LAYER_MAPPING vs HuggingFace state dict)")
    report.append("="*70)

    errors = 0
    for gemmini_name, hf_prefix, layer_type in LAYER_MAPPING:
        if layer_type == "fc":
            # Check FC weight
            w_key = "classifier.1.weight"
            b_key = "classifier.1.bias"
            missing = [k for k in (w_key, b_key) if k not in state_dict]
            if missing:
                report.append(f"  MISSING FC keys: {missing}")
                errors += 1
            else:
                w_shape = tuple(state_dict[w_key].shape)
                b_shape = tuple(state_dict[b_key].shape)
                report.append(f"  fc_54       OK  W={w_shape}  b={b_shape}")
        else:
            keys_needed = [
                f"{hf_prefix}.convolution.weight",
                f"{hf_prefix}.normalization.weight",
                f"{hf_prefix}.normalization.bias",
                f"{hf_prefix}.normalization.running_mean",
                f"{hf_prefix}.normalization.running_var",
            ]
            missing = [k for k in keys_needed if k not in state_dict]
            if missing:
                report.append(f"  MISSING {gemmini_name} ({hf_prefix}): {missing}")
                errors += 1
            else:
                w_shape = tuple(state_dict[f"{hf_prefix}.convolution.weight"].shape)
                report.append(f"  {gemmini_name:10s}  OK  conv_w={w_shape}")

    if errors == 0:
        report.append("\n  RESULT: All 54 layers found in HuggingFace state dict ✓")
    else:
        report.append(f"\n  RESULT: {errors} MISSING layers — LAYER_MAPPING is WRONG")
    return errors


# ---------------------------------------------------------------------------
# CHECK 2: Preprocessing — float model on images.h reference data
# ---------------------------------------------------------------------------
def check_preprocessing(model, images_int8, report):
    report.append("\n" + "="*70)
    report.append("CHECK 2: Preprocessing — float model on images.h (4 reference images)")
    report.append(f"  Expected top-1 labels: {EXPECTED_LABELS}")
    report.append("="*70)

    # images_int8: [4, 224, 224, 3] int8  (pixel - 128 encoding)
    images_i32 = images_int8.astype(np.float32)    # keep as [-128, 127]

    # Method A: x_float = (pixel - 128)  — what x_scale_0=1.0 implies
    x_a = torch.from_numpy(images_i32.transpose(0, 3, 1, 2))  # [4,3,224,224]

    # Method B: x_float = (pixel - 128 + 128) / 255 = pixel / 255  ∈ [0,1]
    x_b_np = (images_i32 + 128.0) / 255.0  # [4,224,224,3] in [0,1]
    x_b = torch.from_numpy(x_b_np.transpose(0, 3, 1, 2))

    # Method C: standard ImageNet normalization — (pixel/255 - mean) / std
    x_c_np = (x_b_np - IMAGENET_MEAN) / IMAGENET_STD   # broadcast over [H,W,3]
    x_c = torch.from_numpy(x_c_np.transpose(0, 3, 1, 2))

    model.eval()
    with torch.no_grad():
        logits_a = model(pixel_values=x_a).logits.numpy()   # [4, 1000]
        logits_b = model(pixel_values=x_b).logits.numpy()
        logits_c = model(pixel_values=x_c).logits.numpy()

    def top3(logits_row):
        idx = np.argsort(logits_row)[::-1][:3]
        return [(int(i), float(logits_row[i])) for i in idx]

    methods = [
        ("A  pixel-128 as float   (x_scale_0=1.0, current)", logits_a),
        ("B  pixel/255 ∈[0,1]    (no norm, x_scale_0=1/255)", logits_b),
        ("C  standard ImageNet   CORRECT for microsoft/resnet-50", logits_c),
    ]

    for method_name, logits in methods:
        report.append(f"\n  Method {method_name}")
        top1_correct = 0
        for i in range(4):
            t3 = top3(logits[i])
            pred = t3[0][0]
            expected = EXPECTED_LABELS[i]
            ok = "✓" if pred == expected else "✗"
            report.append(f"    Image {i}: pred={pred:4d} (score={t3[0][1]:+.2f})  "
                          f"expected={expected}  {ok}  | "
                          f"top3={[c for c,_ in t3]}")
            if pred == expected:
                top1_correct += 1
        report.append(f"    Top-1 correct: {top1_correct}/4")

    # Also check what the float logit range looks like for Method C
    report.append(f"\n  [Method C logit statistics for sanity]")
    for i in range(4):
        l = logits_c[i]
        report.append(f"    Image {i}: logit min={l.min():.2f}  max={l.max():.2f}  "
                      f"std={l.std():.2f}")

    report.append(f"\n  [Method A logit statistics — what INT8 model tries to approximate]")
    for i in range(4):
        l = logits_a[i]
        report.append(f"    Image {i}: logit min={l.min():.2f}  max={l.max():.2f}  "
                      f"std={l.std():.2f}")


# ---------------------------------------------------------------------------
# CHECK 3: Weight/scale comparison — params.h vs fresh re-extraction
# ---------------------------------------------------------------------------
LAYER_ACTIVATION = {
    "conv_1":  True,
    "conv_2":  True,  "conv_3":  True,  "conv_4":  False, "conv_5":  False,
    "conv_6":  True,  "conv_7":  True,  "conv_8":  False,
    "conv_9":  True,  "conv_10": True,  "conv_11": False,
    "conv_12": True,  "conv_13": True,  "conv_14": False, "conv_15": False,
    "conv_16": True,  "conv_17": True,  "conv_18": False,
    "conv_19": True,  "conv_20": True,  "conv_21": False,
    "conv_22": True,  "conv_23": True,  "conv_24": False,
    "conv_25": True,  "conv_26": True,  "conv_27": False, "conv_28": False,
    "conv_29": True,  "conv_30": True,  "conv_31": False,
    "conv_32": True,  "conv_33": True,  "conv_34": False,
    "conv_35": True,  "conv_36": True,  "conv_37": False,
    "conv_38": True,  "conv_39": True,  "conv_40": False,
    "conv_41": True,  "conv_42": True,  "conv_43": False,
    "conv_44": True,  "conv_45": True,  "conv_46": False, "conv_47": False,
    "conv_48": True,  "conv_49": True,  "conv_50": False,
    "conv_51": True,  "conv_52": True,  "conv_53": False,
}

SAVE_XSCALE_BEFORE = {
    "conv_2":  "conv_5",
    "conv_12": "conv_15",
    "conv_25": "conv_28",
    "conv_44": "conv_47",
}


def reextract_layers(state_dict, x_scale_0):
    """Re-run extraction logic and return {name: {w_int, b_int, output_scale}}."""
    layers = {}
    x_scale = x_scale_0
    shortcut_x_scale = {}

    for gemmini_name, hf_prefix, layer_type in LAYER_MAPPING:
        if gemmini_name in SAVE_XSCALE_BEFORE:
            shortcut_x_scale[SAVE_XSCALE_BEFORE[gemmini_name]] = x_scale
        eff_x = shortcut_x_scale.pop(gemmini_name, x_scale)

        if layer_type == "fc":
            fc_w_float = state_dict["classifier.1.weight"].float().numpy()
            fc_b_float = state_dict["classifier.1.bias"].float().numpy()
            w_gemmini  = fc_w_float.T
            w_int, w_scale = quantize_weight_int8(w_gemmini)
            b_int_1d = quantize_bias_int32(fc_b_float, w_scale * x_scale)
            b_int_2d = np.tile(b_int_1d.reshape(1, -1), (BATCH_SIZE, 1))
            fc_y_range = max(
                float(np.max(np.abs(fc_b_float))
                      + np.std(fc_w_float) * np.sqrt(float(fc_w_float.shape[1])) * x_scale * 3),
                5.0
            )
            os_val, _ = compute_output_scale(w_scale, x_scale, fc_y_range)
            layers[gemmini_name] = {
                "w": w_int.flatten().tolist(),
                "b": b_int_2d.flatten().tolist(),
                "output_scale": os_val,
            }
        else:
            w_float, b_float, bn_gamma, bn_beta = extract_and_fold(state_dict, hf_prefix)
            has_relu = LAYER_ACTIVATION.get(gemmini_name, False)
            w_int, w_scale = quantize_weight_int8(w_float)
            b_int = quantize_bias_int32(b_float, w_scale * eff_x)
            y_range = estimate_activation_range(bn_gamma, bn_beta, has_relu)
            os_val, _ = compute_output_scale(w_scale, eff_x, y_range)
            layers[gemmini_name] = {
                "w": w_int.flatten().tolist(),
                "b": b_int.flatten().tolist(),
                "output_scale": os_val,
            }
            x_scale = y_range / 127.0

    return layers


def compare_layers(ref, gen, label, report):
    report.append(f"\n  --- Variant: {label} ---")
    total_w_match, total_w_total = 0, 0
    total_b_close, total_b_far   = 0, 0
    discrepancies = []

    for name in sorted(ref.keys()):
        if name not in gen:
            report.append(f"    {name}: MISSING in re-extracted set")
            continue
        r, g = ref[name], gen[name]

        # Weights
        rw = np.array(r.get("w", []), dtype=np.int8)
        gw = np.array(g.get("w", []), dtype=np.int8)
        if rw.size > 0 and gw.size == rw.size:
            exact = int(np.sum(rw == gw))
            diff  = int(np.sum(rw != gw))
            max_d = int(np.max(np.abs(rw.astype(np.int16) - gw.astype(np.int16))))
            w_flag = f"✓ all exact" if diff == 0 else \
                     f"✗ {diff}/{rw.size} differ (max_diff={max_d})"
            total_w_match += exact; total_w_total += rw.size
        else:
            w_flag = f"shape mismatch ref={rw.size} gen={gw.size}"

        # Biases
        rb = np.array(r.get("b", []), dtype=np.int64)
        gb = np.array(g.get("b", []), dtype=np.int64)
        if rb.size > 0 and gb.size == rb.size:
            b_d   = np.abs(rb - gb)
            b_max = int(b_d.max())
            thr   = max(1, int(np.abs(rb).max() * 0.01))
            close = int(np.sum(b_d <= thr))
            b_flag = f"{100*close//rb.size}% within 1% (max_diff={b_max})"
            total_b_close += close; total_b_far += (rb.size - close)
        else:
            b_flag = "shape mismatch"

        # Scale
        ros = r.get("output_scale")
        gos = g.get("output_scale")
        if ros is not None and gos is not None:
            if abs(ros - gos) < 1e-9:
                s_flag = "✓ exact"
            elif abs(ros - gos) / max(abs(ros), 1e-9) < 0.02:
                s_flag = "≈ within 2%"
            else:
                ratio = gos / ros if ros != 0 else float("inf")
                s_flag = f"✗ ref={ros:.4e}  gen={gos:.4e}  (ratio={ratio:.3f})"
                discrepancies.append((name, ros, gos, ratio))
        else:
            s_flag = "missing"

        report.append(f"    {name:12s}  W:{w_flag:38s}  B:{b_flag:30s}  scale:{s_flag}")

    total_w = total_w_total
    report.append(f"\n    WEIGHT SUMMARY: {total_w_match}/{total_w} exact "
                  f"({100.0*total_w_match/max(total_w,1):.1f}%)")
    report.append(f"    BIAS   SUMMARY: {total_b_close} within 1%  ({total_b_far} further)")

    if discrepancies:
        report.append(f"\n    SCALE DISCREPANCIES (ratio far from 1.0):")
        for name, ros, gos, ratio in sorted(discrepancies, key=lambda x: abs(x[3]-1), reverse=True)[:10]:
            report.append(f"      {name:12s}  ref={ros:.4e}  gen={gos:.4e}  ratio={ratio:.3f}")

    return total_w_match, total_w


def check_weights(state_dict, params_path, report):
    report.append("\n" + "="*70)
    report.append("CHECK 3: Weight/scale comparison — resnet50_params.h vs fresh extraction")
    report.append("="*70)

    ref = parse_params_h(params_path)
    report.append(f"  Reference header: {params_path}")
    report.append(f"  Layers in reference: {len(ref)}")

    variants = [
        ("x_scale_0=1.0   (current extraction)", 1.0),
        ("x_scale_0=128/127 (symmetric INT8)",   128.0 / 127.0),
    ]

    best_pct    = -1.0
    best_label  = None

    for label, x_scale_0 in variants:
        print(f"  Re-extracting with {label} ...", end=" ", flush=True)
        gen = reextract_layers(state_dict, x_scale_0)
        print("done")
        m, t = compare_layers(ref, gen, label, report)
        pct = 100.0 * m / max(t, 1)
        if pct > best_pct:
            best_pct   = pct
            best_label = label

    report.append(f"\n  BEST MATCH: {best_label}  ({best_pct:.1f}% exact weight match)")

    # Spot-check: what is the fc_54 output_scale in the reference?
    if "fc_54" in ref:
        report.append(f"\n  fc_54 output_scale in params.h: {ref['fc_54'].get('output_scale', 'N/A'):.6e}")
        # Expected range: INT8 logit = 5 should map to float ≈ 5 * fc_y_range/127
        os_val = ref["fc_54"].get("output_scale", 0)
        if os_val > 0:
            repr_logit_5 = round(5.0 / os_val)
            report.append(f"  Float logit 5.0 → INT8 output ≈ {repr_logit_5}  "
                          f"(ideal: ~50-70, terrible if ≤5)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    report = ["ResNet50 ImageNet INT8 Verification Report",
              f"Model: {MODEL_NAME}",
              ""]

    print(f"Loading {MODEL_NAME} from HuggingFace ...")
    model = ResNetForImageClassification.from_pretrained(MODEL_NAME)
    model.eval()
    state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    n_classes = state_dict["classifier.1.weight"].shape[0]
    report.append(f"Model loaded: {n_classes} output classes")
    print(f"Model loaded ({n_classes} classes)")

    # Check 1: Architecture
    print("\n[Check 1] Architecture ...")
    arch_errors = check_architecture(state_dict, report)

    # Check 2: Preprocessing (requires images.h)
    print("\n[Check 2] Preprocessing ...")
    if not os.path.exists(IMAGES_H):
        report.append(f"\n  SKIP: {IMAGES_H} not found")
        print(f"  SKIP: images.h not found at {IMAGES_H}")
    else:
        images_int8 = parse_images_h(IMAGES_H)
        check_preprocessing(model, images_int8, report)

    # Check 3: Weight comparison
    print("\n[Check 3] Weight comparison ...")
    if not os.path.exists(PARAMS_H):
        report.append(f"\n  SKIP: {PARAMS_H} not found")
        print(f"  SKIP: resnet50_params.h not found at {PARAMS_H}")
    else:
        check_weights(state_dict, PARAMS_H, report)

    # Write report
    report_text = "\n".join(report)
    with open(REPORT_PATH, "w") as f:
        f.write(report_text)
    print(f"\nReport saved to: {REPORT_PATH}")
    print("\n" + report_text)


if __name__ == "__main__":
    main()
