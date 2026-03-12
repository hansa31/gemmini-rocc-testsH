#!/usr/bin/env python3
"""
export_weights.py — Download BERT-Tiny (cased, SST-2) and export every
weight tensor as an individual .npy file under weights/.

Also generates:
  weights/manifest.txt   — human-readable list of names, shapes, dtypes
  weights/arch_info.json — key architecture numbers for the C code

Usage:
    conda run -n ImageNet python export_weights.py
"""

import json
import os
import sys

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

MODEL_NAME = "M-FAC/bert-tiny-finetuned-sst2"
WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")

# ── Load model ──────────────────────────────────────────────────────
print(f"Loading model: {MODEL_NAME}")
config = AutoConfig.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

# ── Verify architecture matches BERT-Tiny expectations ──────────────
print("\n=== Architecture Verification ===")
expected = {
    "num_hidden_layers": 2,
    "hidden_size": 128,
    "num_attention_heads": 2,
    "intermediate_size": 512,
}

all_ok = True
for key, exp_val in expected.items():
    actual = getattr(config, key, None)
    status = "✓" if actual == exp_val else "✗ MISMATCH"
    if actual != exp_val:
        all_ok = False
    print(f"  {key}: expected={exp_val}, actual={actual}  {status}")

if not all_ok:
    print("\n⚠  Architecture does not match BERT-Tiny. Aborting.")
    sys.exit(1)
print("  All architecture checks passed.\n")

# ── Export weights ──────────────────────────────────────────────────
os.makedirs(WEIGHTS_DIR, exist_ok=True)

state = model.state_dict()
manifest_lines = []

print(f"Exporting {len(state)} tensors to {WEIGHTS_DIR}/\n")
print(f"{'Tensor Name':<65} {'Shape':<25} {'dtype'}")
print("-" * 100)

for name, tensor in state.items():
    safe_name = name.replace("/", "_").replace(".", "_")
    arr = tensor.cpu().numpy()
    npy_path = os.path.join(WEIGHTS_DIR, f"{safe_name}.npy")
    np.save(npy_path, arr)

    shape_str = str(list(arr.shape))
    print(f"  {name:<63} {shape_str:<25} {arr.dtype}")
    manifest_lines.append(f"{safe_name}.npy  {name}  {shape_str}  {arr.dtype}")

# ── Write manifest ──────────────────────────────────────────────────
manifest_path = os.path.join(WEIGHTS_DIR, "manifest.txt")
with open(manifest_path, "w") as f:
    f.write("# file_name  original_name  shape  dtype\n")
    for line in manifest_lines:
        f.write(line + "\n")
print(f"\nManifest written to {manifest_path}")

# ── Write architecture info for C code ──────────────────────────────
arch_info = {
    "model_name": MODEL_NAME,
    "num_layers": config.num_hidden_layers,
    "hidden_dim": config.hidden_size,
    "num_attention_heads": config.num_attention_heads,
    "ffn_intermediate_size": config.intermediate_size,
    "head_dim": config.hidden_size // config.num_attention_heads,
    "vocab_size": config.vocab_size,
    "max_position_embeddings": config.max_position_embeddings,
    "num_labels": config.num_labels,
    "hidden_act": config.hidden_act,
}
arch_path = os.path.join(WEIGHTS_DIR, "arch_info.json")
with open(arch_path, "w") as f:
    json.dump(arch_info, f, indent=2)
print(f"Architecture info written to {arch_path}")

# ── Post-export verification ────────────────────────────────────────
print("\n=== Post-Export Verification ===")
npy_files = [f for f in os.listdir(WEIGHTS_DIR) if f.endswith(".npy")]
print(f"  .npy files saved: {len(npy_files)}")
print(f"  state_dict keys:  {len(state)}")
assert len(npy_files) == len(state), (
    f"Mismatch: {len(npy_files)} files vs {len(state)} state_dict entries!"
)

# Spot-check: reload a few weights and compare
spot_checks = list(state.keys())[:3]
for name in spot_checks:
    safe_name = name.replace("/", "_").replace(".", "_")
    loaded = np.load(os.path.join(WEIGHTS_DIR, f"{safe_name}.npy"))
    original = state[name].cpu().numpy()
    if np.array_equal(loaded, original):
        print(f"  ✓ {name} — reload matches")
    else:
        print(f"  ✗ {name} — RELOAD MISMATCH!")
        sys.exit(1)

total_bytes = sum(
    os.path.getsize(os.path.join(WEIGHTS_DIR, f)) for f in npy_files
)
print(f"\n  Total weight data: {total_bytes / 1024:.1f} KB")
print("\n✓ Export complete. Weights are ready for C integration.")
