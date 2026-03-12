#!/usr/bin/env python3
"""
reference_inference.py — Run BERT-Tiny (cased, fine-tuned on SST-2) on the
full SST-2 validation split and report accuracy.

Expected accuracy: ~82–84 %

Usage:
    conda run -n ImageNet python reference_inference.py
"""

import sys
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset

MODEL_NAME = "M-FAC/bert-tiny-finetuned-sst2"
MAX_LEN = 128

# ── Load model & tokeniser ──────────────────────────────────────────
print(f"Loading model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

# ── Quick sanity check on a known sentence ──────────────────────────
sanity_text = "This movie is absolutely wonderful!"
sanity_inputs = tokenizer(sanity_text, return_tensors="pt",
                          truncation=True, max_length=MAX_LEN,
                          padding="max_length")
with torch.no_grad():
    sanity_logits = model(**sanity_inputs).logits

sanity_pred = torch.argmax(sanity_logits, dim=-1).item()
label_map = {0: "negative", 1: "positive"}
print(f"Sanity check: \"{sanity_text}\"")
print(f"  Prediction : {label_map.get(sanity_pred, sanity_pred)}")
print(f"  Raw logits : {sanity_logits.tolist()}")
assert sanity_pred == 1, (
    f"Sanity check FAILED — expected positive (1), got {sanity_pred}. "
    "Model may not have loaded correctly."
)
print("  ✓ Sanity check passed.\n")

# ── Evaluate on SST-2 validation split ──────────────────────────────
print("Loading SST-2 validation split …")
dataset = load_dataset("glue", "sst2", split="validation")
total = len(dataset)
print(f"  {total} examples.\n")

correct = 0
for i, item in enumerate(dataset):
    inputs = tokenizer(item["sentence"], return_tensors="pt",
                       truncation=True, max_length=MAX_LEN,
                       padding="max_length")
    with torch.no_grad():
        logits = model(**inputs).logits
    pred = torch.argmax(logits, dim=-1).item()
    correct += int(pred == item["label"])

    # Progress every 100 examples
    if (i + 1) % 100 == 0 or (i + 1) == total:
        pct = correct / (i + 1) * 100
        print(f"  [{i+1:>4}/{total}]  running accuracy = {pct:.2f}%")

accuracy = correct / total * 100
print(f"\n{'='*40}")
print(f"Final Accuracy: {correct}/{total} = {accuracy:.2f}%")
print(f"{'='*40}")

# ── Final verification gate ─────────────────────────────────────────
if accuracy < 75.0:
    print("\n⚠  WARNING: Accuracy is below 75 %. Something may be wrong "
          "(wrong model, tokeniser mismatch, etc.).")
    sys.exit(1)
else:
    print("\n✓ Accuracy looks reasonable for BERT-Tiny on SST-2.")
