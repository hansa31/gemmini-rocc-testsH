#!/usr/bin/env python3
"""
Prepare SST-2 validation data for BERT-Tiny Gemmini inference.

Preprocessing pipeline:
  1. Load SST-2 validation split from HuggingFace datasets (872 labelled examples)
  2. Tokenize with M-FAC/bert-tiny-finetuned-sst2 tokenizer (max_length=128, padding)
  3. Compute embeddings from exported .npy weights:
       emb = word_emb[input_ids] + pos_emb[positions] + type_emb[token_type_ids]
       emb = LayerNorm(emb, gamma, beta)
  4. Quantize embeddings to int8 (symmetric per-tensor, same as generate_bert_params.py)
  5. Write output files:
       sst2_val_<N>.bin             — N × SEQ_LEN × HIDDEN_DIM int8 (embedded sequences)
       sst2_val_<N>_labels.txt      — N lines, one label per line (0 or 1)
       sst2_val_<N>_attn_masks.bin  — N × SEQ_LEN uint8 (1=real token, 0=padding)

Output is compatible with bert-tiny-sst2-stream.c which reads one example at
a time and computes accuracy + F1 over the full validation set.

Usage:
  python prepare_sst2.py                          # default: all 872 val examples
  python prepare_sst2.py --num-examples 100       # first 100 examples only
  python prepare_sst2.py --output-dir /path/to/   # custom output directory
  python prepare_sst2.py --verify                 # cross-check vs PyTorch model
  python prepare_sst2.py --split validation       # default split

Dependencies:
  pip install transformers datasets torch numpy
  (or: conda run -n ImageNet python prepare_sst2.py)
"""

import argparse
import os
import sys

import numpy as np

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR  = os.path.join(SCRIPT_DIR, "weights")

MODEL_NAME   = "M-FAC/bert-tiny-finetuned-sst2"
SEQ_LEN      = 128
HIDDEN_DIM   = 128


def load_npy(name):
    """Load a .npy weight file from the weights/ directory."""
    path = os.path.join(WEIGHTS_DIR, name + ".npy")
    if not os.path.exists(path):
        sys.exit(f"[ERROR] Weight file not found: {path}\n"
                 "        Run export_weights.py first.")
    return np.load(path).astype(np.float32)


def compute_embeddings_numpy(input_ids, token_type_ids, attention_mask):
    """Compute BERT embeddings from .npy weights (pure numpy, no PyTorch).

    emb = word_emb[input_ids] + pos_emb[0..seq_len-1] + type_emb[token_type_ids]
    emb = LayerNorm(emb, gamma, beta, eps=1e-12)

    Args:
        input_ids:      np.ndarray [batch, seq_len] int64
        token_type_ids: np.ndarray [batch, seq_len] int64
        attention_mask: np.ndarray [batch, seq_len] int64

    Returns:
        embeddings: np.ndarray [batch, seq_len, hidden_dim] float32
    """
    word_emb  = load_npy("bert_embeddings_word_embeddings_weight")       # [30522, 128]
    pos_emb   = load_npy("bert_embeddings_position_embeddings_weight")   # [512, 128]
    type_emb  = load_npy("bert_embeddings_token_type_embeddings_weight") # [2, 128]
    ln_gamma  = load_npy("bert_embeddings_LayerNorm_weight")             # [128]
    ln_beta   = load_npy("bert_embeddings_LayerNorm_bias")               # [128]

    batch_size, seq_len = input_ids.shape

    # Lookup embeddings
    emb = word_emb[input_ids]  # [batch, seq, 128]
    positions = np.arange(seq_len, dtype=np.int64)[np.newaxis, :]  # [1, seq]
    emb += pos_emb[positions]          # broadcast [1, seq, 128] → [batch, seq, 128]
    emb += type_emb[token_type_ids]    # [batch, seq, 128]

    # LayerNorm: (emb - mean) / sqrt(var + eps) * gamma + beta
    mean = emb.mean(axis=-1, keepdims=True)       # [batch, seq, 1]
    var  = emb.var(axis=-1, keepdims=True)         # [batch, seq, 1]
    emb_normed = (emb - mean) / np.sqrt(var + 1e-12)
    emb_out = emb_normed * ln_gamma + ln_beta      # [batch, seq, 128]

    return emb_out


def quantize_embeddings(emb_float):
    """Symmetric per-tensor int8 quantization of embedding outputs.

    Same scheme as generate_bert_params.py:
      scale = max(|emb|) / 127
      emb_q = round(emb / scale).clip(-128, 127)

    Returns:
        emb_q: np.ndarray int8, same shape as emb_float
        scale: float (quantization scale)
    """
    max_abs = max(float(np.max(np.abs(emb_float))), 1e-6)
    scale   = max_abs / 127.0
    emb_q   = np.round(emb_float / scale).clip(-128, 127).astype(np.int8)
    return emb_q, scale


def verify_vs_pytorch(input_ids_np, token_type_ids_np, attention_mask_np, emb_numpy):
    """Cross-check numpy embeddings against PyTorch model output."""
    try:
        import torch
        from transformers import AutoModel
    except ImportError:
        print("WARNING: torch/transformers not available — skipping verification.")
        return

    print("\n--- Verification: comparing numpy embeddings to PyTorch ---")
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()

    # Run just the embeddings layer in PyTorch
    with torch.no_grad():
        input_ids_t      = torch.from_numpy(input_ids_np).long()
        token_type_ids_t = torch.from_numpy(token_type_ids_np).long()
        # PyTorch embedding layer output
        pt_emb = model.embeddings(
            input_ids=input_ids_t,
            token_type_ids=token_type_ids_t,
        ).numpy()

    # Compare
    abs_diff = np.abs(emb_numpy - pt_emb)
    max_diff = float(np.max(abs_diff))
    mean_diff = float(np.mean(abs_diff))
    print(f"  Max  absolute diff: {max_diff:.6f}")
    print(f"  Mean absolute diff: {mean_diff:.6f}")

    if max_diff < 1e-4:
        print("  OK: numpy embeddings match PyTorch (< 1e-4).")
    elif max_diff < 1e-2:
        print("  WARN: small differences detected (< 1e-2), likely float rounding.")
    else:
        print("  ERROR: large difference detected! Check weight loading.")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare SST-2 data for BERT-Tiny Gemmini inference.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--split", default="validation", choices=["validation", "train"],
                        help="SST-2 split to use (default: validation)")
    parser.add_argument("--num-examples", type=int, default=None,
                        help="Number of examples to process (default: all)")
    parser.add_argument("--output-dir", default=os.path.join(SCRIPT_DIR, ".."),
                        help="Output directory for .bin and _labels.txt (default: transformers/)")
    parser.add_argument("--output-prefix", default=None,
                        help="Prefix for output filenames (default: sst2_<split>)")
    parser.add_argument("--verify", action="store_true",
                        help="Cross-check embeddings against PyTorch model")
    args = parser.parse_args()

    # ── Load dependencies ──────────────────────────────────────────────
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("ERROR: transformers is not installed.")
        print("  Install with:  pip install transformers")
        sys.exit(1)

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: datasets is not installed.")
        print("  Install with:  pip install datasets")
        sys.exit(1)

    # ── Verify weight files exist ──────────────────────────────────────
    required_weights = [
        "bert_embeddings_word_embeddings_weight",
        "bert_embeddings_position_embeddings_weight",
        "bert_embeddings_token_type_embeddings_weight",
        "bert_embeddings_LayerNorm_weight",
        "bert_embeddings_LayerNorm_bias",
    ]
    for w in required_weights:
        path = os.path.join(WEIGHTS_DIR, w + ".npy")
        if not os.path.exists(path):
            sys.exit(f"[ERROR] Missing weight file: {path}\n"
                     "        Run export_weights.py first:\n"
                     "          python setup_BERT/export_weights.py")

    # ── Load dataset ───────────────────────────────────────────────────
    print(f"Loading SST-2 {args.split} split...")
    dataset = load_dataset("glue", "sst2", split=args.split)
    total_available = len(dataset)
    print(f"  {total_available} examples available.")

    num_examples = args.num_examples if args.num_examples is not None else total_available
    num_examples = min(num_examples, total_available)

    # ── Tokenize ───────────────────────────────────────────────────────
    print(f"\nTokenizing {num_examples} examples (max_length={SEQ_LEN})...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    all_input_ids      = np.zeros((num_examples, SEQ_LEN), dtype=np.int64)
    all_token_type_ids = np.zeros((num_examples, SEQ_LEN), dtype=np.int64)
    all_attention_mask = np.zeros((num_examples, SEQ_LEN), dtype=np.int64)
    all_labels         = np.zeros(num_examples, dtype=np.int32)

    for i in range(num_examples):
        item    = dataset[i]
        encoded = tokenizer(
            item["sentence"],
            truncation=True,
            max_length=SEQ_LEN,
            padding="max_length",
            return_tensors="np",
        )
        all_input_ids[i]      = encoded["input_ids"][0]
        all_token_type_ids[i] = encoded["token_type_ids"][0]
        all_attention_mask[i] = encoded["attention_mask"][0]
        all_labels[i]         = item["label"]

        if (i + 1) % 200 == 0 or i == 0:
            print(f"  [{i+1}/{num_examples}] tokenized")

    # Distribution
    n_neg = int(np.sum(all_labels == 0))
    n_pos = int(np.sum(all_labels == 1))
    print(f"  Label distribution: NEGATIVE={n_neg}, POSITIVE={n_pos}")

    # ── Compute embeddings ─────────────────────────────────────────────
    print(f"\nComputing embeddings from .npy weights...")
    emb_float = compute_embeddings_numpy(all_input_ids, all_token_type_ids, all_attention_mask)
    print(f"  Embedding shape: {emb_float.shape}")
    print(f"  Embedding range: [{emb_float.min():.4f}, {emb_float.max():.4f}]")

    # ── Optional verification ──────────────────────────────────────────
    if args.verify:
        # Verify on first 10 examples to save time
        n_verify = min(10, num_examples)
        verify_vs_pytorch(
            all_input_ids[:n_verify],
            all_token_type_ids[:n_verify],
            all_attention_mask[:n_verify],
            emb_float[:n_verify],
        )

    # ── Quantize to int8 ──────────────────────────────────────────────
    print(f"\nQuantizing embeddings to int8...")
    emb_q, emb_scale = quantize_embeddings(emb_float)
    print(f"  Quantization scale: {emb_scale:.8f}")
    print(f"  Quantized range: [{emb_q.min()}, {emb_q.max()}]")

    # ── Write output files ─────────────────────────────────────────────
    if args.output_prefix is None:
        prefix = f"sst2_{args.split}"
    else:
        prefix = args.output_prefix

    os.makedirs(args.output_dir, exist_ok=True)

    bin_path  = os.path.join(args.output_dir, f"{prefix}_{num_examples}.bin")
    lbl_path  = os.path.join(args.output_dir, f"{prefix}_{num_examples}_labels.txt")
    mask_path = os.path.join(args.output_dir, f"{prefix}_{num_examples}_attn_masks.bin")

    # Embeddings: [N, SEQ_LEN, HIDDEN_DIM] int8, contiguous
    with open(bin_path, "wb") as f:
        f.write(emb_q.tobytes())

    # Labels: one integer per line
    with open(lbl_path, "w") as f:
        for label in all_labels:
            f.write(f"{label}\n")

    # Attention masks: [N, SEQ_LEN] uint8, contiguous
    attn_masks_u8 = all_attention_mask.astype(np.uint8)
    with open(mask_path, "wb") as f:
        f.write(attn_masks_u8.tobytes())

    # ── Summary ────────────────────────────────────────────────────────
    emb_size  = os.path.getsize(bin_path)
    mask_size = os.path.getsize(mask_path)

    print(f"\nDone! Wrote {num_examples} examples.")
    print(f"  {bin_path}  ({emb_size / 1024:.1f} KB)")
    print(f"  {lbl_path}")
    print(f"  {mask_path}  ({mask_size / 1024:.1f} KB)")
    print(f"\nEach example: {SEQ_LEN} x {HIDDEN_DIM} = {SEQ_LEN * HIDDEN_DIM} bytes (int8)")
    print(f"Total embedding binary: {num_examples} x {SEQ_LEN * HIDDEN_DIM} = {emb_size} bytes")
    print(f"Quantization scale: {emb_scale:.8f}")
    print(f"\nTo use with bert-tiny-sst2-stream.c, set:")
    print(f"  #define NUM_EXAMPLES    {num_examples}")
    print(f"  #define EXAMPLE_SIZE    ({SEQ_LEN} * {HIDDEN_DIM})")
    print(f'  #define EMBEDDINGS_BIN  "{os.path.basename(bin_path)}"')
    print(f'  #define LABELS_TXT      "{os.path.basename(lbl_path)}"')
    print(f'  #define ATTN_MASKS_BIN  "{os.path.basename(mask_path)}"')


if __name__ == "__main__":
    main()
