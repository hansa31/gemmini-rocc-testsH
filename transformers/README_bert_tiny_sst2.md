# BERT-Tiny SST-2 on Gemmini

End-to-end inference of **BERT-Tiny fine-tuned on SST-2** (binary sentiment
classification) targeted at the Gemmini accelerator.

---

## Hardware-Agnostic Design — `elem_t` / `acc_t` Typedefs

Both files are written entirely in terms of the typedefs defined in
`include/gemmini_params.h`, following the same pattern as `imagenet/mobilenet_v1.c`:

```c
typedef int8_t  elem_t;   // ← change this for a different hardware config
typedef int32_t acc_t;
```

| File | Weight header | `gemmini_params.h` | `elem_t` | `acc_t` |
|------|---------------|--------------------|----------|---------|
| `bert-tiny-sst2.c` | `bert_params.h` | standard (int8 config) | `int8_t` | `int32_t` |
| `bert-tiny-sst2-fp.c` | `bert_params_fp.h` | float config | `float` | `float` |

Neither file contains `float` variable declarations in the encoder — all buffers,
weights, and activations are declared as `elem_t` or `acc_t`.  If the hardware
configuration changes, recompile; no source changes are needed.

The same `tiled_matmul_auto` / `tiled_norm_auto` / `tiled_resadd_auto` calls
appear in both files.  The execution mode is selected at runtime via a
command-line argument, as in `mobilenet_v1.c`:

| Argument | Mode |
|----------|------|
| `ws` (default) | Weight Stationary — Gemmini hardware |
| `os` | Output Stationary — Gemmini hardware |
| `cpu` | Software fallback — RISC-V scalar core |

---

## Model Architecture

```
Input sequence (SEQ_LEN=128 tokens)
        │  elem_t [128][128]
        ▼
 ┌────────────────────────────────────────────────────────────┐
 │  [Embedding — word + position + token_type + LayerNorm]   │
 │  (not implemented here; replaced by dummy bert_input.h)   │
 └──────────────────────────┬─────────────────────────────────┘
                            │  elem_t [SEQ_LEN × HIDDEN_DIM]
          ┌─────────────────▼────────────────────┐
          │         Encoder Layer 0 (of 2)        │
          │  tiled_matmul_auto   ← Q,K,V proj     │
          │  tiled_matmul_auto   ← scores         │
          │  tiled_matmul_auto   ← ctx × V        │
          │  tiled_matmul_auto   ← output proj    │
          │  tiled_norm_auto     ← LayerNorm       │
          │  tiled_resadd_auto   ← residual add    │
          │  tiled_matmul_auto   ← FF1 + IGELU     │
          │  tiled_matmul_auto   ← FF2             │
          │  tiled_norm_auto     ← LayerNorm        │
          │  tiled_resadd_auto   ← residual add     │
          └─────────────────┬────────────────────┘
                            │  (× 2 layers)
          ┌─────────────────▼────────────────────┐
          │         Encoder Layer 1 (of 2)        │
          └─────────────────┬────────────────────┘
                            │  elem_t [SEQ_LEN × HIDDEN_DIM]
                   CLS token = row 0
                            │  elem_t [HIDDEN_DIM]
               ┌────────────▼────────────┐
               │  Pooler (CPU scalar)    │  acc_t dot + (elem_t)tanhf
               └────────────┬────────────┘
               ┌────────────▼────────────┐
               │  Classifier (CPU)       │  acc_t dot
               └────────────┬────────────┘
                    elem_t logits[2]
                            │
                  (float)logits[1] > (float)logits[0]
                            │
                    POSITIVE / NEGATIVE
```

**BERT-Tiny parameters** (`setup_BERT/weights/arch_info.json`):

| Parameter | Value |
|-----------|-------|
| Encoder layers | 2 |
| Hidden dim | 128 |
| Attention heads | 2 × 64 |
| FFN dim | 512 |
| Labels | 2 (NEGATIVE / POSITIVE) |
| Reference accuracy | 83% on SST-2 (float PyTorch) |

---

## File Structure

```
transformers/
├── bert-tiny-sst2.c         ← int8 Gemmini path   (standard gemmini_params.h)
├── bert-tiny-sst2-fp.c      ← float Gemmini path  (float   gemmini_params.h)
├── bert_params.h            ← (generated) int8  elem_t / int32 acc_t weights
├── bert_params_fp.h         ← (generated) float elem_t / float acc_t weights
├── bert_input.h             ← (generated) dummy elem_t input  (like images.h)
├── transformer.c            ← Original Gemmini transformer benchmark
├── README.md                ← transformer.c documentation
├── README_bert_tiny_sst2.md ← this file
└── setup_BERT/
    ├── export_weights.py         ← Step 1: download model → .npy
    ├── generate_bert_params.py   ← Step 2: .npy → C headers
    ├── reference_inference.py    ← PyTorch accuracy check (~83%)
    ├── install_deps.sh
    └── weights/  *.npy  arch_info.json  manifest.txt
```

---

## Step-by-Step Build

### Step 1 — Install Python dependencies (once)

```bash
cd transformers/setup_BERT
bash install_deps.sh
```

### Step 2 — Export model weights

Downloads BERT-Tiny from HuggingFace and saves 41 float32 `.npy` files.

```bash
conda run -n ImageNet python setup_BERT/export_weights.py
```

Optionally verify PyTorch accuracy (~83%):
```bash
conda run -n ImageNet python setup_BERT/reference_inference.py
```

### Step 3 — Generate C headers

```bash
conda run -n ImageNet python setup_BERT/generate_bert_params.py
```

Produces three headers in `transformers/`:

| Header | Contents | Compiles with |
|--------|----------|---------------|
| `bert_params.h` | `elem_t` (int8) weights transposed to gemmini B-matrix layout; `acc_t` (int32) biases; float pooler/clf | standard `gemmini_params.h` |
| `bert_params_fp.h` | `elem_t` (float) original weights, PyTorch layout; `acc_t` (float) biases | float `gemmini_params.h` |
| `bert_input.h` | `elem_t bert_input[128][128]` — values `((t*3+d*7)%9)-4` | both |

> `bert_params_fp.h` intentionally does **not** compile against the standard
> int8 `gemmini_params.h` — float literal array initialisers assigned to
> `int8_t` arrays produce a compiler error, preventing accidental misuse.

### Step 4a — int8 Gemmini path (`bert-tiny-sst2.c`)

```bash
riscv64-unknown-elf-gcc -O2 -DBAREMETAL -Iinclude \
    transformers/bert-tiny-sst2.c -o build/bert-tiny-sst2-baremetal -lm

# Weight-stationary Gemmini (default)
spike --extension=gemmini build/bert-tiny-sst2-baremetal

# Output-stationary Gemmini
spike --extension=gemmini build/bert-tiny-sst2-baremetal os

# Software fallback on RISC-V scalar core (useful for debugging, no accelerator)
spike --extension=gemmini build/bert-tiny-sst2-baremetal cpu
```

### Step 4b — Float Gemmini path (`bert-tiny-sst2-fp.c`)

First compile against a `gemmini_params.h` that defines `typedef float elem_t`:

```bash
riscv64-unknown-elf-gcc -O2 -DBAREMETAL -Iinclude_fp \
    transformers/bert-tiny-sst2-fp.c -o build/bert-tiny-sst2-fp-baremetal -lm

spike --extension=gemmini build/bert-tiny-sst2-fp-baremetal [ws|os|cpu]
```

`include_fp/` should contain a `gemmini_params.h` with:
```c
typedef float   elem_t;
typedef float   acc_t;
```

---

## Expected Output

```
=== BERT-Tiny SST-2 — int8 Gemmini path (mode=ws) ===
hidden=128  ffn=512  heads=2  layers=2  seq_len=128

Encoder layer 0: XXXXXXXX cycles
Encoder layer 1: XXXXXXXX cycles

Logits:     NEGATIVE=X.XXXXXX   POSITIVE=X.XXXXXX
Prediction: NEGATIVE/POSITIVE (label 0/1)
Total cycles: XXXXXXXXX
```

Both files print in this format.  Comparing the logit values between the two
paths shows the numerical effect of int8 quantisation vs. original float weights.

---

## Replacing the Dummy Input with Real Tokens

`bert_input.h` contains a deterministic pattern **(not a valid sentence)**.
For real inference, replace it with:

1. Tokenise the sentence with BERT WordPiece.
2. Look up `word_embedding[token_id] + position_embedding[pos] + token_type_embedding[0]`.
3. Apply `LayerNorm(sum)` — weights in `bert_embeddings_LayerNorm_weight/bias.npy`.
4. For the int8 path: quantise to `elem_t` (int8).
   For the float path: keep as `elem_t` (float).
5. Write the resulting `elem_t [SEQ_LEN][HIDDEN_DIM]` array to a new header,
   include it in place of `bert_input.h`.

---

## Weight Layout Notes

| Header | Encoder weights | Bias | Why |
|--------|----------------|------|-----|
| `bert_params.h` | Transposed from PyTorch `[out,in]` → gemmini `[in,out]` | `acc_t` (int32) | `tiled_matmul_auto` B-matrix is `[K][N]` |
| `bert_params_fp.h` | PyTorch native `[out,in]` — **no transpose** | `acc_t` (float) | `bert-tiny-sst2-fp.c` accesses `W[j][k]` = `W[out_j][in_k]`, computing `out = in @ W.T + b` |

The pooler and classifier are small 2-row / 128-row operations that run on
the CPU scalar core in both files (tanh is not a Gemmini instruction).
They accumulate in `acc_t` and output in `elem_t`, so they adapt
automatically to both hardware configurations.
