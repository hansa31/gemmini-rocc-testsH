# Transformer Benchmark for Gemmini Accelerator

## What Is This?

`transformer.c` is a **benchmark program** that runs transformer model inference on the [Gemmini](https://github.com/ucb-bar/gemmini) hardware accelerator. Gemmini is a matrix-multiply accelerator designed for RISC-V processors (part of the [Chipyard](https://github.com/ucb-bar/chipyard) SoC framework). This code exercises Gemmini's accelerated matrix operations to measure how efficiently it can run transformer workloads like BERT.

**In short:** this is not a training script or a general-purpose transformer library — it is a hardware test that benchmarks transformer inference on a custom accelerator, reporting cycle counts.

---

## Quick Transformer Recap

A transformer layer has two main stages:

1. **Multi-Head Attention** — The model learns *what to pay attention to* in a sequence.
2. **Feed-Forward Network (FFN)** — A two-layer neural network applied to each position independently.

Each stage is followed by **Layer Normalization** and a **Residual Addition** (adding the input back to the output to help training stability).

---

## Code Structure

The file defines three core functions and two helper macros:

### `attention()`
Implements multi-head attention:

| Step | Operation | Description |
|------|-----------|-------------|
| 1 | Q = input × Wq | Compute Query matrix |
| 2 | K = enc_out × Wk | Compute Key matrix |
| 3 | V = enc_out × Wv | Compute Value matrix |
| 4 | attn = softmax(Q × Kᵀ) | Attention scores (per head) |
| 5 | out = attn × V | Weighted values (per head) |
| 6 | out = out × Wo | Project back to hidden dimension |
| 7 | out = LayerNorm(out) | Normalize |
| 8 | out = out + input | Residual connection |

For **self-attention** (encoder), `input` and `enc_out` are the same tensor. For **cross-attention** (decoder), `enc_out` comes from the encoder's output.

### `ffn()`
Implements the feed-forward network:

| Step | Operation | Description |
|------|-----------|-------------|
| 1 | out = GELU(input × FF1_w + FF1_b) | First linear layer with GELU activation |
| 2 | out = input × FF2_w + FF2_b | Second linear layer (project back to hidden_dim) |
| 3 | out = LayerNorm(out) | Normalize |
| 4 | out = out + input | Residual connection |

### `encoder_decoder()`
Combines the above into a full transformer layer:
- **Encoder mode** (`enc_out == NULL`): runs self-attention → FFN.
- **Decoder mode** (`enc_out != NULL`): runs self-attention → cross-attention → FFN.

Returns the total cycle count for the layer.

### `ENCODER_DECODER` macro
Allocates all the weight and buffer arrays as `static` variables and calls `encoder_decoder()`. This is a convenience wrapper so each benchmark call is self-contained.

### `PRINT_ENCODER_DECODER` macro
Allocates input/output arrays, runs a benchmark, and prints the configuration and cycle count.

---

## What `main()` Benchmarks

The program runs two encoder configurations:

| Name | Hidden Dim | FFN Dim | Heads | Seq Len |
|------|-----------|---------|-------|---------|
| **bert-base** | 768 | 3072 | 12 | 128 |
| **transformer-small** | 512 | 1024 | 4 | 128 |

Both are run as **encoder** layers (self-attention only, no cross-attention).

---

## Key Gemmini Functions Used

| Function | Purpose |
|----------|---------|
| `tiled_matmul_auto` | Accelerated matrix multiplication (the workhorse) |
| `tiled_norm_auto` | Accelerated layer normalization |
| `tiled_resadd_auto` | Accelerated element-wise residual addition |
| `gemmini_fence` | Ensures all pending accelerator operations complete before proceeding |
| `gemmini_flush` | Resets/initializes the accelerator |

All heavy computation is offloaded to the Gemmini accelerator rather than running on the RISC-V CPU.

---

## How to Build and Run

```bash
# From the repo root
./build.sh

# Run on the Gemmini ISA simulator (spike)
cd build/transformers
spike --extension=gemmini transformer-baremetal
```

The output will print cycle counts for each transformer configuration, which can be used to evaluate Gemmini's performance on transformer workloads.
