# Gemmini MLP Benchmarks

Cycle-accurate MLP (Multi-Layer Perceptron) inference benchmarks for the [Gemmini](https://github.com/ucb-bar/gemmini) systolic array accelerator. Each program runs a forward pass through a fixed-weight (zero-initialised) MLP entirely on Gemmini hardware and prints per-layer and total hardware cycle counts.

---

## Standard MLP Benchmarks

Five benchmarks based on well-known, citable network architectures. All use `SYS_DIM=16` zero-padding and `batch_size=64`.

### 1. LeNet-300-100 (`mlp_lenet300`)

| | |
|---|---|
| **Source** | LeCun et al., "Gradient-Based Learning Applied to Document Recognition", 1998 |
| **Why** | The original fully-connected MNIST classifier. Referenced in virtually every neural network pruning, quantization, and accelerator paper as a baseline. The de facto "hello world" MLP benchmark. |
| **Topology** | `784 → 300 → 100 → 10` (3 layers) |
| **Zero-padded** | `784 → 304 → 112 → 16` |
| **Total weight params** | 784×304 + 304×112 + 112×16 = **274,048** |

### 2. BERT-base FFN (`mlp_bert_ffn`)

| | |
|---|---|
| **Source** | Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers", 2019 |
| **Why** | The feed-forward sub-network inside every transformer encoder layer of BERT-base. BERT-base has 12 of these blocks, so this FFN dominates total compute. Standard for benchmarking transformer inference on accelerators. |
| **Topology** | `768 → 3072 → 768` (2 layers, 4× expansion) |
| **Zero-padded** | `768 → 3072 → 768` (already aligned) |
| **Total weight params** | 768×3072 + 3072×768 = **4,718,592** |

### 3. GPT-2 Small FFN (`mlp_gpt2_ffn`)

| | |
|---|---|
| **Source** | Radford et al., "Language Models are Unsupervised Multitask Learners", 2019 |
| **Why** | The feed-forward block in GPT-2 Small (12 layers). Represents the autoregressive/generative transformer family, complementing BERT's encoder-only architecture. Same 4× expansion ratio but with larger hidden dimension (1024 vs 768), giving ~1.8× more FLOPs per layer. |
| **Topology** | `1024 → 4096 → 1024` (2 layers, 4× expansion) |
| **Zero-padded** | `1024 → 4096 → 1024` (already aligned) |
| **Total weight params** | 1024×4096 + 4096×1024 = **8,388,608** |

### 4. DLRM Bottom MLP (`mlp_dlrm_bottom`)

| | |
|---|---|
| **Source** | Naumov et al., "Deep Learning Recommendation Model for Personalization and Recommendation Systems", 2019. Part of MLPerf benchmark suite. |
| **Why** | The bottom (embedding-processing) MLP stack from Meta's DLRM. Recommendation models are among the largest datacenter workloads. DLRM is the official MLPerf recommendation benchmark. This stack features rapidly shrinking layer sizes, stressing the accelerator's ability to handle small tiles efficiently. |
| **Topology** | `512 → 256 → 128 → 64` (3 layers, halving) |
| **Zero-padded** | `512 → 256 → 128 → 64` (already aligned) |
| **Total weight params** | 512×256 + 256×128 + 128×64 = **172,032** |

### 5. DLRM Top MLP (`mlp_dlrm_top`)

| | |
|---|---|
| **Source** | Same as above (Naumov et al., 2019 / MLPerf) |
| **Why** | The top (prediction) MLP stack from DLRM. Deeper than the bottom stack (4 layers) with a large-to-tiny narrowing pattern (1024 → 1). Tests the accelerator with a mix of large and very small matrix multiplications in a single inference pass. |
| **Topology** | `1024 → 1024 → 512 → 256 → 1` (4 layers) |
| **Zero-padded** | `1024 → 1024 → 512 → 256 → 16` |
| **Total weight params** | 1024×1024 + 1024×512 + 512×256 + 256×16 = **1,704,960** |

### Summary table

| Benchmark | Layers | Raw topology | FLOPs per inference (batch=64) | Characterizes |
|---|---|---|---|---|
| `mlp_lenet300` | 3 | 784→300→100→10 | ~31M | Classic shallow MLP |
| `mlp_bert_ffn` | 2 | 768→3072→768 | ~604M | Transformer encoder FFN |
| `mlp_gpt2_ffn` | 2 | 1024→4096→1024 | ~1074M | Transformer decoder FFN |
| `mlp_dlrm_bottom` | 3 | 512→256→128→64 | ~22M | Rec-sys small/fast stack |
| `mlp_dlrm_top` | 4 | 1024→1024→512→256→1 | ~218M | Rec-sys deep narrowing stack |

FLOPs = `2 × batch × dim_in × dim_out` per layer, summed across all layers.

---

## Legacy Benchmarks (mlp1–mlp4)

The original `mlp1`–`mlp4` (and `mlp1_32`–`mlp4_32`) are custom/ad-hoc topologies used for earlier development testing. They are **not** drawn from published architectures. Kept for backwards compatibility.

| Programs | Layers | Network shape (raw → zero-padded, batch=64) |
|---|---|---|
| `mlp1` / `mlp1_32` | 6 | `784→2500→2000→1500→1000→500→10` → `832→2560→2048→1536→1024→512→64` |
| `mlp2` / `mlp2_32` | 2 | `784→800→10` → `832→832→64` |
| `mlp3` / `mlp3_32` | 2 | `400→500→440` → `448→512→448` |
| `mlp4` / `mlp4_32` | 2 | `3036→4554→3036` → `3072→4608→3072` |

---

## File Overview

```
mlps/
├── Makefile                           # Builds all programs (baremetal / linux / pk)
├── gemmini_matmul_generator.ipynb     # Python generator for parameter .h and .c files
├── README.md
│
│  ── Standard benchmarks ──
├── parameters_lenet300.h              # LeNet-300-100 arrays (784→304→112→16)
├── parameters_bert_ffn.h              # BERT-base FFN arrays (768→3072→768)
├── parameters_gpt2_ffn.h             # GPT-2 Small FFN arrays (1024→4096→1024)
├── parameters_dlrm_bottom.h           # DLRM bottom MLP arrays (512→256→128→64)
├── parameters_dlrm_top.h             # DLRM top MLP arrays (1024→1024→512→256→16)
│
├── mlp_lenet300.c                     # LeNet-300-100 cycle benchmark
├── mlp_bert_ffn.c                     # BERT-base FFN cycle benchmark
├── mlp_gpt2_ffn.c                    # GPT-2 Small FFN cycle benchmark
├── mlp_dlrm_bottom.c                  # DLRM bottom MLP cycle benchmark
├── mlp_dlrm_top.c                    # DLRM top MLP cycle benchmark
│
│  ── Legacy benchmarks ──
├── test.c                             # Small 4-layer dev/sanity-check
├── parameters.h                       # Arrays for test.c
├── parameters[1-4].h                  # Arrays for mlp[1-4]
├── parameters[5-8].h                  # Arrays for mlp[1-4]_32
├── mlp[1-4].c                         # Custom topology benchmarks (default config)
└── mlp[1-4]_32.c                      # Custom topology benchmarks (SYS_DIM=32)
```

---

## Build

```bash
# Standard benchmarks
make mlp_lenet300-baremetal
make mlp_bert_ffn-linux
make mlp_gpt2_ffn-pk
make mlp_dlrm_bottom-baremetal
make mlp_dlrm_top-linux

# Legacy benchmarks
make mlp1-baremetal
make mlp1_32-linux
# etc.
```

Compiler flags: `-O2 -ffast-math -march=rv64gc -mcmodel=medany`.

---

## How It Works

1. Large static weight/activation matrices are declared in the parameter headers (zero-initialised, `int8_t`, DMA-aligned via `row_align`).
2. The C programs call `tiled_matmul_nn_auto` for each layer, which issues Gemmini RoCC instructions to execute the tiled matrix multiplication on the systolic array.
3. `read_cycles()` wraps each layer call to record hardware cycle counts.
4. Results (per-layer cycles + total) are printed at the end.

---

## Code Generator

`gemmini_matmul_generator.ipynb` is the Jupyter notebook that can generate parameter `.h` files and C test files. To create new topologies, set `SYS_DIM`, `batch_size`, and the `layers` list in the notebook and re-run it.
