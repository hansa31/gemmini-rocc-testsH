# setup_BERT — BERT-Tiny Weight Export for Gemmini C Inference

This directory sets up everything needed to go from a HuggingFace BERT-Tiny model to
`.npy` weight files that can be loaded by a C implementation running on the Gemmini
accelerator.

## Model

**[M-FAC/bert-tiny-finetuned-sst2](https://huggingface.co/M-FAC/bert-tiny-finetuned-sst2)** — BERT-Tiny fine-tuned on the SST-2 (Stanford Sentiment Treebank) binary sentiment classification task.

| Parameter | Value |
|-----------|-------|
| Layers | 2 |
| Hidden dim | 128 |
| Attention heads | 2 |
| Head dim | 64 |
| FFN intermediate size | 512 |
| Vocab size | 30 522 |
| Max sequence length | 512 |
| Task labels | 2 (positive / negative) |

## Directory Structure

```
setup_BERT/
├── install_deps.sh           # Install Python dependencies into ImageNet conda env
├── reference_inference.py    # Run PyTorch inference on SST-2 val set (accuracy check)
├── export_weights.py         # Export all weights as .npy files
├── weights/                  # (generated) .npy files + manifest + arch_info.json
│   ├── arch_info.json
│   ├── manifest.txt
│   ├── bert_embeddings_word_embeddings_weight.npy
│   ├── bert_encoder_layer_0_attention_self_query_weight.npy
│   ├── ...                   # 41 .npy files total
│   └── classifier_bias.npy
└── README.md                 # This file
```

## Prerequisites

- Conda with the **ImageNet** environment already created
- Internet access (to download the model from HuggingFace)

## Step-by-Step Usage

### 1. Install Dependencies

```bash
cd setup_BERT
bash install_deps.sh
```

This installs `torch`, `transformers`, `datasets`, `tokenizers`, and `numpy` into the
`ImageNet` conda environment, then prints version verification.

### 2. Verify PyTorch Inference (Reference Accuracy)

```bash
conda run -n ImageNet python reference_inference.py
```

This script:
1. Loads the model from HuggingFace
2. Runs a **sanity check** on a known positive sentence (asserts the model predicts "positive")
3. Evaluates on the **full SST-2 validation set** (872 examples)
4. Prints running accuracy and a final score
5. **Fails with exit code 1** if accuracy drops below 75% (something is wrong)

**Expected output:** ~83% accuracy.

### 3. Export Weights for C

```bash
conda run -n ImageNet python export_weights.py
```

This script:
1. Loads the model and **verifies the architecture** matches BERT-Tiny (L=2, H=128, A=2, FFN=512)
2. Saves every weight tensor as a `.npy` file in `weights/`
3. Writes `weights/manifest.txt` (file-name → original-name → shape → dtype mapping)
4. Writes `weights/arch_info.json` (architecture constants for the C code)
5. **Spot-checks** reloaded `.npy` files against the original tensors
6. Reports total weight data size (~17 MB)

## Weight File Naming Convention

Each `.npy` file is named after the PyTorch state_dict key with `/` and `.` replaced by `_`. Examples:

| PyTorch key | File name | Shape |
|-------------|-----------|-------|
| `bert.encoder.layer.0.attention.self.query.weight` | `bert_encoder_layer_0_attention_self_query_weight.npy` | [128, 128] |
| `bert.encoder.layer.0.attention.self.query.bias` | `bert_encoder_layer_0_attention_self_query_bias.npy` | [128] |
| `bert.encoder.layer.0.intermediate.dense.weight` | `bert_encoder_layer_0_intermediate_dense_weight.npy` | [512, 128] |
| `classifier.weight` | `classifier_weight.npy` | [2, 128] |

## How These Weights Map to transformer.c

The existing [transformer.c](../transformer.c) implements encoder layers with `attention()` + `ffn()`. Here's the mapping for each BERT encoder layer:

| transformer.c parameter | BERT weight |
|------------------------|-------------|
| `Wq` | `encoder.layer.N.attention.self.query.weight` |
| `Wk` | `encoder.layer.N.attention.self.key.weight` |
| `Wv` | `encoder.layer.N.attention.self.value.weight` |
| `Wo` | `encoder.layer.N.attention.output.dense.weight` |
| `Wq_b`, `Wk_b`, `Wv_b` | Corresponding `.bias` tensors |
| `Wo_b` | `encoder.layer.N.attention.output.dense.bias` |
| `ff1_w` | `encoder.layer.N.intermediate.dense.weight` (512×128) |
| `ff2_w` | `encoder.layer.N.output.dense.weight` (128×512) |
| `ff1_b` | `encoder.layer.N.intermediate.dense.bias` (512) |
| `ff2_b` | `encoder.layer.N.output.dense.bias` (128) |

**Note:** The current `transformer.c` benchmarks BERT-base (H=768) and a small transformer (H=512). To run BERT-Tiny, you would add a call with `hidden_dim=128, expansion_dim=512, num_heads=2, seq_len=128`.

## Verified Results

| Check | Result |
|-------|--------|
| Dependencies install | torch 2.10.0, transformers 5.3.0, datasets 4.7.0 |
| Sanity check (positive sentence) | Passed |
| SST-2 validation accuracy | **83.03%** (724/872) |
| Weight tensors exported | 41 |
| Spot-check reload | All matched |
| Total weight size | ~17 MB |
