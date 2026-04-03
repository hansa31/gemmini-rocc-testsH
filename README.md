# Gemmini Benchmark Suite

An open-source benchmark suite for the [Gemmini](https://github.com/ucb-bar/gemmini) systolic array accelerator. This suite extends the original Gemmini test infrastructure with full-dataset, end-to-end inference benchmarks for convolutional and transformer neural networks, including complete weight extraction pipelines from publicly available pretrained models.

---

## What Is Included

### Benchmark Tiers

**1. GEMM Micro-Benchmarks** (`bareMetalC/`)
Raw matrix-multiply performance at dimensions representative of real workloads:
- `bench_gemm_resnet.c` — ResNet-50 FC layer (M=256, N=1000, K=2048)
- `bench_gemm_bert.c` — BERT-base hidden dimension (M=768, N=768, K=768)
- Small/medium/large synthetic GEMM benchmarks
- ~60 unit tests covering convolution, depthwise conv, transposed conv, matmul variants (OS/WS), residual add, softmax, layer norm, padding, and more

**2. MLP Benchmarks** (`mlps/`)
Five architectures measured by cycle count (no dataset required):

| Binary | Architecture | Batch |
|--------|-------------|-------|
| `mlp_lenet300` | 784→300→100→10 | 64 |
| `mlp_bert_ffn` | 768→3072→768 | 64 |
| `mlp_gpt2_ffn` | 1024→4096→1024 | 64 |
| `mlp_dlrm_bottom` | 512→256→128→64 | 64 |
| `mlp_dlrm_top` | 1024→1024→512→256→1 | 64 |

**3. CNN Inference** (`imagenet/`)
Full streaming inference from binary dataset files — no static image arrays. Evaluates top-1/top-5 accuracy over 500–10,000 images with batch=4:

| Model | Dataset | Resolution | Precision |
|-------|---------|------------|-----------|
| ResNet-50 | ImageNet | 224×224 | INT8, FP32 |
| ResNet-50 | CIFAR-10 | 32×32 (modified stem) | INT8, FP32 |
| MobileNetV2 | ImageNet | 224×224 | INT8, FP32 |
| MobileNetV2 | CIFAR-10 | 224×224 | INT8, FP32 |

Key benchmark files: `resnet50_v1.c`, `resnet50_cifar10_stream.c`, `mobilenet_v1.c`, `mobilenet_cifar10_stream.c`.

**4. Transformer Inference** (`transformers/`)
- `transformer.c` — BERT-base (H=768, 12-head, seq=128) and transformer-small (H=512, 4-head) cycle benchmarks with synthetic input
- `bert-tiny-sst2-stream.c` — **End-to-end BERT-Tiny SST-2** evaluation on the full 872-example validation set, reporting accuracy, macro-F1, confusion matrix, and cycle count

---

## Weight Extraction from Pretrained Models

All weights are extracted from publicly available HuggingFace models using the scripts in `imagenet/float_weights/` and `transformers/setup_BERT/`.

### ResNet-50

```bash
# ImageNet INT8 weights (from microsoft/resnet-50)
python imagenet/float_weights/extract_resnet50_imagenet_int8.py

# CIFAR-10 INT8 weights (from edadaltocg/resnet50_cifar10)
python imagenet/float_weights/extract_resnet50_cifar10_int.py

# CIFAR-10 FP32 weights
python imagenet/float_weights/extract_resnet50_cifar10_float.py
```

**Quantization method:** Per-layer symmetric INT8 (`scale = max(|w|) / 127`). BatchNorm statistics are folded into convolutional weights. Output scales are rounded to powers of 2 for efficient hardware shifting. Scale is propagated layer-by-layer using BN running mean and variance.

### MobileNetV2

```bash
# ImageNet INT8 weights (from google/mobilenet_v2_1.0_224)
python imagenet/float_weights/extract_mobilenet_imagenet_int8.py

# ImageNet FP32 weights
python imagenet/float_weights/extract_mobilenet_weights.py

# CIFAR-10 INT8 weights (from jialicheng/cifar10_mobilenet-v2)
python imagenet/float_weights/extract_mobilenet_cifar10_int.py

# CIFAR-10 FP32 weights
python imagenet/float_weights/extract_mobilenet_cifar10_float.py
```

MobileNetV2 uses per-channel matmul (`PC_MM`) for the depthwise layers. Architecture: 53 layers (expand → depthwise → project inverted residual blocks) with BatchNorm folded.

### BERT-Tiny (SST-2)

```bash
cd transformers/setup_BERT

# Install dependencies
bash install_deps.sh

# 1. Download weights from M-FAC/bert-tiny-finetuned-sst2 and export as .npy
python export_weights.py

# 2. Quantize to INT8 and generate bert_params.h + bert_input.h
python generate_bert_params.py

# 3. Prepare SST-2 validation set (pre-computes embeddings, writes binary files)
python prepare_sst2.py

# 4. Verify FP32 accuracy (~83%) with PyTorch reference
python reference_inference.py
```

Model: BERT-Tiny (L=2, H=128, 2-head, FFN=512). Weights are transposed to Gemmini layout ([K][N] order). INT8 symmetric quantization is applied to all attention and FFN matrices; biases remain INT32.

---

## Dataset Preparation

### ImageNet (for ResNet-50 and MobileNetV2)

Download the [ILSVRC 2012 validation set](https://image-net.org/) and place images in `imagenet/python_scripts/ILSVRC2012_img_val/`.

```bash
python imagenet/python_scripts/prepare_imagenet.py
```

This resizes each image (shortest edge → 256, center-crop → 224×224), converts to BGR, quantizes pixels as `int8 = clip(pixel - 128, -128, 127)`, and writes a flat binary `imagenet_val_50000.bin` plus a labels file. The binary is streamed 4 images at a time (~600 KB RAM footprint regardless of dataset size).

### CIFAR-10

```bash
# For ResNet-50 (32x32, CIFAR-10 normalization)
python imagenet/python_scripts/prepare_cifar10.py --image-size 32 --normalization cifar10standard

# For MobileNetV2 (upscaled to 224x224)
python imagenet/python_scripts/prepare_cifar10.py --image-size 224
```

CIFAR-10 is downloaded automatically via `torchvision`. Outputs `cifar10_test_10000_resnet50.bin` or `cifar10_test_10000_224x224.bin` with labels.

### SST-2 (for BERT-Tiny)

Prepared automatically by `transformers/setup_BERT/prepare_sst2.py` (see above). Downloads from HuggingFace `datasets`, tokenizes with the BERT-Tiny tokenizer, computes word + position + type embeddings + LayerNorm in PyTorch, and writes pre-embedded INT8 sequences to `sst2_validation_872.bin` and attention masks to `sst2_validation_872_attn_masks.bin`.

---

## Build

```bash
# Build all benchmarks
git submodule update --init --recursive
./build.sh
```

Binaries are installed in `build/`. Each test has three variants: `-baremetal`, `-linux`, `-pk`.

To build a single test:
```bash
./build_single.sh imagenet/resnet50_cifar10_stream.c
```

To build with custom Gemmini parameters:
```bash
./build_with_params.sh <params_file>
```

---

## Running Benchmarks

```bash
# All benchmark suites
./run_all_benchmarks.sh <config_name>

# Individual suites
./run_benchmarks.sh <config_name>           # GEMM micro-benchmarks
./run_mlp_benchmarks.sh <config_name>       # MLP benchmarks
./run_imagenet_benchmarks.sh <config_name>  # CNN inference
./run_transformer_benchmarks.sh <config_name>  # Transformer inference
```

Results are written to timestamped CSV files in `results/`. Each benchmark outputs a `CSV,` sentinel line for automated collection.

To run on the Gemmini ISA simulator (`spike`):
```bash
# Install esp-isa-sim (or use Chipyard's build-toolchains.sh esp-tools)
spike --extension=gemmini build/imagenet/resnet50_cifar10_stream-baremetal
```

---

## Writing Your Own Tests

`bareMetalC/template.c` is a minimal Gemmini test template. Copy it, add your test name to `bareMetalC/Makefile`, and run `./build.sh`.

Key headers:
- `include/gemmini.h` — RoCC instruction definitions and tiled matmul primitives
- `include/gemmini_nn.h` — `ConvParams`, `FcParams`, `tiled_conv_auto`, `tiled_matmul_nn_auto`, `global_average_pool`
- `include/gemmini_params.h` — Hardware configuration (array dimension, data type, SRAM sizes)

---

## Repository Structure

```
gemmini-rocc-tests/
├── bareMetalC/              # Unit tests and GEMM micro-benchmarks
├── imagenet/                # ResNet-50 and MobileNetV2 inference
│   ├── float_weights/       # Weight extraction and quantization scripts
│   ├── python_scripts/      # Dataset preparation scripts
│   ├── resnet50_verify/     # INT8 software simulation for ResNet-50
│   └── mobilenet_verify/    # INT8 software simulation for MobileNetV2
├── transformers/            # BERT-Tiny and transformer cycle benchmarks
│   └── setup_BERT/          # Weight export, quantization, dataset prep
├── mlps/                    # MLP benchmarks
├── include/                 # Shared C headers
│   └── GemminiParams/       # Hardware config variants (FP16, BF16, FP8)
├── run_all_benchmarks.sh    # Master runner
└── build.sh                 # Master build script
```
