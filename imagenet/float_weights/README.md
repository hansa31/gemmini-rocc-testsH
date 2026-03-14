# MobileNetV2 Weight Extraction for Gemmini

## Quick Start

```bash
# 1. Install dependencies
pip install torch transformers torchvision numpy

# 2. Generate ImageNet weights (FP32, 1000 classes, 224x224)
python extract_mobilenet_weights.py

# 3. Generate CIFAR-10 weights (FP32, 10 classes, 32x32)
python extract_mobilenet_cifar10_float.py

# 4. Generate CIFAR-10 weights (INT8 quantized, 10 classes, 32x32)
python extract_mobilenet_cifar10_int.py
```

Each script downloads its model from HuggingFace automatically on first run and writes C header files into the parent `imagenet/` directory.

### Generated files

| Script | Output headers | C inference file |
|--------|---------------|-----------------|
| `extract_mobilenet_weights.py` | `mobilenet_params_float.h` | `mobilenet_float.c` |
| `extract_mobilenet_cifar10_float.py` | `mobilenet_cifar10_params_float.h`, `cifar10_images.h` | `mobilenet_cifar10_float.c` |
| `extract_mobilenet_cifar10_int.py` | `mobilenet_cifar10_params.h`, `cifar10_images.h` | `mobilenet_cifar10.c` |

---

## Scripts

### `extract_mobilenet_weights.py` — ImageNet FP32

Extracts FP32 weights from `google/mobilenet_v2_1.0_224` (ImageNet, 1000 classes, 224×224 input). Folds BatchNorm into convolution weights and biases. Outputs `mobilenet_params_float.h` with all 53 layers (52 conv + 1 FC) formatted for Gemmini's `ConvParams` and `FcParams` structs.

### `extract_mobilenet_cifar10_float.py` — CIFAR-10 FP32

Extracts FP32 weights from `jialicheng/cifar10_mobilenet-v2` (CIFAR-10, 10 classes, 32×32 input). This model was fine-tuned from the same `google/mobilenet_v2_1.0_224` base, so the architecture and state dict structure are identical — only the final classifier head has 10 outputs instead of 1000.

Key differences from the ImageNet script:
- **Spatial dimensions** recomputed for 32×32 input (see table below)
- **FC layer**: 10 output classes (`fc_53_w[10][1280]`, `fc_53_b[10][4]`)
- **`output_scale=1.0f`** for all layers (float mode, no quantization scaling needed)
- **Also generates `cifar10_images.h`** with 4 CIFAR-10 test images

### `extract_mobilenet_cifar10_int.py` — CIFAR-10 INT8

Same model as the float version, but applies post-training INT8 quantization:

- **Per-layer symmetric weight quantization**: `scale = max(|w|) / 127`, `w_int8 = clip(round(w / scale), -128, 127)`
- **INT32 biases**: `b_int32 = round(b_float / (w_scale × x_scale))`
- **Power-of-2 output_scale**: expressed as `(1.0 / (1 << N))` for hardware-friendly shifting
- **Scale propagation**: tracks activation scale layer-by-layer using BN running statistics (gamma, beta) to estimate activation ranges via the 3-sigma rule
- **Also generates `cifar10_images.h`** (integer pixel values, compatible with both int and float C files)

No pre-quantized INT8 CIFAR-10 MobileNetV2 model exists on HuggingFace — this script performs the quantization from the FP32 checkpoint.

---

## Architecture

All three scripts extract the same MobileNetV2 architecture (53 layers):

| Layer | Type | Kernel | In Ch | Out Ch | Stride |
|-------|------|--------|-------|--------|--------|
| conv_1 | Conv | 3×3 | 3 | 32 | 2 |
| conv_dw_2 | DW | 3×3 | 32 | 32 | 1 |
| conv_3 | Conv | 1×1 | 32 | 16 | 1 |
| conv_4..conv_51 | Inverted residual blocks ×16 | — | — | — | — |
| conv_52 | Conv | 1×1 | 320 | 1280 | 1 |
| fc_53 | FC | — | 1280 | 10 or 1000 | — |

Each inverted residual block has 3 layers: expand (1×1, ReLU) → depthwise (3×3, ReLU) → project (1×1, no activation). Block 0 has no expand layer (just DW → project).

### Spatial dimensions: 32×32 vs 224×224

Weight shapes are **identical** regardless of input size — only buffer sizes and ConvParams spatial fields change.

| Stage | 224×224 | 32×32 | n_patches (224 → 32) |
|-------|---------|-------|---------------------|
| conv_1 output | 112×112 | 16×16 | 50176 → 1024 |
| After stride-2 DW | 56×56 | 8×8 | 12544 → 256 |
| After stride-2 DW | 28×28 | 4×4 | 3136 → 64 |
| After stride-2 DW | 14×14 | 2×2 | 784 → 16 |
| After stride-2 DW | 7×7 | 1×1 | 196 → 4 |
| Global avg pool | 1×1 | 1×1 | — |

---

## C Inference Files

Located in the parent `imagenet/` directory:

| File | Params header | Images header | Labels |
|------|--------------|---------------|--------|
| `mobilenet.c` | `mobilenet_params.h` | `images.h` | `{75, 900, 125, 897}` (ImageNet) |
| `mobilenet_float.c` | `mobilenet_params_float.h` | `images.h` | `{75, 900, 125, 897}` (ImageNet) |
| `mobilenet_cifar10.c` | `mobilenet_cifar10_params.h` | `cifar10_images.h` | `{3, 8, 8, 0}` (CIFAR-10) |
| `mobilenet_cifar10_float.c` | `mobilenet_cifar10_params_float.h` | `cifar10_images.h` | `{3, 8, 8, 0}` (CIFAR-10) |

The CIFAR-10 C files are copies of `mobilenet.c` with only 3 lines changed (the two `#include` lines and the `correct[]` array). All layer calls read dimensions from params structs, so the same inference code works for both input sizes.

### CIFAR-10 test images

`cifar10_images.h` contains 4 images from the CIFAR-10 test set (indices 0–3):
- Image 0: **cat** (label 3)
- Image 1: **ship** (label 8)
- Image 2: **ship** (label 8)
- Image 3: **airplane** (label 0)

Images are stored as `static const elem_t images[4][32][32][3]` with pixel values centered around zero (`pixel - 128`, range [-128, 127]). This integer representation works for both INT8 and FP32 inference (integer values implicitly cast to float when `elem_t` is float).

---

## BatchNorm Folding

All scripts fold BatchNorm parameters into convolution weights before export. Given conv weight $W$, BN parameters $\gamma, \beta, \mu, \sigma^2$:

$$W_{\text{folded}} = W \cdot \frac{\gamma}{\sqrt{\sigma^2 + \epsilon}}$$

$$b_{\text{folded}} = \beta - \gamma \cdot \frac{\mu}{\sqrt{\sigma^2 + \epsilon}}$$

This eliminates BN as a separate operation — the folded weights produce identical output in a single matmul + bias.

## INT8 Quantization Details

The INT8 script (`extract_mobilenet_cifar10_int.py`) uses per-layer symmetric quantization with scale propagation:

1. **Weight quantization**: $w_{\text{scale}} = \frac{\max(|W|)}{127}$, $W_{\text{int8}} = \text{clip}(\text{round}(W / w_{\text{scale}}), -128, 127)$
2. **Bias quantization**: $b_{\text{int32}} = \text{round}(b / (w_{\text{scale}} \times x_{\text{scale}}))$
3. **Activation range estimation**: $y_{\text{range}} \approx |\beta| + 3|\gamma|$ (from BN running statistics)
4. **Output scale**: $\text{output\_scale} = \frac{w_{\text{scale}} \times x_{\text{scale}}}{y_{\text{scale}}}$, rounded to the nearest power of 2 and expressed as `(1.0 / (1 << N))`
5. **Scale propagation**: The output scale of each layer determines the input scale of the next layer ($x_{\text{scale}}^{(l+1)} = y_{\text{scale}}^{(l)}$)

---

## HuggingFace Models

| Model ID | Dataset | Classes | Input | Accuracy |
|----------|---------|---------|-------|----------|
| `google/mobilenet_v2_1.0_224` | ImageNet | 1000 | 224×224 | ~71.8% top-1 |
| `jialicheng/cifar10_mobilenet-v2` | CIFAR-10 | 10 | 224×224* | 84.46% |

*The CIFAR-10 model was trained at 224×224. Running at native 32×32 will yield lower accuracy due to the aggressive spatial reduction (5 stride-2 layers reduce 32×32 to 1×1). For best accuracy at 32×32, consider changing conv_1 stride from 2 to 1 and re-fine-tuning.

## Notes

- The existing ImageNet files (`mobilenet.c`, `mobilenet_params.h`, `images.h`, `mobilenet_float.c`) are **not modified**.
- All scripts use the HuggingFace Transformers `MobileNetV2ForImageClassification` class and expect its state dict naming convention.
- The `LAYER_ARCH` list in the CIFAR-10 scripts defines the architecture programmatically, making spatial dimension computation automatic for any input size.
