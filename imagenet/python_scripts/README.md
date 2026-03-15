# ImageNet Preparation Scripts for Gemmini

Prepare ImageNet validation images as input for `mobilenet_v1.c`.

## End-to-End: How to Run MobileNetV1 on the Full ImageNet Validation Set

### 1. Install Python dependencies

```bash
pip install numpy Pillow
```

### 2. Prepare the binary image file

```bash
cd imagenet/python_scripts

python prepare_imagenet.py \
    --imagenet-dir /path/to/ILSVRC2012_img_val \
    --labels-file /path/to/val.txt \
    --num-images 50000 \
    --output-dir ../
```

This produces two files in the `imagenet/` directory:
- `imagenet_val_50000.bin` — 50K images as signed int8, ~7.2 GB
- `imagenet_val_50000_labels.txt` — one label per line

The `val.txt` labels file has lines like: `ILSVRC2012_val_00000001.JPEG 65`

For a smaller test run, use `--num-images 200`.

### 3. (Optional) Verify the prepared data

```bash
python verify_images_bin.py ../imagenet_val_50000.bin ../imagenet_val_50000_labels.txt --num-images 50000 --show 3
```

### 4. Configure `mobilenet_v1.c`

Edit the `#define`s at the top of `mobilenet_v1.c` to match your files:

```c
#define NUM_IMAGES 50000
#define IMAGES_BIN_FILE "imagenet_val_50000.bin"
#define LABELS_TXT_FILE "imagenet_val_50000_labels.txt"
```

For a 200-image test:
```c
#define NUM_IMAGES 200
#define IMAGES_BIN_FILE "imagenet_val_200.bin"
#define LABELS_TXT_FILE "imagenet_val_200_labels.txt"
```

### 5. Build and run

```bash
# Build (from gemmini-rocc-tests root)
make -C imagenet

# Run (place .bin and .txt files in the working directory)
./imagenet/mobilenet_v1-linux ws conv
```

Arguments: `[ws|os|cpu] [conv|matmul] [check]`

## How It Works

- **Data format**: `elem_t = int8_t` (signed, -128 to 127). The Python script
  maps pixel values [0, 255] → [-128, 127] by default (`zero_point=-128`),
  matching the original `images.h` format.
- **Binary layout**: flat contiguous int8 values, images back-to-back, HWC
  (224×224×3 per image). Equivalent to `int8_t images[N][224][224][3]`.
- **Streaming**: `mobilenet_v1.c` reads 4 images (one batch) at a time via
  `fread`, keeping image memory constant at ~600 KB regardless of dataset size.
- **Labels**: loaded fully into memory (50K ints = ~200 KB).

| Images | Binary size | Image RAM in C |
|--------|-------------|----------------|
| 200    | ~29 MB      | ~600 KB        |
| 50,000 | ~7.2 GB     | ~600 KB        |

---

# CIFAR-10 Preparation Scripts for Gemmini

Prepare CIFAR-10 images as input for `mobilenet_cifar10_stream.c` and
`mobilenet_cifar10_float_stream.c`.

## End-to-End: How to Run MobileNetV2-CIFAR10 on CIFAR-10 Images

### 1. Install Python dependencies

```bash
pip install numpy torchvision
```

> `torchvision` handles downloading and loading CIFAR-10 automatically.
> No manual dataset download is required.

### 2. Prepare the binary image file

```bash
cd imagenet/python_scripts

# Full test set (10,000 images) — default
python prepare_cifar10.py

# Training set (50,000 images)
python prepare_cifar10.py --split train

# Subset — first 1,000 test images
python prepare_cifar10.py --num-images 1000

# Custom output directory
python prepare_cifar10.py --output-dir /path/to/output/

# Visualize first 5 preprocessed images before writing
python prepare_cifar10.py --visualize 5
```

CIFAR-10 is downloaded automatically to `/tmp/cifar10_data` on the first run
(override with `--data-root /your/cache`).

This produces two files (default: in `imagenet/`):
- `cifar10_test_10000.bin` — 10K images as signed int8, ~30 MB
- `cifar10_test_10000_labels.txt` — one label per line (0–9)

CIFAR-10 class mapping: `0=airplane 1=automobile 2=bird 3=cat 4=deer 5=dog 6=frog 7=horse 8=ship 9=truck`

### 3. Configure `mobilenet_cifar10_stream.c`

Edit the `#define`s at the top of the C source to match your files:

```c
#define NUM_IMAGES      10000
#define BATCH_SIZE      4
#define IMAGES_BIN_FILE "cifar10_test_10000.bin"
#define LABELS_TXT_FILE "cifar10_test_10000_labels.txt"
```

For a quick 1,000-image test:
```c
#define NUM_IMAGES      1000
#define BATCH_SIZE      4
#define IMAGES_BIN_FILE "cifar10_test_1000.bin"
#define LABELS_TXT_FILE "cifar10_test_1000_labels.txt"
```

### 4. Build and run

```bash
# Build (from gemmini-rocc-tests root)
make -C imagenet

# Run (place .bin and .txt files in the working directory)
./imagenet/mobilenet_cifar10_stream-linux
```

### How It Works

- **Preprocessing**: each uint8 pixel is mapped to int8 via
  `clip(pixel - 128, -128, 127)`, matching the original `cifar10_images.h`
  generation pipeline.
- **Data format**: `int8_t`, HWC layout, RGB order.
- **Binary layout**: flat contiguous int8 values, images back-to-back,
  32×32×3 bytes per image. Equivalent to `int8_t images[N][32][32][3]`.
- **Streaming**: the C program reads `BATCH_SIZE=4` images at a time via
  `fread`, keeping image memory constant (~12 KB) regardless of dataset size.
- **Labels**: loaded fully into memory (10K ints ≈ 40 KB).

| Images | Binary size | Image RAM in C |
|--------|-------------|----------------|
| 1,000  | ~2.9 MB     | ~12 KB         |
| 10,000 | ~29 MB      | ~12 KB         |
| 50,000 | ~147 MB     | ~12 KB         |
