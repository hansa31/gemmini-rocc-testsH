# ImageNet Preparation Scripts for Gemmini

Python utilities to prepare ImageNet validation images as input for the
MobileNetV1 inference program (`mobilenet_v1.c`) running on the Gemmini accelerator.

## Requirements

```bash
pip install numpy Pillow
```

## Scripts

### `prepare_imagenet.py`

Reads ImageNet validation JPEGs, preprocesses them (resize to 256, center-crop
to 224×224, quantize to int8), and writes:

- **`imagenet_val_N.bin`** — flat binary file of N images, each 224×224×3 int8
  values in HWC (row-major) layout, contiguous
- **`imagenet_val_N_labels.txt`** — one integer label per line

#### Quick start (full 50K validation set)

```bash
python prepare_imagenet.py \
    --imagenet-dir /path/to/ILSVRC2012_img_val \
    --labels-file /path/to/val.txt \
    --num-images 50000 \
    --output-dir ../ \
    --output-prefix imagenet_val
```

Where `val.txt` has lines like:
```
ILSVRC2012_val_00000001.JPEG 65
ILSVRC2012_val_00000002.JPEG 970
...
```

#### Subset (e.g. 200 images for testing)

```bash
python prepare_imagenet.py \
    --imagenet-dir /path/to/ILSVRC2012_img_val \
    --labels-file /path/to/val.txt \
    --num-images 200 \
    --output-dir ../ \
    --output-prefix imagenet_val
```

#### Custom quantization

If your model uses a different quantization scheme:

```bash
python prepare_imagenet.py \
    --imagenet-dir /path/to/val \
    --labels-file val.txt \
    --scale 0.0078125 \
    --zero-point -128 \
    --output-dir ../
```

### `verify_images_bin.py`

Reads back a binary file and prints statistics + first few pixels to verify
correctness.

```bash
python verify_images_bin.py ../imagenet_val_200.bin ../imagenet_val_200_labels.txt --num-images 200
```

## Binary Format

The `.bin` file is a flat array of `int8` values:

```
[image_0: 224*224*3 bytes] [image_1: 224*224*3 bytes] ... [image_N-1]
```

Each image is stored in **HWC** (Height, Width, Channel) order with **RGB**
channel ordering. This matches the layout in the original `images.h`:

```c
static const elem_t images[4][224][224][3] = ...;
```

The C program (`mobilenet_v1.c`) reads 4 images (one batch) at a time using
sequential `fread`, keeping memory usage constant regardless of dataset size.

## Data sizes

| Images | Binary size |
|--------|-------------|
| 4      | ~588 KB     |
| 200    | ~29 MB      |
| 50,000 | ~7.2 GB     |

The C program only needs ~600 KB of image memory at runtime (one batch of 4).
