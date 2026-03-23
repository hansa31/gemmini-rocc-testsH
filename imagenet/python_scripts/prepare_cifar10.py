#!/usr/bin/env python3
"""
Prepare CIFAR-10 images for Gemmini inference.

Preprocessing pipeline:
  1. Load image from torchvision CIFAR10 dataset (PIL, RGB, 32x32)
  2. Optionally resize to target resolution (e.g. 224x224 for MobileNetV2)
  3. Convert to numpy uint8 array [H, W, 3]
  4. np.clip((img.astype(np.int32) - 128), -128, 127).astype(np.int8)

Output format:
  <prefix>_<N>.bin        — N images, each H*W*3 int8 values (HWC, RGB), contiguous
  <prefix>_<N>_labels.txt — N lines, one integer label per line (0-9)

CIFAR-10 classes:
  0=airplane, 1=automobile, 2=bird, 3=cat, 4=deer,
  5=dog, 6=frog, 7=horse, 8=ship, 9=truck

Usage:
  python prepare_cifar10.py                                                   # default: 224x224, pixelminus128 -> cifar10_test_10000_224x224.bin
  python prepare_cifar10.py --image-size 224                                  # 224x224 (for MobileNetV2-CIFAR10) -> cifar10_test_10000_224x224.bin
  python prepare_cifar10.py --image-size 32 --normalization cifar10standard   # 32x32 (for ResNet50-CIFAR10)     -> cifar10_test_10000_resnet50.bin
  python prepare_cifar10.py --split train                                     # 50000 training images
  python prepare_cifar10.py --num-images 1000                                 # first 1000 test images
  python prepare_cifar10.py --output-dir /path/to/                            # custom output directory
  python prepare_cifar10.py --visualize 5                                     # show first 5 images

The output binary is compatible with mobilenet_cifar10_stream.c,
mobilenet_cifar10_float_stream.c, resnet50_cifar10_stream.c, and
resnet50_cifar10_float_stream.c which stream BATCH_SIZE=4 images at a time.
"""

import argparse
import os
import sys

import numpy as np
from PIL import Image

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

DEFAULT_IMAGE_SIZE = 224  # MobileNetV2-CIFAR10 was trained at 224x224

# Per-channel CIFAR-10 dataset statistics (used for ResNet50-CIFAR10 training)
_CIFAR10_MEAN  = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
_CIFAR10_STD   = np.array([0.2023, 0.1994, 0.2010], dtype=np.float32)
_CIFAR10_FMAX  = 2.75   # max|(1.0 - mean_ch)/std_ch| across all channels → used to scale to int8


def preprocess_image(img_pil, image_size=224, normalization="pixelminus128"):
    """Resize (if needed), normalize, and return int8 image.

    normalization options:
      'pixelminus128'   — pixel - 128, range [-128, 127].  For MobileNetV2-CIFAR10
                          (trained with mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]).
      'cifar10standard' — per-channel CIFAR-10 statistics, scaled to int8 via
                          int8 = round((pixel/255 - mean) / std / 2.75 * 127).
                          For ResNet50-CIFAR10 (edadaltocg/resnet50_cifar10).
    """
    if image_size != 32:
        img_pil = img_pil.resize((image_size, image_size), Image.BILINEAR)
    img_np = np.array(img_pil, dtype=np.uint8)
    if normalization == "cifar10standard":
        # Per-channel normalization matching ResNet50-CIFAR10 training stats.
        # int8 = round((pixel/255 - mean_ch) / std_ch / FMAX * 127)
        float_img = img_np.astype(np.float32) / 255.0
        normalized = (float_img - _CIFAR10_MEAN) / _CIFAR10_STD  # approx [-2.43, 2.75]
        return np.clip(np.round(normalized / _CIFAR10_FMAX * 127.0), -128, 127).astype(np.int8)
    else:
        # Default: pixel - 128  (for MobileNetV2; mean=0.5, std=0.5 training)
        return np.clip(img_np.astype(np.int32) - 128, -128, 127).astype(np.int8)


def load_cifar10(split="test", data_root="/tmp/cifar10_data"):
    """Download and load CIFAR-10 via torchvision. Returns (images_np, labels)."""
    try:
        from torchvision.datasets import CIFAR10
    except ImportError:
        print("ERROR: torchvision is not installed.")
        print("  Install with:  pip install torchvision")
        sys.exit(1)

    train = (split == "train")
    dataset = CIFAR10(root=data_root, train=train, download=True)
    print(f"Loaded CIFAR-10 {split} set: {len(dataset)} images")
    return dataset


def visualize_preprocessed(dataset, num_vis=5):
    """Show first N preprocessed images (dequantized back to uint8 for display)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: matplotlib not installed — skipping visualization.")
        return

    n = min(num_vis, len(dataset))
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for i in range(n):
        img_pil, label = dataset[i]
        quantized = preprocess_image(img_pil, image_size=32)  # visualize at native res
        # Dequantize: int8 + 128 -> uint8 for display
        display = np.clip(quantized.astype(np.int32) + 128, 0, 255).astype(np.uint8)
        axes[i].imshow(display)
        axes[i].set_title(f"{CIFAR10_CLASSES[label]} ({label})", fontsize=9)
        axes[i].axis("off")

    plt.suptitle("Preprocessed CIFAR-10 images (dequantized for display)", fontsize=12)
    plt.tight_layout()
    preview_path = "cifar10_preprocessed_preview.png"
    plt.savefig(preview_path, dpi=150)
    print(f"Saved preview to {preview_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Prepare CIFAR-10 images for Gemmini MobileNetV2-CIFAR10 inference.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--split", default="test", choices=["test", "train"],
                        help="Which CIFAR-10 split to use (default: test)")
    parser.add_argument("--num-images", type=int, default=None,
                        help="Number of images to process (default: all in split)")
    parser.add_argument("--output-dir", default="../",
                        help="Output directory for .bin and _labels.txt files (default: ../)")
    parser.add_argument("--output-prefix", default=None,
                        help="Prefix for output filenames (default: cifar10_<split>)")
    parser.add_argument("--data-root", default="/tmp/cifar10_data",
                        help="Directory for torchvision CIFAR-10 download cache (default: /tmp/cifar10_data)")
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE,
                        help=f"Target image resolution (default: {DEFAULT_IMAGE_SIZE}). "
                             "Use 224 for MobileNetV2-CIFAR10, 32 for ResNet50-CIFAR10.")
    parser.add_argument("--normalization", default="pixelminus128",
                        choices=["pixelminus128", "cifar10standard"],
                        help="Normalization mode: 'pixelminus128' = pixel-128 (default, for "
                             "MobileNetV2-CIFAR10 at 224x224); 'cifar10standard' = per-channel "
                             "CIFAR-10 mean/std (for ResNet50-CIFAR10 at 32x32).")
    parser.add_argument("--visualize", type=int, default=0, metavar="N",
                        help="Show first N preprocessed images before writing (e.g. --visualize 5)")
    args = parser.parse_args()

    image_size = args.image_size
    dataset = load_cifar10(split=args.split, data_root=args.data_root)

    num_images = args.num_images if args.num_images is not None else len(dataset)
    num_images = min(num_images, len(dataset))

    if args.output_prefix is None:
        prefix = f"cifar10_{args.split}"
    else:
        prefix = args.output_prefix

    normalization = args.normalization
    print(f"Normalization: {normalization}")

    if args.visualize > 0:
        visualize_preprocessed(dataset, args.visualize)

    os.makedirs(args.output_dir, exist_ok=True)
    if image_size != 32:
        # Non-native resolution: encode size in filename (e.g. 224x224 for MobileNetV2-CIFAR10)
        bin_path = os.path.join(args.output_dir, f"{prefix}_{num_images}_{image_size}x{image_size}.bin")
    elif normalization == "cifar10standard":
        # ResNet50-CIFAR10: CIFAR-10 standard per-channel normalization
        bin_path = os.path.join(args.output_dir, f"{prefix}_{num_images}_resnet50.bin")
    else:
        bin_path = os.path.join(args.output_dir, f"{prefix}_{num_images}.bin")
    lbl_path = os.path.join(args.output_dir, f"{prefix}_{num_images}_labels.txt")

    print(f"\nProcessing {num_images} {args.split} images (resizing to {image_size}x{image_size})...")

    num_written = 0
    with open(bin_path, "wb") as bin_f, open(lbl_path, "w") as lbl_f:
        for i in range(num_images):
            img_pil, label = dataset[i]

            quantized = preprocess_image(img_pil, image_size, normalization)
            assert quantized.shape == (image_size, image_size, 3), \
                f"Unexpected image shape {quantized.shape} at index {i}"
            assert quantized.dtype == np.int8

            bin_f.write(quantized.tobytes())
            lbl_f.write(f"{label}\n")
            num_written += 1

            if (i + 1) % 1000 == 0 or i == 0:
                print(f"  [{i+1}/{num_images}] processed  (last: {CIFAR10_CLASSES[label]})")

    file_size_kb = os.path.getsize(bin_path) / 1024
    print(f"\nDone! Wrote {num_written} images.")
    print(f"  {bin_path}  ({file_size_kb:.1f} KB)")
    print(f"  {lbl_path}")
    print(f"\nEach image: {image_size}x{image_size}x3 = {image_size*image_size*3} bytes (int8, HWC, RGB)")
    print(f"Total binary size: {num_written} x {image_size*image_size*3} = {num_written*image_size*image_size*3} bytes")
    print(f"\nTo use with streaming C inference files, set:")
    print(f"  #define NUM_IMAGES   {num_written}")
    print(f"  #define BATCH_SIZE   4")
    print(f"  #define IMAGE_SIZE   ({image_size} * {image_size} * 3)")
    print(f'  #define IMAGES_BIN_FILE  "{os.path.basename(bin_path)}"')
    print(f'  #define LABELS_TXT_FILE  "{os.path.basename(lbl_path)}"')


if __name__ == "__main__":
    main()
