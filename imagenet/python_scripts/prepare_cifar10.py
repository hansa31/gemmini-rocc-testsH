#!/usr/bin/env python3
"""
Prepare CIFAR-10 images for Gemmini MobileNetV2-CIFAR10 inference.

Preprocessing pipeline (matches extract_mobilenet_cifar10_float.py exactly):
  1. Load image from torchvision CIFAR10 dataset (PIL, RGB, 32x32)
  2. Convert to numpy uint8 array [32, 32, 3]
  3. np.clip((img.astype(np.int32) - 128), -128, 127).astype(np.int8)

Output format:
  <prefix>_<N>.bin        — N images, each 32*32*3 int8 values (HWC, RGB), contiguous
  <prefix>_<N>_labels.txt — N lines, one integer label per line (0-9)

CIFAR-10 classes:
  0=airplane, 1=automobile, 2=bird, 3=cat, 4=deer,
  5=dog, 6=frog, 7=horse, 8=ship, 9=truck

Usage:
  python prepare_cifar10.py                          # default: 10000 test images
  python prepare_cifar10.py --split train            # 50000 training images
  python prepare_cifar10.py --num-images 1000        # first 1000 test images
  python prepare_cifar10.py --output-dir /path/to/  # custom output directory
  python prepare_cifar10.py --visualize 5            # show first 5 images

The output binary is compatible with mobilenet_cifar10_stream.c and
mobilenet_cifar10_float_stream.c which stream BATCH_SIZE=4 images at a time.
"""

import argparse
import os
import sys

import numpy as np

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

IMAGE_SIZE = 32  # CIFAR-10 native resolution


def preprocess_image(img_np):
    """Center and clip a uint8 [32, 32, 3] HWC RGB image to int8.

    Pipeline (matches cifar10_images.h generation in extract_mobilenet_cifar10*.py):
      1. img_np is already [32, 32, 3] uint8, RGB order (torchvision default)
      2. np.clip((img.astype(np.int32) - 128), -128, 127).astype(np.int8)
    No resize or crop needed — CIFAR-10 images are natively 32x32.
    """
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
        img_np = np.array(img_pil, dtype=np.uint8)
        quantized = preprocess_image(img_np)
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
    parser.add_argument("--visualize", type=int, default=0, metavar="N",
                        help="Show first N preprocessed images before writing (e.g. --visualize 5)")
    args = parser.parse_args()

    dataset = load_cifar10(split=args.split, data_root=args.data_root)

    num_images = args.num_images if args.num_images is not None else len(dataset)
    num_images = min(num_images, len(dataset))

    if args.output_prefix is None:
        prefix = f"cifar10_{args.split}"
    else:
        prefix = args.output_prefix

    if args.visualize > 0:
        visualize_preprocessed(dataset, args.visualize)

    os.makedirs(args.output_dir, exist_ok=True)
    bin_path = os.path.join(args.output_dir, f"{prefix}_{num_images}.bin")
    lbl_path = os.path.join(args.output_dir, f"{prefix}_{num_images}_labels.txt")

    print(f"\nProcessing {num_images} {args.split} images...")

    num_written = 0
    with open(bin_path, "wb") as bin_f, open(lbl_path, "w") as lbl_f:
        for i in range(num_images):
            img_pil, label = dataset[i]
            img_np = np.array(img_pil, dtype=np.uint8)  # [32, 32, 3] uint8 RGB
            assert img_np.shape == (IMAGE_SIZE, IMAGE_SIZE, 3), \
                f"Unexpected image shape {img_np.shape} at index {i}"

            quantized = preprocess_image(img_np)
            assert quantized.shape == (IMAGE_SIZE, IMAGE_SIZE, 3)
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
    print(f"\nEach image: {IMAGE_SIZE}x{IMAGE_SIZE}x3 = {IMAGE_SIZE*IMAGE_SIZE*3} bytes (int8, HWC, RGB)")
    print(f"Total binary size: {num_written} x {IMAGE_SIZE*IMAGE_SIZE*3} = {num_written*IMAGE_SIZE*IMAGE_SIZE*3} bytes")
    print(f"\nTo use with mobilenet_cifar10_stream.c, set:")
    print(f"  #define NUM_IMAGES   {num_written}")
    print(f"  #define BATCH_SIZE   4")
    print(f'  #define IMAGES_BIN_FILE  "{os.path.basename(bin_path)}"')
    print(f'  #define LABELS_TXT_FILE  "{os.path.basename(lbl_path)}"')


if __name__ == "__main__":
    main()
