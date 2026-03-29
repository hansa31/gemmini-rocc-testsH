#!/usr/bin/env python3
"""
Prepare ImageNet validation images for Gemmini ResNet-50 inference.
Uses the standard ImageNet normalization pipeline matching the training pipeline:
  1. cv2.resize(img, (224, 224))  — simple stretch, NO center crop
  2. BGR -> RGB via cv2.cvtColor
  3. x_float = (pixel/255 - mean) / std
  4. x_int8 = clip(round(x_float / FMAX * 127), -128, 127)

  ImageNet mean = [0.485, 0.456, 0.406]  (RGB)
  ImageNet std  = [0.229, 0.224, 0.225]  (RGB)
  FMAX = 2.75  (corresponds to x_scale_0 = FMAX/127 in the Gemmini C code)

Output format:
  images.bin  — N images, each 224*224*3 int8 values (HWC, RGB), contiguous
  labels.txt  — N lines, one integer label per line

Usage:
  python prepare_imagenet.py --imagenet-dir /path/to/ILSVRC2012_img_val \
                             --labels-file val.txt \
                             --num-images 50000 \
                             --output-dir ../

  The --labels-file should be a text file with lines like:
      ILSVRC2012_val_00000001.JPEG 65
  (filename followed by integer class label, space-separated)

  If you already have a sorted label list (one int per line), use:
      --labels-file val_labels.txt --labels-format plain

  To visualize first N preprocessed images:
      --visualize 5
"""

import argparse
import os
import sys

import numpy as np

# Standard ImageNet normalization constants
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float64)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float64)
FMAX = 2.75  # INT8 quantization range: x_scale_0 = FMAX/127


# ---------- Preprocessing: standard ImageNet normalization + INT8 quantization ----------

def preprocess_image(img_path, input_size=224):
    """Load an image, stretch-resize to 224x224, and quantize to int8.

    Pipeline (matches training normalization):
      1. cv2.imread  (loads as BGR)
      2. cv2.resize to (224, 224) — stretch, no crop, default INTER_LINEAR
      3. cv2.cvtColor BGR -> RGB
      4. x_float = (pixel/255 - IMAGENET_MEAN) / IMAGENET_STD
      5. x_int8 = clip(round(x_float / FMAX * 127), -128, 127)

    The corresponding x_scale_0 in the Gemmini extraction script is FMAX/127.
    """
    import cv2
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Failed to load image: {img_path}")
    img = cv2.resize(img, (input_size, input_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Normalize: (pixel/255 - mean) / std
    img_f    = img.astype(np.float64) / 255.0
    img_norm = (img_f - IMAGENET_MEAN) / IMAGENET_STD

    # Quantize to INT8 with FMAX scale
    quantized = np.clip(np.round(img_norm / FMAX * 127), -128, 127).astype(np.int8)

    return quantized  # shape (224, 224, 3), dtype int8


def parse_labels_file(labels_path, labels_format="imagenet"):
    """Parse a labels file.

    labels_format:
      "imagenet" — lines like "ILSVRC2012_val_00000001.JPEG 65"
      "plain"    — lines with just an integer label
    """
    filenames = []
    labels = []
    with open(labels_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if labels_format == "plain":
                labels.append(int(line))
                filenames.append(None)
            else:
                parts = line.split()
                filenames.append(parts[0])
                labels.append(int(parts[1]))
    return filenames, labels


def visualize_preprocessed(image_paths, imagenet_dir, num_vis=5):
    """Show first N preprocessed images (dequantized back to uint8 for display)."""
    import matplotlib.pyplot as plt

    n = min(num_vis, len(image_paths))
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for i in range(n):
        img_path = os.path.join(imagenet_dir, image_paths[i])
        quantized = preprocess_image(img_path)
        # Dequantize: int8 * FMAX/127 * std + mean -> float -> uint8
        x_f = quantized.astype(np.float64) * (FMAX / 127.0)
        x_f = x_f * IMAGENET_STD + IMAGENET_MEAN
        display = np.clip(np.round(x_f * 255), 0, 255).astype(np.uint8)
        axes[i].imshow(display)
        axes[i].set_title(image_paths[i], fontsize=8)
        axes[i].axis("off")

    plt.suptitle("Preprocessed images (dequantized for display)", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(imagenet_dir, "..", "preprocessed_preview.png"), dpi=150)
    print(f"Saved preview to preprocessed_preview.png")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Prepare ImageNet images for Gemmini")
    parser.add_argument("--imagenet-dir", required=True,
                        help="Directory containing validation JPEG images")
    parser.add_argument("--labels-file", required=True,
                        help="Labels file (see --labels-format)")
    parser.add_argument("--labels-format", default="imagenet", choices=["imagenet", "plain"],
                        help="'imagenet': 'filename label' per line; 'plain': one int per line")
    parser.add_argument("--num-images", type=int, default=50000,
                        help="Number of images to process (default: 50000)")
    parser.add_argument("--output-dir", default=".",
                        help="Output directory for images.bin and labels.txt")
    parser.add_argument("--output-prefix", default="imagenet_val",
                        help="Prefix for output files (default: imagenet_val)")
    parser.add_argument("--visualize", type=int, default=0,
                        help="Show first N preprocessed images (e.g. --visualize 5)")
    args = parser.parse_args()

    filenames, all_labels = parse_labels_file(args.labels_file, args.labels_format)
    num_images = min(args.num_images, len(all_labels))
    print(f"Processing {num_images} images...")

    # If labels_format is "plain", list image files sorted by name
    if args.labels_format == "plain":
        image_files = sorted([
            f for f in os.listdir(args.imagenet_dir)
            if f.lower().endswith((".jpeg", ".jpg", ".png"))
        ])
        if len(image_files) < num_images:
            print(f"Warning: only {len(image_files)} images found in {args.imagenet_dir}")
            num_images = len(image_files)
    else:
        image_files = filenames

    # Visualize first N images if requested
    if args.visualize > 0:
        visualize_preprocessed(image_files, args.imagenet_dir, args.visualize)

    os.makedirs(args.output_dir, exist_ok=True)
    bin_path = os.path.join(args.output_dir, f"{args.output_prefix}_{num_images}.bin")
    lbl_path = os.path.join(args.output_dir, f"{args.output_prefix}_{num_images}_labels.txt")

    with open(bin_path, "wb") as bin_f, open(lbl_path, "w") as lbl_f:
        for i in range(num_images):
            img_name = image_files[i]
            img_path = os.path.join(args.imagenet_dir, img_name)

            if not os.path.isfile(img_path):
                print(f"Warning: {img_path} not found, skipping")
                continue

            quantized = preprocess_image(img_path)
            assert quantized.shape == (224, 224, 3) and quantized.dtype == np.int8

            # Write raw int8 bytes (HWC layout, contiguous)
            bin_f.write(quantized.tobytes())
            lbl_f.write(f"{all_labels[i]}\n")

            if (i + 1) % 1000 == 0 or i == 0:
                print(f"  [{i+1}/{num_images}] processed")

    file_size_mb = os.path.getsize(bin_path) / (1024 * 1024)
    print(f"\nDone!")
    print(f"  {bin_path}  ({file_size_mb:.1f} MB, {num_images} images)")
    print(f"  {lbl_path}")
    print(f"\nTo use with mobilenet_v1.c, set:")
    print(f"  #define NUM_IMAGES {num_images}")
    print(f'  Image file: "{os.path.basename(bin_path)}"')
    print(f'  Label file: "{os.path.basename(lbl_path)}"')


if __name__ == "__main__":
    main()
