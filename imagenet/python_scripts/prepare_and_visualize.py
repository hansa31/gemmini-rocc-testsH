#!/usr/bin/env python3
"""
Prepare ImageNet images for Gemmini MobileNetV2 — matching the notebook preprocessing.

Uses the SAME pipeline as Imagenet2gemmini.ipynb's create_gemmini_image_header():
  1. cv2.imread → BGR2RGB
  2. cv2.resize(img, (224, 224))  — simple stretch, NO center crop
  3. quantize: int8 = clip(uint8 - 128, -128, 127)

Also visualises the first N preprocessed images (reconstructed back to uint8).

Usage:
  python prepare_and_visualize.py \
      --imagenet-dir /home/hansa/Downloads/ILSVRC2012_img_val \
      --labels-file /path/to/val_correct.txt \
      --num-images 50000 \
      --output-dir /home/hansa/Downloads \
      --visualize 5
"""

import argparse
import os
import sys

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving PNG
import matplotlib.pyplot as plt

# --------------- Preprocessing (matches notebook exactly) ---------------

def preprocess_image(img_path, input_size=224):
    """Load, stretch-resize to 224x224, quantize to int8.

    This matches the notebook's create_gemmini_image_header pipeline:
      cv2.imread → BGR2RGB → cv2.resize(224,224) → uint8-128
    """
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"cv2 could not read: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (input_size, input_size))
    quantized = np.clip(img_resized.astype(np.int32) - 128, -128, 127).astype(np.int8)
    return quantized  # (224, 224, 3), int8


def reconstruct_for_display(img_int8):
    """Reverse the quantization for visual inspection: int8 + 128 → uint8."""
    return np.clip(img_int8.astype(np.int32) + 128, 0, 255).astype(np.uint8)


# --------------- Labels parsing ---------------

def parse_labels_file(labels_path, labels_format="imagenet"):
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


# --------------- Visualisation ---------------

def visualize_images(images_int8, titles, save_path):
    """Show a row of reconstructed preprocessed images and save to PNG."""
    n = len(images_int8)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, img, title in zip(axes, images_int8, titles):
        display = reconstruct_for_display(img)
        ax.imshow(display)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    plt.suptitle("Preprocessed images (cv2 stretch-resize, uint8-128)", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Visualisation saved to: {save_path}")


# --------------- Main ---------------

def main():
    parser = argparse.ArgumentParser(
        description="Prepare ImageNet for Gemmini (notebook-matching cv2 pipeline)")
    parser.add_argument("--imagenet-dir", required=True,
                        help="Directory with validation JPEG images")
    parser.add_argument("--labels-file", required=True,
                        help="Labels file (see --labels-format)")
    parser.add_argument("--labels-format", default="imagenet",
                        choices=["imagenet", "plain"],
                        help="'imagenet': 'filename label'; 'plain': one int per line")
    parser.add_argument("--num-images", type=int, default=50000,
                        help="Number of images to process")
    parser.add_argument("--output-dir", default=".",
                        help="Output directory for .bin, labels, and visualisation")
    parser.add_argument("--output-prefix", default="imagenet_val",
                        help="Prefix for output files")
    parser.add_argument("--visualize", type=int, default=5,
                        help="Number of images to visualise (0 to skip)")
    args = parser.parse_args()

    filenames, all_labels = parse_labels_file(args.labels_file, args.labels_format)
    num_images = min(args.num_images, len(all_labels))
    print(f"Processing {num_images} images with cv2 stretch-resize pipeline...")

    # If plain format, discover image files by sorted name
    if args.labels_format == "plain":
        image_files = sorted([
            f for f in os.listdir(args.imagenet_dir)
            if f.lower().endswith((".jpeg", ".jpg", ".png"))
        ])
        if len(image_files) < num_images:
            print(f"Warning: only {len(image_files)} images found")
            num_images = min(num_images, len(image_files))
    else:
        image_files = filenames

    os.makedirs(args.output_dir, exist_ok=True)
    bin_path = os.path.join(args.output_dir, f"{args.output_prefix}_{num_images}.bin")
    lbl_path = os.path.join(args.output_dir, f"{args.output_prefix}_{num_images}_labels.txt")

    vis_images = []        # collect first N for visualisation
    vis_titles = []

    with open(bin_path, "wb") as bin_f, open(lbl_path, "w") as lbl_f:
        for i in range(num_images):
            img_name = image_files[i]
            img_path = os.path.join(args.imagenet_dir, img_name)

            if not os.path.isfile(img_path):
                print(f"Warning: {img_path} not found, skipping")
                continue

            quantized = preprocess_image(img_path)

            # Print stats for first few images
            if i < 5:
                vals = quantized.flatten()
                print(f"  Image {i} ({img_name}): "
                      f"range=[{vals.min()},{vals.max()}], "
                      f"mean={vals.mean():.1f}, "
                      f"first6={list(quantized[0,0,:])}")

            # Collect for visualisation
            if len(vis_images) < args.visualize:
                vis_images.append(quantized)
                vis_titles.append(f"#{i}: {img_name[:30]}\nlbl={all_labels[i]}")

            bin_f.write(quantized.tobytes())
            lbl_f.write(f"{all_labels[i]}\n")

            if (i + 1) % 1000 == 0 or i == 0:
                print(f"  [{i+1}/{num_images}] processed")

    file_size_mb = os.path.getsize(bin_path) / (1024 * 1024)
    print(f"\nDone!")
    print(f"  {bin_path}  ({file_size_mb:.1f} MB, {num_images} images)")
    print(f"  {lbl_path}")

    # Visualise
    if args.visualize > 0 and vis_images:
        vis_path = os.path.join(args.output_dir,
                                f"{args.output_prefix}_preview.png")
        visualize_images(vis_images, vis_titles, vis_path)


if __name__ == "__main__":
    main()
