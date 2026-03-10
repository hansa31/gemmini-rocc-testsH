#!/usr/bin/env python3
"""
Prepare ImageNet validation images for Gemmini MobileNetV1 inference.

Reads ImageNet validation images, preprocesses them (resize, center-crop,
quantize to int8), and writes a flat binary file + labels text file
that mobilenet_v1.c can stream batch-by-batch.

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
"""

import argparse
import os
import struct
import sys

import numpy as np

try:
    from PIL import Image
except ImportError:
    print("Pillow is required: pip install Pillow")
    sys.exit(1)


# ---------- Preprocessing matching the original Gemmini quantization ----------

def preprocess_image(img_path, input_size=224, scale=1.0, zero_point=-128):
    """Load an image, resize/center-crop to input_size, and quantize to int8.

    The preprocessing mirrors the original Gemmini images.h quantization:
      1. Resize shortest side to 256, bilinear interpolation
      2. Center crop to 224x224
      3. Convert to float [0, 255]
      4. Apply quantization: int8_val = round(pixel * scale) + zero_point
         Default: scale=1, zero_point=-128 maps [0,255] -> [-128,127]
         This matches the signed int8 range seen in the original images.h.
    """
    img = Image.open(img_path).convert("RGB")

    # Resize shortest side to 256
    w, h = img.size
    if w < h:
        new_w = 256
        new_h = int(256 * h / w)
    else:
        new_h = 256
        new_w = int(256 * w / h)
    img = img.resize((new_w, new_h), Image.BILINEAR)

    # Center crop to input_size x input_size
    w, h = img.size
    left = (w - input_size) // 2
    top = (h - input_size) // 2
    img = img.crop((left, top, left + input_size, top + input_size))

    # Convert to numpy float, then quantize
    pixels = np.array(img, dtype=np.float32)  # shape (224, 224, 3), range [0, 255]
    quantized = np.round(pixels * scale + zero_point).astype(np.int32)
    quantized = np.clip(quantized, -128, 127).astype(np.int8)

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
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Quantization scale factor (default: 1.0)")
    parser.add_argument("--zero-point", type=int, default=-128,
                        help="Quantization zero point (default: -128, maps [0,255] to [-128,127])")
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

    os.makedirs(args.output_dir, exist_ok=True)
    bin_path = os.path.join(args.output_dir, f"{args.output_prefix}_{num_images}.bin")
    lbl_path = os.path.join(args.output_dir, f"{args.output_prefix}_{num_images}_labels.txt")

    image_size = 224 * 224 * 3  # bytes per image

    with open(bin_path, "wb") as bin_f, open(lbl_path, "w") as lbl_f:
        for i in range(num_images):
            img_name = image_files[i]
            img_path = os.path.join(args.imagenet_dir, img_name)

            if not os.path.isfile(img_path):
                print(f"Warning: {img_path} not found, skipping")
                continue

            quantized = preprocess_image(img_path, scale=args.scale, zero_point=args.zero_point)
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
