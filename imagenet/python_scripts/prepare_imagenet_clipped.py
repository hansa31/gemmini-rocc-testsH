#!/usr/bin/env python3
"""
Prepare ImageNet validation images for Gemmini MobileNetV2 inference.
>>> CLIPPED / RESCALED VERSION <<<

Same pipeline as prepare_imagenet.py, but with an extra rescaling step
that compresses the int8 range to approximately [-70, 85] instead of
the full [-128, 127]. This matches the quantize_image() function in
the Imagenet2gemmini.ipynb notebook.

Purpose:
  Test whether the narrower range improves accuracy on this particular
  Gemmini MobileNetV2 weight export. Compare results with the standard
  prepare_imagenet.py output to decide which preprocessing is correct.

Output files use the suffix '_clipped' to distinguish them:
  imagenet_val_50000_clipped.bin
  imagenet_val_50000_clipped_labels.txt
"""

import argparse
import os
import sys

import numpy as np

try:
    from PIL import Image
except ImportError:
    print("Pillow is required: pip install Pillow")
    sys.exit(1)


# ---------- Preprocessing with rescaled quantization ----------

def preprocess_image(img_path, input_size=224):
    """Load an image, center-crop to square, resize, and quantize to int8
    with rescaling to a narrower target range.

    Preprocessing pipeline:
      1. Center-crop to square (crop longer dimension to match shorter)
      2. Resize to 224x224 with LANCZOS interpolation
      3. Quantize with rescaling:
         a. shifted = uint8_pixel - 128       (maps [0,255] -> [-128,127])
         b. scale = (85 - (-70)) / (127 - (-128))  ≈ 0.608
         c. rescaled = round(shifted * scale)  (maps to ~[-78, 77])
         d. Cast to int8
    """
    img = Image.open(img_path).convert("RGB")

    # Center-crop to square
    w, h = img.size
    short_edge = min(w, h)
    left = (w - short_edge) // 2
    top = (h - short_edge) // 2
    img = img.crop((left, top, left + short_edge, top + short_edge))

    # Resize to input_size x input_size
    img = img.resize((input_size, input_size), Image.LANCZOS)

    # Quantize with rescaling to narrower range
    pixels = np.array(img, dtype=np.int32)  # [0, 255]
    shifted = pixels - 128                   # [-128, 127]

    # Rescale: compress full int8 range to target range [-70, 85]
    target_min, target_max = -70, 85
    scale = (target_max - target_min) / (127 - (-128))  # ≈ 0.608
    rescaled = np.round(shifted * scale).astype(np.int8)

    return rescaled  # shape (224, 224, 3), dtype int8


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
    parser = argparse.ArgumentParser(
        description="Prepare ImageNet images for Gemmini (CLIPPED/RESCALED version)")
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
    args = parser.parse_args()

    filenames, all_labels = parse_labels_file(args.labels_file, args.labels_format)
    num_images = min(args.num_images, len(all_labels))
    print(f"Processing {num_images} images (CLIPPED/RESCALED mode)...")

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
    bin_path = os.path.join(args.output_dir, f"{args.output_prefix}_{num_images}_clipped.bin")
    lbl_path = os.path.join(args.output_dir, f"{args.output_prefix}_{num_images}_clipped_labels.txt")

    with open(bin_path, "wb") as bin_f, open(lbl_path, "w") as lbl_f:
        for i in range(num_images):
            img_name = image_files[i]
            img_path = os.path.join(args.imagenet_dir, img_name)

            if not os.path.isfile(img_path):
                print(f"Warning: {img_path} not found, skipping")
                continue

            quantized = preprocess_image(img_path)
            assert quantized.shape == (224, 224, 3) and quantized.dtype == np.int8

            bin_f.write(quantized.tobytes())
            lbl_f.write(f"{all_labels[i]}\n")

            if (i + 1) % 1000 == 0 or i == 0:
                print(f"  [{i+1}/{num_images}] processed")

    file_size_mb = os.path.getsize(bin_path) / (1024 * 1024)
    print(f"\nDone! (CLIPPED/RESCALED version)")
    print(f"  {bin_path}  ({file_size_mb:.1f} MB, {num_images} images)")
    print(f"  {lbl_path}")
    print(f"\nTo use with mobilenet_v1.c, update the #define filenames:")
    print(f'  #define IMAGES_BIN_FILE "{os.path.basename(bin_path)}"')
    print(f'  #define LABELS_TXT_FILE "{os.path.basename(lbl_path)}"')


if __name__ == "__main__":
    main()
