#!/usr/bin/env python3
"""
Verify a prepared images.bin file by reading back and displaying sample pixels.

Usage:
  python verify_images_bin.py images.bin labels.txt --num-images 200
"""

import argparse
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Verify Gemmini image binary file")
    parser.add_argument("images_bin", help="Path to images binary file")
    parser.add_argument("labels_txt", help="Path to labels text file")
    parser.add_argument("--num-images", type=int, required=True,
                        help="Number of images in the binary file")
    parser.add_argument("--show", type=int, default=5,
                        help="Number of images to preview (default: 5)")
    args = parser.parse_args()

    image_size = 224 * 224 * 3
    expected_bytes = args.num_images * image_size

    # Check file size
    import os
    actual_bytes = os.path.getsize(args.images_bin)
    print(f"Binary file: {args.images_bin}")
    print(f"  Expected size: {expected_bytes} bytes ({args.num_images} images x {image_size} bytes)")
    print(f"  Actual size:   {actual_bytes} bytes")

    if actual_bytes != expected_bytes:
        actual_count = actual_bytes // image_size
        print(f"  WARNING: size mismatch! Contains ~{actual_count} complete images")
    else:
        print(f"  OK: size matches")

    # Read and verify labels
    with open(args.labels_txt, "r") as f:
        labels = [int(line.strip()) for line in f if line.strip()]
    print(f"\nLabels file: {args.labels_txt}")
    print(f"  Number of labels: {len(labels)}")
    if len(labels) != args.num_images:
        print(f"  WARNING: label count ({len(labels)}) != image count ({args.num_images})")

    # Load and preview
    data = np.fromfile(args.images_bin, dtype=np.int8)
    n_show = min(args.show, args.num_images)

    print(f"\nFirst {n_show} images:")
    for i in range(n_show):
        offset = i * image_size
        img = data[offset:offset + image_size].reshape(224, 224, 3)
        print(f"\n  Image {i} (label={labels[i] if i < len(labels) else '?'}):")
        print(f"    Shape: {img.shape}, dtype: {img.dtype}")
        print(f"    Range: [{img.min()}, {img.max()}]")
        print(f"    Mean:  {img.mean():.1f}")
        # Print first 5 pixels (matches the debug prints in mobilenet_v1.c)
        print(f"    First 5 pixels (R,G,B):")
        for p in range(5):
            print(f"      Pixel {p}: {img[0, p, 0]}, {img[0, p, 1]}, {img[0, p, 2]}")

    print("\nDone.")


if __name__ == "__main__":
    main()
