#!/usr/bin/env bash
# Build a single test target from a specific subdirectory.
# Usage: ./build_single.sh <subdir> <target_suffix>
# Example: ./build_single.sh imagenet resnet50-baremetal
# Example: ./build_single.sh bareMetalC matmul-baremetal

set -e

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <subdir> <target>"
    echo ""
    echo "Subdirs: imagenet, bareMetalC, mlps, transformers"
    echo ""
    echo "Examples:"
    echo "  $0 imagenet resnet50-baremetal"
    echo "  $0 imagenet resnet50-linux"
    echo "  $0 bareMetalC matmul-baremetal"
    exit 1
fi

SUBDIR="$1"
TARGET="$2"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"

# Run autoconf + configure if build dir doesn't exist yet
if [ ! -d "$BUILD_DIR" ]; then
    echo "Build directory not found, running configure..."
    cd "$SCRIPT_DIR"
    autoconf && mkdir -p build && cd build && ../configure
    cd "$SCRIPT_DIR"
fi

mkdir -p "$BUILD_DIR/$SUBDIR"

VARS="abs_top_srcdir=$SCRIPT_DIR \
      src_dir=$SCRIPT_DIR/$SUBDIR \
      XLEN=64 \
      PREFIX=examples-$SUBDIR"

if [[ $(which riscv64-unknown-linux-gnu-gcc) ]]; then
    make -C "$BUILD_DIR/$SUBDIR" \
        -f "$SCRIPT_DIR/$SUBDIR/Makefile" \
        $VARS \
        "$TARGET"
else
    VARS="$VARS BAREMETAL_ONLY=1"
    make -C "$BUILD_DIR/$SUBDIR" \
        -f "$SCRIPT_DIR/$SUBDIR/Makefile" \
        $VARS \
        "$TARGET"
fi

echo ""
echo "Built: $BUILD_DIR/$SUBDIR/$TARGET"
