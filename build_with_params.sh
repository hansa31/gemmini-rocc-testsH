#!/usr/bin/env bash
# Build a single test target using a specific Gemmini parameter configuration.
#
# Usage: ./build_with_params.sh <param_type> <subdir> <target>
#
# Examples:
#   ./build_with_params.sh FP16 imagenet resnet50-baremetal
#   ./build_with_params.sh FP8  bareMetalC matmul-baremetal
#   ./build_with_params.sh BF16 imagenet resnet50-linux

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARAMS_DIR="$SCRIPT_DIR/include/GemminiParams"
PARAMS_DEST="$SCRIPT_DIR/include/gemmini_params.h"

# --- List available param types ------------------------------------------------
list_available() {
    echo "Available param types:"
    for d in "$PARAMS_DIR"/*/; do
        name="$(basename "$d")"
        if [ -f "$d/gemmini_params.h" ]; then
            echo "  $name"
        else
            echo "  $name  (no gemmini_params.h — empty)"
        fi
    done
}

# --- Usage ---------------------------------------------------------------------
if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <param_type> <subdir> <target>"
    echo ""
    echo "Subdirs: imagenet, bareMetalC, mlps, transformers"
    echo ""
    list_available
    echo ""
    echo "Examples:"
    echo "  $0 FP16 imagenet resnet50-baremetal"
    echo "  $0 FP8  bareMetalC matmul-baremetal"
    exit 1
fi

PARAM_TYPE="$1"
SUBDIR="$2"
TARGET="$3"

PARAM_SRC="$PARAMS_DIR/$PARAM_TYPE/gemmini_params.h"

# --- Validate param type -------------------------------------------------------
if [ ! -d "$PARAMS_DIR/$PARAM_TYPE" ]; then
    echo "Error: Unknown param type '$PARAM_TYPE'"
    echo ""
    list_available
    exit 1
fi

if [ ! -f "$PARAM_SRC" ]; then
    echo "Error: No gemmini_params.h found in $PARAMS_DIR/$PARAM_TYPE/"
    echo ""
    list_available
    exit 1
fi

# --- Swap params ---------------------------------------------------------------
echo "==> Using $PARAM_TYPE params: $PARAM_SRC"
cp "$PARAM_SRC" "$PARAMS_DEST"
echo "==> Replaced include/gemmini_params.h"

# --- Build (reuse existing build logic) ----------------------------------------
BUILD_DIR="$SCRIPT_DIR/build"

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

if [[ $(which riscv64-unknown-linux-gnu-gcc 2>/dev/null) ]]; then
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
echo "Built ($PARAM_TYPE): $BUILD_DIR/$SUBDIR/$TARGET"
