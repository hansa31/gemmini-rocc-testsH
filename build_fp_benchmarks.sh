#!/usr/bin/env bash
# Build all FP benchmark (bench_gemm_*) and MLP linux binaries for a given
# Gemmini param configuration, and collect them into an output directory.
#
# Usage: ./build_fp_benchmarks.sh [PARAM_TYPE]
#
# If PARAM_TYPE is omitted, builds for ALL available param types that have
# a gemmini_params.h file (e.g. FP16, FP8).
#
# Output binaries are placed in:
#   build/output/<PARAM_TYPE>/bareMetalC/  — bench_gemm_* linux binaries
#   build/output/<PARAM_TYPE>/mlps/        — mlp_* linux binaries
#
# Examples:
#   ./build_fp_benchmarks.sh FP16      # build only FP16
#   ./build_fp_benchmarks.sh           # build all (FP16, FP8, ...)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARAMS_DIR="$SCRIPT_DIR/include/GemminiParams"
PARAMS_DEST="$SCRIPT_DIR/include/gemmini_params.h"
BUILD_DIR="$SCRIPT_DIR/build"
OUTPUT_DIR="$BUILD_DIR/output"

# ── Targets ────────────────────────────────────────────────────────────────────
# bareMetalC bench_gemm targets (linux binaries)
BENCH_TARGETS=(
    bench_gemm_small-linux
    bench_gemm_medium-linux
    bench_gemm_large-linux
    bench_gemm_bert-linux
    bench_gemm_resnet-linux
)

# mlps targets (linux binaries)
MLP_TARGETS=(
    mlp_lenet300-linux
    mlp_bert_ffn-linux
    mlp_gpt2_ffn-linux
    mlp_dlrm_bottom-linux
    mlp_dlrm_top-linux
)

# ── Helpers ────────────────────────────────────────────────────────────────────
list_available() {
    echo "Available param types:"
    for d in "$PARAMS_DIR"/*/; do
        name="$(basename "$d")"
        if [ -f "$d/gemmini_params.h" ]; then
            echo "  $name"
        else
            echo "  $name  (empty — skipped)"
        fi
    done
}

ensure_build_dir() {
    if [ ! -d "$BUILD_DIR" ]; then
        echo "==> Build directory not found, running configure..."
        cd "$SCRIPT_DIR"
        autoconf && mkdir -p build && cd build && ../configure
        cd "$SCRIPT_DIR"
    fi
}

build_subdir() {
    local subdir="$1"
    shift
    local targets=("$@")

    mkdir -p "$BUILD_DIR/$subdir"

    local vars="abs_top_srcdir=$SCRIPT_DIR \
                src_dir=$SCRIPT_DIR/$subdir \
                XLEN=64 \
                PREFIX=examples-$subdir"

    for target in "${targets[@]}"; do
        echo "    Building $target ..."
        make -C "$BUILD_DIR/$subdir" \
            -f "$SCRIPT_DIR/$subdir/Makefile" \
            $vars \
            "$target"
    done
}

build_for_param() {
    local param_type="$1"
    local param_src="$PARAMS_DIR/$param_type/gemmini_params.h"

    if [ ! -f "$param_src" ]; then
        echo "Skipping $param_type (no gemmini_params.h)"
        return
    fi

    echo ""
    echo "================================================================"
    echo "  Building for: $param_type"
    echo "================================================================"

    # Swap params
    cp "$param_src" "$PARAMS_DEST"
    echo "==> Replaced include/gemmini_params.h with $param_type"

    # Clean previous build objects so the new params take effect
    make -C "$BUILD_DIR/bareMetalC" -f "$SCRIPT_DIR/bareMetalC/Makefile" \
        abs_top_srcdir="$SCRIPT_DIR" src_dir="$SCRIPT_DIR/bareMetalC" \
        XLEN=64 PREFIX=examples-bareMetalC clean 2>/dev/null || true
    make -C "$BUILD_DIR/mlps" -f "$SCRIPT_DIR/mlps/Makefile" \
        abs_top_srcdir="$SCRIPT_DIR" src_dir="$SCRIPT_DIR/mlps" \
        XLEN=64 PREFIX=examples-mlps clean 2>/dev/null || true

    # Build bareMetalC bench_gemm targets
    echo ""
    echo "── bareMetalC bench_gemm (linux) ──"
    build_subdir bareMetalC "${BENCH_TARGETS[@]}"

    # Build mlps targets
    echo ""
    echo "── mlps (linux) ──"
    build_subdir mlps "${MLP_TARGETS[@]}"

    # Collect binaries into output/<PARAM_TYPE>/
    local out_bench="$OUTPUT_DIR/$param_type/bareMetalC"
    local out_mlps="$OUTPUT_DIR/$param_type/mlps"
    mkdir -p "$out_bench" "$out_mlps"

    for target in "${BENCH_TARGETS[@]}"; do
        cp "$BUILD_DIR/bareMetalC/$target" "$out_bench/"
    done
    for target in "${MLP_TARGETS[@]}"; do
        cp "$BUILD_DIR/mlps/$target" "$out_mlps/"
    done

    echo ""
    echo "==> $param_type binaries collected in $OUTPUT_DIR/$param_type/"
}

# ── Main ───────────────────────────────────────────────────────────────────────
ensure_build_dir

if [ "$#" -ge 1 ]; then
    # Build for a specific param type
    PARAM_TYPE="$1"
    if [ ! -d "$PARAMS_DIR/$PARAM_TYPE" ]; then
        echo "Error: Unknown param type '$PARAM_TYPE'"
        echo ""
        list_available
        exit 1
    fi
    build_for_param "$PARAM_TYPE"
else
    # Build for all available param types
    echo "No param type specified — building for all available types."
    list_available
    echo ""
    for d in "$PARAMS_DIR"/*/; do
        param_type="$(basename "$d")"
        build_for_param "$param_type"
    done
fi

# ── Summary ────────────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  Done! Output binaries:"
echo "================================================================"
find "$OUTPUT_DIR" -type f | sort | while read -r f; do
    echo "  $f"
done
