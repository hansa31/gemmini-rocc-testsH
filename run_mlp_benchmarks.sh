#!/bin/bash
# Runs the 5 MLP benchmark binaries and records per-layer cycles, total cycles,
# and wall time into mlp_bench_results.csv in the same directory as this script.

DIR="$(cd "$(dirname "$0")" && pwd)"
CSV_FILE="$DIR/mlp_bench_results.csv"

BINARIES=(
    mlp_bert_ffn-linux
    mlp_dlrm_bottom-linux
    mlp_dlrm_top-linux
    mlp_gpt2_ffn-linux
    mlp_lenet300-linux
)

# Write header (overwrites any previous run)
{
    echo "# Gemmini MLP Benchmark Results — $(date '+%Y-%m-%d %H:%M:%S')"
    echo "benchmark,total_cycles,wall_ns,layer_0_cycles,layer_1_cycles,layer_2_cycles,layer_3_cycles"
} > "$CSV_FILE"

for bin in "${BINARIES[@]}"; do
    BIN_PATH="$DIR/$bin"

    if [[ ! -x "$BIN_PATH" ]]; then
        echo "SKIP: $bin not found or not executable in $DIR" >&2
        continue
    fi

    echo "Running $bin ..."

    OUTPUT="$("$BIN_PATH" 2>&1)"
    EXIT_CODE=$?

    if [[ $EXIT_CODE -ne 0 ]]; then
        echo "WARNING: $bin exited with code $EXIT_CODE" >&2
    fi

    # Parse total cycles
    TOTAL_CYCLES=$(printf '%s\n' "$OUTPUT" | grep -i 'Overall cycles taken' | grep -oE '[0-9]+' | tail -1)

    if [[ -z "$TOTAL_CYCLES" ]]; then
        echo "WARNING: no cycle count found for $bin" >&2
        continue
    fi

    # Parse wall time printed by the binary (nanoseconds)
    WALL_NS=$(printf '%s\n' "$OUTPUT" | grep -i 'Wall time:' | grep -oE '[0-9]+' | tail -1)

    # Parse per-layer cycles (layer 0, 1, 2, 3) — empty string if layer absent
    L0=$(printf '%s\n' "$OUTPUT" | grep -i 'Cycles taken in layer 0' | grep -oE '[0-9]+' | tail -1)
    L1=$(printf '%s\n' "$OUTPUT" | grep -i 'Cycles taken in layer 1' | grep -oE '[0-9]+' | tail -1)
    L2=$(printf '%s\n' "$OUTPUT" | grep -i 'Cycles taken in layer 2' | grep -oE '[0-9]+' | tail -1)
    L3=$(printf '%s\n' "$OUTPUT" | grep -i 'Cycles taken in layer 3' | grep -oE '[0-9]+' | tail -1)

    # Strip the "-linux" suffix for the benchmark name
    BENCH_NAME="${bin%-linux}"

    ROW="$BENCH_NAME,$TOTAL_CYCLES,$WALL_NS,$L0,$L1,$L2,$L3"
    echo "$ROW" >> "$CSV_FILE"
    echo "  -> $ROW"
done

echo ""
echo "Results saved to: $CSV_FILE"
