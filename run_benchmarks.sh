#!/bin/sh
# Runs the 5 bench_gemm benchmarks and records clock cycles + wall time into
# bench_results.csv in the same directory as this script and the binaries.

DIR="$(cd "$(dirname "$0")" && pwd)"
BIN_DIR="$DIR/bareMetalC"
CSV_FILE="$DIR/bench_results.csv"

# Write header (overwrites any previous run)
echo "# Gemmini GEMM Benchmark Results -- $(date '+%Y-%m-%d %H:%M:%S')" > "$CSV_FILE"
echo "benchmark,M,N,K,flops,min_cycles,avg_cycles,max_cycles,min_wall_ns,avg_wall_ns,max_wall_ns" >> "$CSV_FILE"

for bin in bench_gemm_small-linux bench_gemm_medium-linux bench_gemm_large-linux bench_gemm_resnet-linux bench_gemm_bert-linux; do
    BIN_PATH="$BIN_DIR/$bin"

    if [ ! -x "$BIN_PATH" ]; then
        echo "SKIP: $bin not found or not executable in $BIN_DIR" >&2
        continue
    fi

    echo "Running $bin ..."
    OUTPUT="$("$BIN_PATH" 2>&1)"
    EXIT_CODE=$?

    if [ $EXIT_CODE -ne 0 ]; then
        echo "WARNING: $bin exited with code $EXIT_CODE" >&2
    fi

    # Each binary prints exactly one line starting with "CSV,"
    CSV_LINE="$(printf '%s\n' "$OUTPUT" | grep '^CSV,' | tail -1)"

    if [ -z "$CSV_LINE" ]; then
        echo "WARNING: no CSV output line found for $bin" >&2
        continue
    fi

    # Strip the "CSV," sentinel and append the data row
    ROW=$(echo "$CSV_LINE" | sed 's/^CSV,//')
    echo "$ROW" >> "$CSV_FILE"
    echo "  -> $ROW"
done

echo ""
echo "Results saved to: $CSV_FILE"
