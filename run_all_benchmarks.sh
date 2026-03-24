#!/bin/sh
# run_all_benchmarks.sh — Master script to run all Gemmini benchmark suites
# sequentially and collect all CSVs + logs into a single results directory.
#
# Usage:
#   ./run_all_benchmarks.sh <config_name>
#
# Examples:
#   ./run_all_benchmarks.sh baseline
#   ./run_all_benchmarks.sh 16x16_mesh_int8
#   ./run_all_benchmarks.sh dse_run3_large_sp
#
# Output:
#   results/<YYYYMMDD_HHMMSS>_<config_name>/
#     ├── master.log              — full console output
#     ├── bench_gemm_results.csv
#     ├── mlp_results.csv
#     ├── imagenet_results.csv
#     ├── transformer_results.csv
#     ├── imagenet_logs/          — per-model logs
#     └── transformer_logs/       — per-model logs

set -eu

# ── Parse config name ──────────────────────────────────────────────────────
if [ $# -lt 1 ]; then
    echo "Usage: $0 <config_name>"
    echo ""
    echo "  config_name: A short label for the Gemmini configuration being tested."
    echo "               This is saved in the results directory name and in the CSVs."
    echo ""
    echo "Examples:"
    echo "  $0 baseline"
    echo "  $0 16x16_mesh_int8"
    echo "  $0 dse_run3_large_sp"
    exit 1
fi

# Strip leading dashes so both "./run_all.sh baseline" and
# "./run_all.sh --baseline" work the same way.
CONFIG=$(echo "$1" | sed 's/^--//;s/^-//')

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_DIR="${SCRIPT_DIR}/results/${TIMESTAMP}_${CONFIG}"

mkdir -p "${RESULTS_DIR}"

MASTER_LOG="${RESULTS_DIR}/master.log"

echo "=========================================================="
echo "         Gemmini Full Benchmark Suite"
echo "=========================================================="
echo "  Config:     ${CONFIG}"
echo "  Timestamp:  ${TIMESTAMP}"
echo "  Results:    ${RESULTS_DIR}"
echo "=========================================================="
echo ""

SUITE_PASS=0
SUITE_FAIL=0

# ── Helper: run a sub-script and collect its outputs ───────────────────────
run_suite() {
    SUITE_NAME="$1"
    SCRIPT_PATH="$2"
    SUITE_LOG="${RESULTS_DIR}/${SUITE_NAME}.log"

    echo "----------------------------------------------------------"
    echo "  SUITE: ${SUITE_NAME}"
    echo "----------------------------------------------------------"

    if [ ! -f "${SCRIPT_PATH}" ]; then
        echo "[SKIP] ${SUITE_NAME} -- script not found: ${SCRIPT_PATH}"
        SUITE_FAIL=$((SUITE_FAIL + 1))
        return
    fi

    sh "${SCRIPT_PATH}" 2>&1 | tee "${SUITE_LOG}"

    echo ""
    SUITE_PASS=$((SUITE_PASS + 1))
}

# ═══════════════════════════════════════════════════════════════════════════
# Run everything inside a subshell piped to tee for master log
# ═══════════════════════════════════════════════════════════════════════════
{

# 1. GEMM benchmarks
run_suite "bench_gemm" "${SCRIPT_DIR}/run_benchmarks.sh"

if [ -f "${SCRIPT_DIR}/bench_results.csv" ]; then
    cp "${SCRIPT_DIR}/bench_results.csv" "${RESULTS_DIR}/bench_gemm_results.csv"
fi

# 2. MLP benchmarks
run_suite "mlp" "${SCRIPT_DIR}/run_mlp_benchmarks.sh"

if [ -f "${SCRIPT_DIR}/mlp_bench_results.csv" ]; then
    cp "${SCRIPT_DIR}/mlp_bench_results.csv" "${RESULTS_DIR}/mlp_results.csv"
fi

# 3. ImageNet / CIFAR-10 CNN benchmarks
run_suite "imagenet" "${SCRIPT_DIR}/run_imagenet_benchmarks.sh"

LATEST_IMGNET_CSV=$(ls -t "${SCRIPT_DIR}"/imagenet_results_*.csv 2>/dev/null | head -1)
if [ -n "${LATEST_IMGNET_CSV}" ]; then
    cp "${LATEST_IMGNET_CSV}" "${RESULTS_DIR}/imagenet_results.csv"
fi

if [ -d "${SCRIPT_DIR}/imagenet_logs" ]; then
    cp -r "${SCRIPT_DIR}/imagenet_logs" "${RESULTS_DIR}/imagenet_logs"
fi

# 4. Transformer benchmarks
run_suite "transformer" "${SCRIPT_DIR}/run_transformer_benchmarks.sh"

LATEST_XFMR_CSV=$(ls -t "${SCRIPT_DIR}"/transformer_results_*.csv 2>/dev/null | head -1)
if [ -n "${LATEST_XFMR_CSV}" ]; then
    cp "${LATEST_XFMR_CSV}" "${RESULTS_DIR}/transformer_results.csv"
fi

if [ -d "${SCRIPT_DIR}/transformer_logs" ]; then
    cp -r "${SCRIPT_DIR}/transformer_logs" "${RESULTS_DIR}/transformer_logs"
fi

# ═══════════════════════════════════════════════════════════════════════════
# Write a run summary file
# ═══════════════════════════════════════════════════════════════════════════
SUMMARY="${RESULTS_DIR}/run_summary.txt"
cat > "${SUMMARY}" <<EOF
Gemmini Benchmark Run Summary
=============================
Config:     ${CONFIG}
Timestamp:  ${TIMESTAMP}
Date:       $(date)

Suites run: $((SUITE_PASS + SUITE_FAIL))
  Passed:   ${SUITE_PASS}
  Failed:   ${SUITE_FAIL}

Files collected:
$(ls -la "${RESULTS_DIR}/" 2>/dev/null)
EOF

echo ""
echo "=========================================================="
echo "                    Run Complete"
echo "=========================================================="
echo "  Config:     ${CONFIG}"
echo "  Suites:     $((SUITE_PASS + SUITE_FAIL)) total (${SUITE_PASS} passed, ${SUITE_FAIL} failed)"
echo "  Results:    ${RESULTS_DIR}"
echo "=========================================================="
echo ""
echo "Contents of results directory:"
ls -la "${RESULTS_DIR}/"

} 2>&1 | tee -a "${MASTER_LOG}"
