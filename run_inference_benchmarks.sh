#!/bin/sh
# run_inference_benchmarks.sh — Run only the ImageNet/CIFAR-10 CNN and
# Transformer benchmark suites, and collect all CSVs + logs into a
# single results directory.
#
# Usage:
#   ./run_inference_benchmarks.sh <config_name>
#
# Examples:
#   ./run_inference_benchmarks.sh baseline
#   ./run_inference_benchmarks.sh 16x16_mesh_int8

set -eu

if [ $# -lt 1 ]; then
    echo "Usage: $0 <config_name>"
    echo ""
    echo "  config_name: A short label for the Gemmini configuration being tested."
    echo ""
    echo "Examples:"
    echo "  $0 baseline"
    echo "  $0 16x16_mesh_int8"
    exit 1
fi

CONFIG=$(echo "$1" | sed 's/^--//;s/^-//')

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_DIR="${SCRIPT_DIR}/results/${TIMESTAMP}_${CONFIG}"

mkdir -p "${RESULTS_DIR}"

MASTER_LOG="${RESULTS_DIR}/master.log"

echo "=========================================================="
echo "      Gemmini Inference Benchmark Suite"
echo "=========================================================="
echo "  Config:     ${CONFIG}"
echo "  Timestamp:  ${TIMESTAMP}"
echo "  Results:    ${RESULTS_DIR}"
echo "=========================================================="
echo ""

SUITE_PASS=0
SUITE_FAIL=0

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

{

# 1. ImageNet / CIFAR-10 CNN benchmarks
run_suite "imagenet" "${SCRIPT_DIR}/run_imagenet_benchmarks.sh"

LATEST_IMGNET_CSV=$(ls -t "${SCRIPT_DIR}"/imagenet_results_*.csv 2>/dev/null | head -1)
if [ -n "${LATEST_IMGNET_CSV}" ]; then
    cp "${LATEST_IMGNET_CSV}" "${RESULTS_DIR}/imagenet_results.csv"
fi

if [ -d "${SCRIPT_DIR}/imagenet_logs" ]; then
    cp -r "${SCRIPT_DIR}/imagenet_logs" "${RESULTS_DIR}/imagenet_logs"
fi

# 2. Transformer benchmarks
run_suite "transformer" "${SCRIPT_DIR}/run_transformer_benchmarks.sh"

LATEST_XFMR_CSV=$(ls -t "${SCRIPT_DIR}"/transformer_results_*.csv 2>/dev/null | head -1)
if [ -n "${LATEST_XFMR_CSV}" ]; then
    cp "${LATEST_XFMR_CSV}" "${RESULTS_DIR}/transformer_results.csv"
fi

if [ -d "${SCRIPT_DIR}/transformer_logs" ]; then
    cp -r "${SCRIPT_DIR}/transformer_logs" "${RESULTS_DIR}/transformer_logs"
fi

# Write summary
SUMMARY="${RESULTS_DIR}/run_summary.txt"
cat > "${SUMMARY}" <<EOF
Gemmini Inference Benchmark Run Summary
=======================================
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
