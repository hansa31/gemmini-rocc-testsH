#!/bin/sh
# run_transformer_benchmarks.sh — Run transformer benchmarks and record metrics to CSV.
#
# On FPGA: place script in /Test/, binary in /Test/transformers/
# Binary needs its .bin/.txt data files in the cwd (transformers/).

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BIN_DIR="${SCRIPT_DIR}/transformers"
LOG_DIR="${SCRIPT_DIR}/transformer_logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
CSV_FILE="${SCRIPT_DIR}/transformer_results_${TIMESTAMP}.csv"

mkdir -p "${LOG_DIR}"

# CSV header
echo "model,dataset,total_examples,accuracy_pct,macro_f1,min_example_cycles,avg_example_cycles,min_example_wall_ns,avg_example_wall_ns,examples_per_sec" > "${CSV_FILE}"

echo "========================================"
echo " Transformer Benchmark Runner"
echo " Output CSV: ${CSV_FILE}"
echo " Logs dir:   ${LOG_DIR}"
echo "========================================"
echo ""

PASS=0
FAIL=0

for bench in bert-tiny-sst2-stream-linux; do
    BIN_PATH="${BIN_DIR}/${bench}"
    LOG_SUFFIX=$(echo "${bench}" | sed 's/\.linux//')
    LOG_FILE="${LOG_DIR}/${LOG_SUFFIX}_${TIMESTAMP}.log"

    if [ ! -f "${BIN_PATH}" ]; then
        echo "[SKIP] ${bench} -- binary not found at ${BIN_PATH}"
        FAIL=$((FAIL + 1))
        continue
    fi

    echo "[RUN]  ${bench} ..."
    echo "       Log: ${LOG_FILE}"

    # Run from BIN_DIR so binary can find its .bin/.txt data files
    (cd "${BIN_DIR}" && "./${bench}" 2>&1) | tee "${LOG_FILE}"

    CSV_LINE=$(grep "^CSV," "${LOG_FILE}" | tail -1)

    if [ -z "${CSV_LINE}" ]; then
        echo "[FAIL] ${bench} -- no CSV line found in output"
        FAIL=$((FAIL + 1))
        continue
    fi

    ROW=$(echo "${CSV_LINE}" | sed 's/^CSV,//')
    echo "${ROW}" >> "${CSV_FILE}"
    echo "[DONE] ${bench}"
    echo ""
    PASS=$((PASS + 1))
done

echo "========================================"
echo " Results: ${PASS} passed, ${FAIL} failed"
echo " CSV saved to: ${CSV_FILE}"
echo " Logs in:      ${LOG_DIR}"
echo "========================================"
echo ""
echo "CSV contents:"
cat "${CSV_FILE}"
