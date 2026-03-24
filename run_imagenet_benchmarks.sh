#!/bin/sh
# run_imagenet_benchmarks.sh — Run 4 ImageNet/CIFAR-10 CNN benchmarks (500 images each)
# and record metrics to CSV + individual log files.
#
# On FPGA: place script in /Test/, binaries in /Test/imagenet/
# Each binary streams images from .bin files that must be in the cwd.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BIN_DIR="${SCRIPT_DIR}/imagenet"
LOG_DIR="${SCRIPT_DIR}/imagenet_logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
CSV_FILE="${SCRIPT_DIR}/imagenet_results_${TIMESTAMP}.csv"

# Create log directory
mkdir -p "${LOG_DIR}"

# CSV header -- unified format across all models
# (CIFAR-10 models output 0.00 for top-10 fields)
echo "model,dataset,input_size,total_images,final_top1_pct,final_top5_pct,final_top10_pct,best_window_top1_pct,best_window_top5_pct,min_batch_cycles,avg_batch_cycles,min_batch_wall_ns,avg_batch_wall_ns,images_per_sec" > "${CSV_FILE}"

echo "========================================"
echo " ImageNet/CIFAR-10 CNN Benchmark Runner"
echo " Output CSV: ${CSV_FILE}"
echo " Logs dir:   ${LOG_DIR}"
echo "========================================"
echo ""

PASS=0
FAIL=0

for bench in resnet50_cifar10_stream-linux mobilenet_cifar10_stream-linux resnet50_v1-linux mobilenet_v1-linux; do
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
    (cd "${BIN_DIR}" && "${BIN_PATH}" 2>&1) | tee "${LOG_FILE}"

    # Extract the CSV line (format: CSV,model,dataset,...)
    CSV_LINE=$(grep "^CSV," "${LOG_FILE}" | tail -1)

    if [ -z "${CSV_LINE}" ]; then
        echo "[FAIL] ${bench} -- no CSV line found in output"
        FAIL=$((FAIL + 1))
        continue
    fi

    # Strip the "CSV," prefix and append to CSV file
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
