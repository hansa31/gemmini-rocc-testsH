#!/usr/bin/env bash
# ============================================================
# install_deps.sh — Install all Python dependencies into the
# ImageNet conda environment for BERT-Tiny SST-2 setup.
# Usage:  bash install_deps.sh
# ============================================================
set -euo pipefail

ENV_NAME="ImageNet"

echo "=============================="
echo " Installing deps in conda env: $ENV_NAME"
echo "=============================="

# Install PyTorch (CPU-only is enough for weight export; add CUDA index if you
# need GPU inference).  transformers, datasets, and tokenizers are from HuggingFace.
conda run -n "$ENV_NAME" pip install \
    torch --extra-index-url https://download.pytorch.org/whl/cpu \
    transformers \
    datasets \
    tokenizers \
    numpy

echo ""
echo "=== Verification ==="
conda run -n "$ENV_NAME" python -c "
import torch, transformers, datasets, numpy as np
print(f'  torch        {torch.__version__}')
print(f'  transformers {transformers.__version__}')
print(f'  datasets     {datasets.__version__}')
print(f'  numpy        {np.__version__}')
print('All dependencies OK.')
"

echo ""
echo "Done. You can now run the scripts in this directory with:"
echo "  conda run -n $ENV_NAME python script.py"
