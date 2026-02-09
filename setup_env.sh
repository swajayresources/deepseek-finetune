#!/bin/bash
# Environment Setup for DeepSeek-Coder Fine-Tuning

set -e

echo "=================================================="
echo "DeepSeek-Coder 1.3B - Environment Setup"
echo "=================================================="

# Uninstall conflicting packages
echo ""
echo "[1/4] Removing conflicting packages..."
pip uninstall -y numpy mpi4py bitsandbytes peft unsloth 2>/dev/null || true

# Install correct NumPy first
echo ""
echo "[2/4] Installing NumPy 1.26.4..."
pip install numpy==1.26.4

# Install required packages (UPDATED VERSIONS for Python 3.12)
echo ""
echo "[3/4] Installing dependencies..."
pip install \
  torch==2.4.0 \
  transformers==4.44.0 \
  datasets==2.20.0 \
  deepspeed==0.15.0 \
  accelerate==0.34.0 \
  sentencepiece \
  tiktoken \
  evaluate \
  wandb

# Verify installations
echo ""
echo "[4/4] Verifying installation..."
python3 -c "import numpy; print(f'NumPy: {numpy.__version__}')"
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python3 -c "import deepspeed; print(f'DeepSpeed: {deepspeed.__version__}')"

echo ""
echo "✓ Environment setup complete!"
echo ""
echo "Next steps:"
echo "  1. Download dataset: huggingface-cli download code_search_net python --repo-type dataset --local-dir data/csn_python"
echo "  2. Run preprocessing: python scripts/preprocess.py"
echo "  3. Start training: bash launch_training.sh"
