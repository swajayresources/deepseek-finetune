#!/bin/bash
# Launch Training with Safety Checks

set -e

echo "=================================================="
echo "DeepSeek-Coder 1.3B - Training Launch"
echo "=================================================="

# Check disk space
echo ""
echo "[1/5] Checking disk space..."
FREE_GB=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
echo "  Available: ${FREE_GB}GB"

if [ "$FREE_GB" -lt 50 ]; then
    echo "  ⚠️  WARNING: Less than 50GB free!"
fi

# Check GPU
echo ""
echo "[2/5] Checking GPU..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Check files exist
echo ""
echo "[3/5] Checking required files..."
for file in "configs/ds_zero2.json" "scripts/train.py" "data/tokenized"; do
    if [ ! -e "$file" ]; then
        echo "  ✗ Missing: $file"
        exit 1
    fi
done
echo "  ✓ All files present"

# Set environment variables
echo ""
echo "[4/5] Setting environment variables..."
export DEEPSPEED_DISABLE_MPI=1
export DS_ACCELERATOR=cuda
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

echo "  DEEPSPEED_DISABLE_MPI=1"
echo "  CUDA_VISIBLE_DEVICES=0"

# Launch training
echo ""
echo "[5/5] Launching training..."
echo "=================================================="
echo ""

deepspeed --num_gpus=1 scripts/train.py

echo ""
echo "=================================================="
echo "Training completed!"
echo "=================================================="
```

---

## 📄 6. `.gitignore`
```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
ENV/
env/

# Data (too large for git)
data/
*.jsonl
*.arrow
*.parquet

# Model checkpoints (too large for git)
checkpoints/
*.bin
*.safetensors
*.pt
*.pth

# Logs
logs/
*.log
wandb/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Temporary files
*.tmp
temp/
tmp/
