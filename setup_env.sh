#!/bin/bash

# DeepSeek Finetune Environment Setup Script

echo "Setting up DeepSeek finetuning environment..."

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install DeepSpeed
pip install deepspeed

# Install Transformers and dependencies
pip install transformers
pip install datasets
pip install accelerate
pip install wandb
pip install sentencepiece
pip install protobuf

# Create necessary directories
mkdir -p checkpoints
mkdir -p logs
mkdir -p outputs

echo "Environment setup complete!"
echo "Activate the environment with: source venv/bin/activate"
