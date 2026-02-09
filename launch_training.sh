#!/bin/bash

# DeepSeek Training Launch Script

# Activate virtual environment
source venv/bin/activate

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_PROJECT="deepseek-finetune"

# Training parameters
NUM_GPUS=4
MASTER_PORT=29500

# Launch distributed training with DeepSpeed
deepspeed --num_gpus=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    scripts/train.py \
    --deepspeed configs/ds_zero2.json \
    --model_name_or_path deepseek-ai/deepseek-coder-1.3b-base \
    --data_path data/train.json \
    --output_dir outputs/ \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "steps" \
    --eval_steps 100 \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 3 \
    --learning_rate 2e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --fp16 True \
    --report_to wandb

echo "Training complete!"
