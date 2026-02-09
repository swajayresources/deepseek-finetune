"""
DeepSeek-Coder 1.3B - Full Fine-Tuning with DeepSpeed ZeRO-2
Production-grade training script
"""

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import load_from_disk

MODEL_NAME = "deepseek-ai/deepseek-coder-1.3b-base"

print("="*80)
print("DeepSeek-Coder 1.3B Fine-Tuning")
print("="*80)

# Load tokenizer
print("\n[1/5] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token  # CRITICAL FIX

# Load model
print("[2/5] Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16
)

print(f"  Model parameters: {model.num_parameters() / 1e9:.2f}B")

# Load dataset
print("[3/5] Loading tokenized dataset...")
dataset = load_from_disk("data/tokenized")
print(f"  Train samples: {len(dataset['train'])}")

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Training arguments
print("[4/5] Configuring training...")
training_args = TrainingArguments(
    output_dir="checkpoints",
    overwrite_output_dir=True,

    # Training
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,

    # Optimizer (HF is source of truth)
    learning_rate=2e-5,
    adam_beta1=0.9,
    adam_beta2=0.95,  # DeepSeek-specific
    weight_decay=0.1,

    # Scheduler
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,

    # Logging
    logging_steps=50,
    logging_dir="logs",
    report_to="none",  # Change to "wandb" if you want W&B

    # Checkpointing (SAFE for 132GB disk)
    save_strategy="steps",
    save_steps=10000,
    save_total_limit=2,
    save_only_model=True,

    # Memory & Precision
    bf16=True,
    gradient_checkpointing=False,
    
    # DeepSpeed
    deepspeed="configs/ds_zero2.json",
    
    # Infrastructure
    remove_unused_columns=False,
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
    
    # Stability
    max_grad_norm=1.0,
    seed=42,
)

# Initialize trainer
print("[5/5] Initializing trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    data_collator=data_collator,
)

# Disk space check
import shutil
disk = shutil.disk_usage(".")
free_gb = disk.free / (1024**3)
print(f"\n✓ Disk space: {free_gb:.1f} GB available")
if free_gb < 50:
    print(f"⚠️  WARNING: Only {free_gb:.1f}GB free (recommend 50GB+)")

# Start training
print("\n" + "="*80)
print("STARTING TRAINING")
print("="*80)
print(f"Total steps: ~{len(dataset['train']) // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps)}")
print(f"Checkpoints: Every {training_args.save_steps} steps")
print("="*80 + "\n")

trainer.train()

# Save final model
print("\n" + "="*80)
print("TRAINING COMPLETE")
print("="*80)
trainer.save_model("checkpoints/final")
print("\n✓ Model saved to: checkpoints/final")
