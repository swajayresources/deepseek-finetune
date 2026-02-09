"""
Training script for DeepSeek finetuning with DeepSpeed support.
"""

import os
import json
import argparse
from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset


@dataclass
class ModelArguments:
    """Arguments for model configuration."""
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store pretrained models downloaded from huggingface.co"}
    )


@dataclass
class DataArguments:
    """Arguments for data configuration."""
    data_path: str = field(
        metadata={"help": "Path to the training data (JSON file)"}
    )
    max_seq_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length"}
    )


def preprocess_function(examples, tokenizer, max_length):
    """Tokenize and prepare the dataset."""
    # Combine instruction and output if available
    if "instruction" in examples:
        texts = []
        for inst, inp, out in zip(examples["instruction"], examples.get("input", [""] * len(examples["instruction"])), examples["output"]):
            if inp:
                text = f"Instruction: {inst}\nInput: {inp}\nOutput: {out}"
            else:
                text = f"Instruction: {inst}\nOutput: {out}"
            texts.append(text)
    else:
        texts = examples["text"]

    # Tokenize
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt"
    )

    # For causal LM, labels are the same as input_ids
    tokenized["labels"] = tokenized["input_ids"].clone()

    return tokenized


def main():
    parser = argparse.ArgumentParser()

    # Model arguments
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default=None)

    # Data arguments
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--max_seq_length", type=int, default=2048)

    # Training arguments
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--evaluation_strategy", type=str, default="steps")
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--fp16", type=bool, default=True)
    parser.add_argument("--deepspeed", type=str, default=None)
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--local_rank", type=int, default=-1)

    args = parser.parse_args()

    # Load tokenizer and model
    print(f"Loading tokenizer and model from {args.model_name_or_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        trust_remote_code=True,
        torch_dtype=torch.float16 if args.fp16 else torch.float32
    )

    # Load dataset
    print(f"Loading dataset from {args.data_path}...")
    dataset = load_dataset("json", data_files=args.data_path, split="train")

    # Preprocess dataset
    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        lambda x: preprocess_function(x, tokenizer, args.max_seq_length),
        batched=True,
        remove_columns=dataset.column_names
    )

    # Split into train and eval
    split_dataset = tokenized_dataset.train_test_split(test_size=0.1)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        evaluation_strategy=args.evaluation_strategy,
        eval_steps=args.eval_steps,
        fp16=args.fp16,
        deepspeed=args.deepspeed,
        report_to=args.report_to,
        load_best_model_at_end=True,
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split_dataset["train"],
        eval_dataset=split_dataset["test"],
        data_collator=data_collator,
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save final model
    print(f"Saving final model to {args.output_dir}...")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)

    print("Training complete!")


if __name__ == "__main__":
    main()
