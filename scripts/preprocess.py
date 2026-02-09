"""
DeepSeek-Coder 1.3B - Data Preprocessing
Tokenizes CodeSearchNet Python dataset
"""

from datasets import load_from_disk
from transformers import AutoTokenizer

MODEL_NAME = "deepseek-ai/deepseek-coder-1.3b-base"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=True
)
tokenizer.pad_token = tokenizer.eos_token

print("Loading dataset...")
ds = load_from_disk("data/csn_python")

def tokenize(example):
    """Tokenize code with EOS token"""
    text = example["func_code_string"] + tokenizer.eos_token
    return tokenizer(
        text,
        truncation=True,
        max_length=2048,
        padding=False,
    )

print("Tokenizing dataset...")
tokenized = ds.map(
    tokenize,
    remove_columns=ds["train"].column_names,
    num_proc=8,
    desc="Tokenizing CodeSearchNet Python"
)

print("Saving tokenized dataset...")
tokenized.save_to_disk("data/tokenized")

print("✓ Preprocessing complete!")
print(f"  Train samples: {len(tokenized['train'])}")
print(f"  Saved to: data/tokenized")
