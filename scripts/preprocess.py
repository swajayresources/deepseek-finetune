"""
Data preprocessing script for DeepSeek finetuning.
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict


def preprocess_data(input_file: str, output_file: str, max_length: int = 2048):
    """
    Preprocess raw data into the format expected by the training script.

    Args:
        input_file: Path to input data file
        output_file: Path to output JSON file
        max_length: Maximum sequence length
    """
    print(f"Preprocessing data from {input_file}...")

    processed_data = []

    # Read input data
    with open(input_file, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    # Process each example
    for idx, example in enumerate(raw_data):
        if 'instruction' in example and 'output' in example:
            # Format: instruction-output pairs
            processed_example = {
                "id": f"example_{idx}",
                "instruction": example['instruction'],
                "input": example.get('input', ''),
                "output": example['output']
            }
        elif 'text' in example:
            # Format: plain text
            processed_example = {
                "id": f"example_{idx}",
                "text": example['text']
            }
        else:
            print(f"Warning: Skipping example {idx} due to unknown format")
            continue

        processed_data.append(processed_example)

    # Save processed data
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=2, ensure_ascii=False)

    print(f"Processed {len(processed_data)} examples")
    print(f"Saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess data for DeepSeek finetuning")
    parser.add_argument("--input", type=str, required=True, help="Input data file")
    parser.add_argument("--output", type=str, required=True, help="Output JSON file")
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum sequence length")

    args = parser.parse_args()

    preprocess_data(args.input, args.output, args.max_length)


if __name__ == "__main__":
    main()
