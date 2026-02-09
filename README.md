# DeepSeek Finetune

A repository for fine-tuning DeepSeek models with distributed training support.

## Setup

Run the setup script to prepare the environment:

```bash
./setup_env.sh
```

## Training

Launch training with:

```bash
./launch_training.sh
```

## Structure

- `configs/` - DeepSpeed and training configurations
- `scripts/` - Data preprocessing and training scripts
- `setup_env.sh` - Environment setup script
- `launch_training.sh` - Training launcher script

## Requirements

- Python 3.8+
- PyTorch
- DeepSpeed
- Transformers

## License

MIT
