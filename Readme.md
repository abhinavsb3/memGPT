# MEMGPT

A GPT-2-style large language model (LLM) repository.This implementation includes full support for distributed training, sharded datasets, benchmark evaluation, and efficient text generation.

---

## 🔧 Features

- Transformer architecture based on GPT-2.
- Configurable training and model hyperparameters via JSON.
- Sharded dataset loading from `.npy` files.
- Mixed-precision training with `torch.autocast`.
- DDP (DistributedDataParallel) support.
- Evaluation support with HellaSwag.
- Modular codebase for easy extensibility.

---

## 📁 Project Structure

```bash
MEMGPT/
├── configs/
│   └── config.json                    # Model and training configuration
│
├── data/
│   ├── edu_fineweb/                   # Sharded training data
│   │   ├── train_000001.npy
│   │   ├── train_000002.npy
│   │   └── test_000001.npy
│   ├── hellaswag/
│   │   └── hellaswag_val.jsonl
│   └── fineweb.py                     # Dataset sharding/processing logic
│
├── model_core/
│   ├── __init__.py
│   ├── attention.py                   # Self-attention module
│   ├── model.py                       # GPT2 model architecture
│   ├── dataloader.py                  # DataLoader_1 class
│   └── training.py                    # train_nanogpt function
│
├── scripts/
│   ├── train.py                       # Entry point to start training
│   ├── evaluate.py                    # Run evaluation
│   └── generate.py                    # Generate text from trained model
│
├── evaluation/
│   ├── __init__.py
│   ├── hellaswag.py                   # HellaSwag dataset preparation
│   └── val_hellaswag.py               # HellaSwag scoring function
│
├── logs/
│   ├── log.txt                        # Training log file
│   └── model_xxxxx.pt                # Checkpoint files
│
├── .gitignore
├── README.md
├── requirements.txt
```

---

## ⚙️ Configuration

Edit `configs/config.json` to configure your model and training setup.

Example:
```json
{
  "model": {
    "block_size": 1024,
    "vocab_size": 50304,
    "n_layer": 12,
    "n_head": 12,
    "n_embd": 768
  },
  "training": {
    "max_steps": 19073,
    "log_dir": "log",
    "total_batch_size": 524288,
    "B": 64,
    "T": 1024,
    "max_lr": 0.0006,
    "min_lr": 0.00006,
    "warmup_steps": 715,
    "weight_decay": 0.1,
    "learning_rate": 0.0006
  }
}
```

---

## 🚀 Training

To start training the model:

```bash
python scripts/train.py
```

This script internally loads `train_nanogpt()` from `model_core/training.py` using the config in `configs/config.json`.

### Optional: Distributed Training

To run training across multiple GPUs using PyTorch DDP:

```bash
torchrun --nproc_per_node=NUM_GPUS scripts/train.py
```

Replace `NUM_GPUS` with the number of GPUs you want to use.

---

## 📊 Evaluation

To evaluate on HellaSwag:

```bash
python scripts/evaluate.py
```

Make sure the `hellaswag_val.jsonl` file is available under `data/hellaswag/`.

---

## ✍️ Text Generation

To generate text from a trained model:

```bash
python scripts/generate.py
```

Make sure to adjust the generation script to point to the correct checkpoint under the `logs/` directory.

---

## 🧩 Requirements

Install required packages:

```bash
pip install -r requirements.txt
```

---

## 📌 Notes

- Ensure your `.npy` sharded data is placed under `data/edu_fineweb/`.
- The log directory and checkpoints will be saved in `logs/`.
- The `DataLoader_1` handles distributed data loading.
- Supports `bfloat16` autocasting for better training efficiency.

---

## 📮 License

MIT License. Feel free to modify and build upon this for research or commercial use.

---

## 🙌 Acknowledgements

Inspired by Andrej Karpathy's nanoGPT. Special thanks to the Andrej Karpathy Youtube tutorials and open-source AI community.

