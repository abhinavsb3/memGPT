
# 🧠 Memorizing Transformer with Grouped Query Attention

An extended GPT-style 118m param model that integrates the key ideas from "Memorizing Transformers" (Wu et al., 2022) with my own modification for practical enhancements like Grouped Query Attention, Alteration in KNN lookup, RoPE, and XL-style memory recurrenceand and an improved DataLoader.

This model is designed for scalable training, long-context understanding, and efficient memory usage.
---

## 🔬 Key Features

- ✅ **Grouped Query Attention**: Efficient query representation by grouping multiple attention heads for shared K/V access  
- ✅ **KNN-based Memory**: Long-term memory retrieval from past activations using a learned KNN mechanism  
- ✅ **XL-style Attention**: Recurrence-based memory layers adapted for KNN and grouped attention logic  
- ✅ **Rotary Positional Encoding**: More efficient and generalizable positional representation than vanilla sin-cos encoding  
- ✅ **Sharded Dataset Loader**: Handles large datasets with sharding and supports data parallelism via PyTorch DDP  
- ✅ **Custom Memory Clearing Logic**: Memory reset and lifespan mechanisms tuned for stability and performance during training  
- ✅ **Mixed Precision & DDP Training**: Efficient large-scale training using `torch.autocast` and `torchrun`  

---

Key Modifications from the Original Paper:
 1) Replaced the default positional encoding with Rotary Positional Embeddings (RoPE) , 
 2) Altered the attention mechanism to use Grouped Query Attention ,
 3) Customized the DataLoader to support sharded datasets and data parallelism ,
 4) Implemented Mixed Precision Training along with Distributed Data Parallel (DDP) support ,
 5) Tweaked several training and model hyperparameters for better adaptability .


## 📁 Project Structure

```bash
MEM_TRANSFORMER/
├── configs/
│   └── config.json                  # Model + training hyperparameters
│
├── data/
│   ├── edu_fineweb/                 # Token-sharded training data
│   │   ├── train_000001.npy
│   │   ├── train_000002.npy
│   │   └── test_000001.npy
│   ├── hellaswag/
│   │   └── hellaswag_val.jsonl
│   └── fineweb.py                   # Sharding logic with memory-aligned sequence control
│
├── model_core/
│   ├── __init__.py
│   ├── attention.py                 # Grouped Query Attention, KNN & XL attention logic.Rotary Positional Encoding implementation
│   ├── model.py                     # Transformer model with memory and RoPE support
│   ├── dataloader.py                # Memory-aware DataLoader
│   └── training.py                  # train_memgpt function
│
├── scripts/
│   ├── train.py                     # Training script (DDP-compatible)
│   ├── evaluate.py                  # Evaluation on benchmarks
│   └── generate.py                  # Text generation from trained model
│
├── evaluation/
│   ├── __init__.py
│   ├── hellaswag.py                 # HellaSwag data loader
│   └── val_hellaswag.py             # Evaluation logic with loss-based scoring
│
├── logs/
│   ├── log.txt                      # Training logs
│   └── model_*.pt                   # Checkpoints
│
├── .gitignore
├── README.md
├── requirements.txt

```
## ⚙️ Configuration 
Edit the config file at configs/config.json to adjust model and training hyperparameters:
```json
{
  "model": {
    "block_size": 1024,
    "vocab_size": 50304,
    "n_layer": 12,
    "n_head": 12,
    "n_embd": 768,
    "n_kv_head": 4,
    "max_knn_memories": 81920
  },
  "training": {
    "max_steps": 19073,
    "log_dir": "log",
    "total_batch_size": 2048,
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
```
```
## 🚀 Training

▶️ Single-GPU Training
```bash
python scripts/train.py

```
▶️ Distributed Training (Multi-GPU with DDP)
```bash
torchrun --nproc_per_node=NUM_GPUS scripts/train.py
```
Replace NUM_GPUS with the number of GPUs available.
```

##📊 Evaluation
Evaluate on the HellaSwag benchmark
```
📊 Evaluation
Evaluate on the HellaSwag benchmark:
python scripts/evaluate.py

Make sure the file data/hellaswag/hellaswag_val.jsonl is present.
The evaluation uses completion scoring based on masked loss comparisons across candidate endings.

🧠 Attention Mechanism Notes
🧩 Grouped Query Attention (GQA)
n_head query heads

n_kv_head shared key/value heads

Query heads are grouped and averaged before memory lookup

More efficient than per-head K/V for large models

🧩 KNN Memory Integration
A maximum memory buffer of 81920 tokens (max_knn_memories)

Query vectors are projected and grouped for efficient KNN search

Careful shape transformations ensure fast grouped matching

🧩 XL-style Attention + Memory Clearing
Recurrence with cached memory states

Implements custom memory clearing to avoid stale token influence

Helps stability in long training runs

💡 Positional Encoding
Rotary Positional Encoding (RoPE) replaces standard sin/cos

RoPE improves generalization over longer contexts

Implemented in model_core/rotary.py

🧩 Dataloader & Dataset Handling
Sharded training data using .npy files

Matching stride and memory alignment logic

Optimized for DDP compatibility and large-scale throughput

Code in model_core/dataloader.py and data/fineweb.py

📦 Requirements
Install dependencies:
```bash
pip install -r requirements.txt
```
Ensure PyTorch and CUDA versions match your GPU setup.




