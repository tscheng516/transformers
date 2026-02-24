# OLMo3-Custom Training Guide

End-to-end guide for training a small (~300 M parameter) **OLMo3-Custom**
causal language model on a **remote single-node multi-GPU** server using the
Hugging Face `Trainer`.  All steps are designed to be **offline-friendly**:
once the environment and data are set up you can train without any network
access.

---

## Table of Contents

1. [Environment Setup](#1-environment-setup)
2. [Installing Dependencies](#2-installing-dependencies)
3. [Weights & Biases Setup](#3-weights--biases-setup)
4. [Storage Layout](#4-storage-layout)
5. [Tokenizing the Dataset](#5-tokenizing-the-dataset)
6. [Training with `torchrun`](#6-training-with-torchrun)
7. [Resuming a Run](#7-resuming-a-run)
8. [muP Initialisation (optional)](#8-mup-initialisation-optional)
9. [Model Sizing Utilities](#9-model-sizing-utilities)
10. [Large Dataset Caveats](#10-large-dataset-caveats)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. Environment Setup

### Conda (recommended)

```bash
conda create -n olmo3 python=3.10 -y
conda activate olmo3
```

### venv

```bash
python3 -m venv ~/envs/olmo3
source ~/envs/olmo3/bin/activate
```

---

## 2. Installing Dependencies

```bash
# Clone the repo (if not already done)
git clone https://github.com/tscheng516/transformers.git
cd transformers

# Install the library in editable mode
pip install -e ".[dev]"

# Core training dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Dataset + tokenization
pip install datasets tokenizers

# Experiment tracking
pip install wandb

# (Optional) muP
pip install mup
```

> **Offline note**  After installing everything once, use
> `pip download -d ./pip_cache -r requirements.txt` to cache wheels for
> future installs without internet access.

---

## 3. Weights & Biases Setup

```bash
# Log in once (stores an API key in ~/.netrc)
wandb login

# Or set the API key as an environment variable (no interactive prompt)
export WANDB_API_KEY=your_api_key_here

# Name the project (optional; defaults to "olmo3_custom")
export WANDB_PROJECT=olmo3_custom
```

To disable W&B logging entirely, pass `--disable_wandb` to the training
script or set:

```bash
export WANDB_DISABLED=true
```

---

## 4. Storage Layout

We recommend the following directory structure:

```
~/experiments/
├── data/
│   ├── raw/                   # raw text files (optional local data)
│   └── wikitext2_tokenized/   # tokenised shards + index.json
│       ├── index.json
│       ├── shard_000000.npy
│       └── shard_000001.npy
├── checkpoints/
│   └── olmo3_custom_300m/     # Trainer checkpoints + final model
└── transformers/              # this repo
```

---

## 5. Tokenizing the Dataset

The `tokenize_dataset.py` script reads raw text, tokenises it, and writes
**uint16 `.npy` shards** plus an `index.json` to disk.  Training can then
proceed without any network access.

### Starter dataset (WikiText-2)

```bash
python scripts/olmo3_custom/tokenize_dataset.py \
    --tokenizer_name_or_path allenai/OLMo-2-1124-7B \
    --hf_dataset wikitext \
    --hf_config wikitext-2-raw-v1 \
    --hf_split train \
    --output_dir ~/experiments/data/wikitext2_tokenized \
    --tokens_per_shard 1000000
```

### Local text / JSONL files

```bash
python scripts/olmo3_custom/tokenize_dataset.py \
    --tokenizer_name_or_path allenai/OLMo-2-1124-7B \
    --local_files ~/data/train.txt ~/data/extra.jsonl \
    --output_dir ~/experiments/data/custom_tokenized
```

### Resumability

If the script is interrupted, re-run the identical command.  It reads
`index.json` and skips shards that already exist, picking up where it left
off.

---

## 6. Training with `torchrun`

### Single GPU

```bash
python scripts/olmo3_custom/train_olmo3_custom.py \
    --tokenizer_name allenai/OLMo-2-1124-7B \
    --shard_dir ~/experiments/data/wikitext2_tokenized \
    --output_dir ~/experiments/checkpoints/olmo3_custom_300m \
    --num_train_epochs 3 \
    --fp16
```

### Multi-GPU (DDP via `torchrun`)

```bash
torchrun \
    --nproc_per_node=4 \
    --master_port=29500 \
    scripts/olmo3_custom/train_olmo3_custom.py \
    --tokenizer_name allenai/OLMo-2-1124-7B \
    --shard_dir ~/experiments/data/wikitext2_tokenized \
    --output_dir ~/experiments/checkpoints/olmo3_custom_300m \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 3 \
    --fp16 \
    --wandb_project olmo3_custom \
    --wandb_run_name olmo3_300m_wikitext2
```

### Key training arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--hidden_size` | 1024 | Model hidden dimension |
| `--num_hidden_layers` | 16 | Number of transformer layers |
| `--num_attention_heads` | 8 | Attention heads |
| `--num_key_value_heads` | 4 | KV heads (GQA) |
| `--intermediate_size` | 4096 | MLP intermediate dimension |
| `--seq_len` | 2048 | Training context length |
| `--learning_rate` | 3e-4 | Peak learning rate |
| `--warmup_ratio` | 0.01 | Fraction of steps used for warmup |
| `--lr_scheduler_type` | cosine | Scheduler: cosine, linear, constant |
| `--max_grad_norm` | 1.0 | Gradient clipping threshold |
| `--fp16` | off | Enable fp16 mixed precision |
| `--save_steps` | 500 | Checkpoint save interval (steps) |
| `--logging_steps` | 10 | Logging interval (steps) |

### What gets logged to W&B

* `train/loss` — per-step training loss
* `train/grad_norm` — L2 gradient norm (logged by `GradNormCallback`)
* `train/learning_rate` — current LR
* `eval/loss` — evaluation loss (if a validation set is available)
* Default Trainer metrics (epoch, global step, etc.)

---

## 7. Resuming a Run

Simply re-run the **same command**.  The Trainer detects the latest
checkpoint in `--output_dir` automatically.

To resume from a specific checkpoint:

```bash
torchrun --nproc_per_node=4 scripts/olmo3_custom/train_olmo3_custom.py \
    ... \
    --resume_from_checkpoint ~/experiments/checkpoints/olmo3_custom_300m/checkpoint-5000
```

Or use `--resume_from_checkpoint latest` to auto-detect:

```bash
    --resume_from_checkpoint latest
```

---

## 8. muP Initialisation (optional)

[muP](https://github.com/microsoft/mup) (Maximal Update Parametrisation)
scales weights to ensure that activations and gradients are independent of
model width, enabling hyperparameter transfer from small proxy models.

### What is implemented

The `scripts/olmo3_custom/mup_init.py` module provides a **pragmatic**
integration:

* All `nn.Linear` weights are rescaled by
  `sqrt(base_hidden_size / hidden_size)` to approximate muP fan-in scaling.
* Width-dependent projection layers get a lower learning rate
  (`base_lr * base_hidden_size / hidden_size`) via separate param groups.

### What is NOT implemented

* True muP requires a "coord-check" to verify that the training signal is
  independent of width.  This is not done automatically.
* Rigorous hyperparameter transfer from a small proxy model is not set up.
  To do this properly you need to train multiple proxy models at different
  widths and verify that the optimal LR is the same.

### Usage

```bash
torchrun --nproc_per_node=4 scripts/olmo3_custom/train_olmo3_custom.py \
    ... \
    --use_mup \
    --mup_base_hidden_size 256
```

If the `mup` package is not installed, the script will print a warning and
fall back to standard initialisation.  To make it fail with a clear error
instead, add `--mup_strict`.

---

## 9. Model Sizing Utilities

The training script includes helpers to estimate parameter counts:

```python
from scripts.olmo3_custom.train_olmo3_custom import (
    make_olmo3_custom_config_300m,
    estimate_param_count,
    print_model_size,
)

# Default ~300M config
config = make_olmo3_custom_config_300m()
print_model_size(config)
# → Estimated parameters: 302,xxxxxx  (302.x M  /  0.30 B)

# Larger model (~1B)
big_config = make_olmo3_custom_config_300m(
    hidden_size=2048,
    intermediate_size=8192,
    num_hidden_layers=24,
    num_attention_heads=16,
    num_key_value_heads=8,
)
print_model_size(big_config)
```

---

## 10. Large Dataset Caveats

WikiText-2 is tiny (~2 M tokens) and is intended only as a **smoke-test**.
For real pre-training you will need billions of tokens.

| Scale | Approximate dataset size | Notes |
|-------|--------------------------|-------|
| Smoke-test | WikiText-2 (~2 M tokens) | Fits in memory |
| Small run | 10–100 B tokens | Several hundred GB of shards |
| Full pre-training | 1–3 T tokens | Multiple TB of storage |

**Tips for large datasets:**

1. **Shard size**: Increase `--tokens_per_shard` (e.g. `100_000_000`) to
   reduce the number of files.
2. **Storage**: Store shards on fast NVMe or a parallel file system; avoid
   network-mounted drives when possible.
3. **Workers**: Increase `--dataloader_num_workers` (e.g. 8–16) so the CPU
   can pre-fetch shards while the GPU trains.
4. **Sequence packing**: For dense token utilisation consider implementing
   sequence packing (context length = multiple documents concatenated).
5. **Validation split**: Create a separate shard directory for validation
   (e.g. `val_tokenized/`) by running `tokenize_dataset.py` on a held-out
   split.  The training script automatically looks for a `val_tokenized/`
   sibling of `--shard_dir`.

---

## 11. Troubleshooting

### `CUDA out of memory`

* Reduce `--per_device_train_batch_size` and increase
  `--gradient_accumulation_steps` to keep the effective batch size the same.
* Enable `--fp16` (already recommended).
* Reduce `--seq_len`.

### `NCCL timeout` or DDP hangs

* Ensure all GPUs are visible: `CUDA_VISIBLE_DEVICES=0,1,2,3`.
* Set `NCCL_TIMEOUT` if your initialisation is slow:
  `export NCCL_TIMEOUT=600`.

### W&B not logging

* Check `wandb login` was run and the API key is valid.
* Set `WANDB_MODE=offline` to log locally and sync later with
  `wandb sync`.

### `FileNotFoundError: No index file found`

You haven't tokenised the dataset yet.  Run `tokenize_dataset.py` first (see
[§5](#5-tokenizing-the-dataset)).

### Resuming crashes with shape mismatch

Make sure the model arguments (hidden size, layers, etc.) are identical to
the run that produced the checkpoint.
