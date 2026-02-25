#!/usr/bin/env python3
"""
train_olmo3_custom.py — End-to-end training script for a small OLMo3-custom model.

Features
--------
* Trains a causal LM from *scratch* using ``transformers.Trainer``.
* Initialises a small (~300M parameter) ``Olmo3Custom`` model or any
  custom size specified on the command line.
* Logs **training loss**, **gradient norm**, and all default Trainer
  metrics to **Weights & Biases**.
* Supports **DDP** via ``torchrun`` (single-node multi-GPU).
* Supports **fp16** (bf16 is disabled by default as per user requirement).
* Provides gradient clipping, warmup + cosine/linear scheduler, and
  regular checkpoint saves so experiments can be safely resumed.
* Optional **muP** initialisation via ``--use_mup``.
* Loads data from shards produced by ``tokenize_dataset.py`` **or** falls
  back to the wikitext-2-raw-v1 HuggingFace dataset for quick testing.

Example — single GPU
---------------------
python train_olmo3_custom.py \\
    --tokenizer_name allenai/OLMo-2-1124-7B \\
    --shard_dir ./data/wikitext2_tokenized \\
    --output_dir ./checkpoints/olmo3_custom_300m \\
    --num_train_epochs 3

Example — multi-GPU with torchrun
-----------------------------------
torchrun --nproc_per_node=4 train_olmo3_custom.py \\
    --tokenizer_name allenai/OLMo-2-1124-7B \\
    --shard_dir ./data/wikitext2_tokenized \\
    --output_dir ./checkpoints/olmo3_custom_300m \\
    --fp16

Resuming
--------
Re-run the exact same command; Trainer detects the latest checkpoint
automatically.
"""

import argparse
import logging
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset

import transformers
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
    set_seed,
)

# Allow running as a top-level script from the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
from scripts.olmo3_custom.data import ShardedTokenDataset  # noqa: E402
from scripts.olmo3_custom.mup_init import apply_mup_init  # noqa: E402

# Model imports (works whether the repo is installed in editable mode or not).
try:
    from transformers.models.olmo3_custom.configuration_olmo3_custom import Olmo3CustomConfig
    from transformers.models.olmo3_custom.modeling_olmo3_custom import Olmo3CustomForCausalLM
except ImportError:
    # Fall back to direct source imports when the package is not installed.
    sys.path.insert(0, str(_REPO_ROOT / "src"))
    from transformers.models.olmo3_custom.configuration_olmo3_custom import Olmo3CustomConfig  # noqa: E402
    from transformers.models.olmo3_custom.modeling_olmo3_custom import Olmo3CustomForCausalLM  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)


# ── model sizing utilities ─────────────────────────────────────────────────────

def make_olmo3_custom_config_300m(
    vocab_size: int = 50304,
    hidden_size: int = 1024,
    intermediate_size: int = 4096,
    num_hidden_layers: int = 16,
    num_attention_heads: int = 8,
    num_key_value_heads: int = 4,
    max_position_embeddings: int = 2048,
    norm_type: str = "rmsnorm",
    norm_pos: str = "pre",
) -> Olmo3CustomConfig:
    """
    Return a config that yields approximately 300 M parameters.

    Default shape:  hidden=1024, layers=16, heads=8, kv_heads=4,
                    intermediate=4096  →  ~300 M params.

    Adjust *hidden_size* / *num_hidden_layers* / *intermediate_size* to
    change the model size.  Call :func:`estimate_param_count` to verify.
    """
    # Accept additional config overrides via kwargs and keep sensible defaults
    return Olmo3CustomConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        max_position_embeddings=max_position_embeddings,
        norm_type=norm_type,
        norm_pos=norm_pos,
        # Stable defaults for training (can be overridden via kwargs)
        rms_norm_eps=1e-5,
        initializer_range=0.02,
        attention_dropout=0.0,
        use_cache=False,          # disable KV-cache during training
        sliding_window=2048,
    )


def estimate_param_count(config: Olmo3CustomConfig) -> int:
    """
    Return an approximate parameter count for a given config without
    allocating the model.
    """
    H = config.hidden_size
    I = config.intermediate_size
    V = config.vocab_size
    L = config.num_hidden_layers
    Nq = config.num_attention_heads
    Nkv = config.num_key_value_heads
    head_dim = H // Nq

    # Embedding table
    embedding = V * H

    # Per-layer params
    # Attention: q, k, v, o projections
    attn = Nq * head_dim * H + 2 * Nkv * head_dim * H + H * H
    # MLP: gate + up + down
    mlp = 2 * H * I + I * H
    # Norms (negligible but we include them)
    norms = 4 * H  # two norms per block (pre + post)

    per_layer = attn + mlp + norms
    total = embedding + L * per_layer
    return total


def print_model_size(config: Olmo3CustomConfig) -> None:
    """Pretty-print estimated parameter count for *config*."""
    params = estimate_param_count(config)
    print(
        f"Estimated parameters: {params:,}  "
        f"({params / 1e6:.1f} M  /  {params / 1e9:.2f} B)"
    )


# ── gradient-norm callback ────────────────────────────────────────────────────

class GradNormCallback(TrainerCallback):
    """Log the gradient L2-norm to W&B after each optimiser step."""

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs,
    ) -> None:
        if model is None:
            return
        total_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                total_norm += param.grad.detach().float().norm(2).item() ** 2
        total_norm = total_norm ** 0.5

        # Log to W&B (Trainer's default integration handles the call).
        if state.is_world_process_zero:
            try:
                import wandb
                if wandb.run is not None:
                    wandb.log({"train/grad_norm": total_norm}, step=state.global_step)
            except ImportError:
                pass


# ── dataset fallback (WikiText-2 from HF) ────────────────────────────────────

class HFDatasetWrapper(Dataset):
    """Wrap a tokenised HuggingFace dataset as a seq-len Dataset."""

    def __init__(self, encodings, seq_len: int = 2048):
        self.input_ids = encodings
        self.seq_len = seq_len
        n = len(self.input_ids) - 1
        self._len = max(0, n // seq_len)

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len
        input_ids = torch.tensor(self.input_ids[start:end], dtype=torch.long)
        labels = torch.tensor(self.input_ids[start + 1 : end + 1], dtype=torch.long)
        return {"input_ids": input_ids, "labels": labels}


def load_hf_fallback_dataset(tokenizer_name: str, seq_len: int = 2048):
    """Load wikitext-2-raw-v1 from HF as a fallback when no shard dir is given."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Install the `datasets` library: pip install datasets")

    logger.info("Fallback: loading wikitext-2-raw-v1 from HuggingFace datasets ...")
    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    raw = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(raw["text"])
    ids = tok.encode(text)
    train_ids = ids[: int(0.9 * len(ids))]
    val_ids = ids[int(0.9 * len(ids)) :]
    return HFDatasetWrapper(train_ids, seq_len=seq_len), HFDatasetWrapper(val_ids, seq_len=seq_len)


# ── argument parsing ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train an OLMo3-custom model from scratch.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- tokenizer / data ---
    p.add_argument("--tokenizer_name", default="allenai/OLMo-2-1124-7B",
                   help="HF tokenizer name or local path.")
    p.add_argument("--shard_dir", default=None,
                   help="Directory of tokenised shards (from tokenize_dataset.py). "
                        "If not provided, falls back to wikitext-2 from HF.")
    p.add_argument("--seq_len", type=int, default=2048,
                   help="Training context length (tokens).")
    p.add_argument("--stride", type=int, default=None,
                   help="Sliding-window stride for shard dataset. Defaults to seq_len.")

    # --- model shape ---
    p.add_argument("--hidden_size", type=int, default=1024)
    p.add_argument("--intermediate_size", type=int, default=4096)
    p.add_argument("--num_hidden_layers", type=int, default=16)
    p.add_argument("--num_attention_heads", type=int, default=8)
    p.add_argument("--num_key_value_heads", type=int, default=4)
    p.add_argument("--max_position_embeddings", type=int, default=2048)
    p.add_argument("--norm_type", default="rmsnorm",
                   choices=["rmsnorm", "dyt", "derf"])
    p.add_argument("--norm_pos", default="pre",
                   choices=["pre", "post", "mid", "sandwich", "hybrid"])

    # Additional config flags added to Olmo3CustomConfig
    p.add_argument("--vocab_size", type=int, default=50304,
                   help="Model vocabulary size.")
    p.add_argument("--hidden_act", default="silu",
                   help="Activation in hidden MLPs.")
    p.add_argument("--initializer_range", type=float, default=0.02,
                   help="Initializer std dev for weight matrices.")
    p.add_argument("--tie_word_embeddings", action="store_true",
                   help="Tie input and output embeddings in the model.")
    p.add_argument("--attention_bias", action="store_true",
                   help="Use biases in attention projections.")
    p.add_argument("--attention_dropout", type=float, default=0.0,
                   help="Dropout for attention probabilities.")
    p.add_argument("--rms_norm_eps", type=float, default=1e-5,
                   help="Epsilon for RMSNorm layers.")
    p.add_argument("--sliding_window", type=int, default=4096,
                   help="Sliding window size for sliding-window attention.")
    p.add_argument("--layer_types", default=None,
                   help="Comma-separated list of layer types (e.g. 'sliding_attention,full_attention').")
    p.add_argument("--intra_norm_pos", default="qk",
                   choices=["qk", "qkv", "qkvc", "qkc", "c"],
                   help="Which internal matrices to normalize in attention.")
    p.add_argument("--ffn_norm_pos", default="pre",
                   choices=["pre", "post", "mid", "sandwich", "none"],
                   help="Normalization position for FFN block.")
    p.add_argument("--alpha_init_value", type=float, default=1.0,
                   help="Initial alpha for DyT/Derf norms.")
    p.add_argument("--shift_init_value", type=float, default=0.0,
                   help="Initial shift for DyT/Derf norms.")
    p.add_argument("--use_gated_attention", action="store_true",
                   help="Enable gated attention mechanism.")
    p.add_argument("--attn_act", default="swish",
                   help="Activation for gated attention.")
    p.add_argument("--rope_theta", type=float, default=None,
                   help="If set, creates rope_parameters={'rope_theta': value} for RoPE config.")

    # --- training ---
    p.add_argument("--output_dir", required=True)
    p.add_argument("--num_train_epochs", type=int, default=3)
    p.add_argument("--per_device_train_batch_size", type=int, default=4)
    p.add_argument("--per_device_eval_batch_size", type=int, default=8)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--max_grad_norm", type=float, default=1.0,
                   help="Gradient clipping max norm.")
    p.add_argument("--warmup_ratio", type=float, default=0.01,
                   help="Fraction of total steps used for warmup.")
    p.add_argument("--lr_scheduler_type", default="cosine",
                   choices=["cosine", "linear", "constant", "cosine_with_restarts"],
                   help="Learning rate scheduler type.")
    p.add_argument("--fp16", action="store_true",
                   help="Enable fp16 mixed-precision training.")
    p.add_argument("--bf16", action="store_true",
                   help="Enable bf16 mixed-precision training (not recommended per user preference).")
    p.add_argument("--save_steps", type=int, default=500,
                   help="Save a checkpoint every N steps.")
    p.add_argument("--eval_steps", type=int, default=500,
                   help="Evaluate every N steps.")
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dataloader_num_workers", type=int, default=4)

    # --- W&B ---
    p.add_argument("--wandb_project", default="olmo3_custom",
                   help="W&B project name.")
    p.add_argument("--wandb_run_name", default=None,
                   help="W&B run name (optional).")
    p.add_argument("--disable_wandb", action="store_true",
                   help="Disable W&B logging entirely.")

    # --- muP ---
    p.add_argument("--use_mup", action="store_true",
                   help="Apply muP (Maximal Update Parametrisation) initialisation.")
    p.add_argument("--mup_base_hidden_size", type=int, default=256,
                   help="Base hidden size used for muP scaling (proxy model width).")
    p.add_argument("--mup_strict", action="store_true",
                   help="Raise an error if the `mup` package is not installed.")

    # --- misc ---
    p.add_argument("--resume_from_checkpoint", default=None,
                   help="Path to a checkpoint directory to resume from. "
                        "Pass 'latest' to auto-detect the latest checkpoint.")

    return p.parse_args()


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    # ── W&B setup ──────────────────────────────────────────────────────────────
    if args.disable_wandb:
        os.environ["WANDB_DISABLED"] = "true"
    else:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
        if args.wandb_run_name:
            os.environ["WANDB_NAME"] = args.wandb_run_name

    # ── model config ──────────────────────────────────────────────────────────
    config = make_olmo3_custom_config_300m(
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads,
        max_position_embeddings=args.max_position_embeddings,
        norm_type=args.norm_type,
        norm_pos=args.norm_pos,
    )

    # Parse optional complex fields from CLI and attach to config
    # layer_types: optional comma-separated string -> list[str]
    layer_types = None
    if args.layer_types:
        layer_types = [s.strip() for s in args.layer_types.split(",") if s.strip()]

    rope_parameters = None
    if args.rope_theta is not None:
        rope_parameters = {"rope_theta": args.rope_theta}

    # Override/attach additional fields on the config object so CLI controls them.
    # These attributes mirror the Olmo3CustomConfig constructor arguments.
    config.vocab_size = args.vocab_size
    config.hidden_act = args.hidden_act
    config.initializer_range = args.initializer_range
    config.tie_word_embeddings = bool(args.tie_word_embeddings)
    config.attention_bias = bool(args.attention_bias)
    config.attention_dropout = float(args.attention_dropout)
    config.rms_norm_eps = float(args.rms_norm_eps)
    config.sliding_window = int(args.sliding_window)
    if layer_types is not None:
        config.layer_types = layer_types
    config.intra_norm_pos = args.intra_norm_pos
    config.ffn_norm_pos = args.ffn_norm_pos
    config.alpha_init_value = float(args.alpha_init_value)
    config.shift_init_value = float(args.shift_init_value)
    config.use_gated_attention = bool(args.use_gated_attention)
    config.attn_act = args.attn_act
    config.rope_parameters = rope_parameters

    logger.info("Model config:\n%s", config.to_json_string())
    print_model_size(config)

    # ── model ─────────────────────────────────────────────────────────────────
    logger.info("Initialising model from scratch ...")
    model = Olmo3CustomForCausalLM(config)

    # Save the initial config alongside the checkpoints.
    os.makedirs(args.output_dir, exist_ok=True)
    config.save_pretrained(args.output_dir)

    if args.use_mup:
        apply_mup_init(
            model,
            base_hidden_size=args.mup_base_hidden_size,
            strict=args.mup_strict,
        )

    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Total model parameters: %d  (%.1f M)", total_params, total_params / 1e6)

    # ── tokenizer ─────────────────────────────────────────────────────────────
    logger.info("Loading tokenizer: %s", args.tokenizer_name)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ── datasets ──────────────────────────────────────────────────────────────
    if args.shard_dir:
        logger.info("Loading shard dataset from: %s", args.shard_dir)
        train_dataset = ShardedTokenDataset(
            shard_dir=args.shard_dir,
            seq_len=args.seq_len,
            stride=args.stride,
        )
        # If a separate validation shard dir exists alongside, use it.
        val_shard_dir = os.path.join(os.path.dirname(args.shard_dir), "val_tokenized")
        if os.path.exists(val_shard_dir):
            eval_dataset = ShardedTokenDataset(
                shard_dir=val_shard_dir,
                seq_len=args.seq_len,
                stride=args.stride,
            )
            logger.info("Using validation shards from: %s", val_shard_dir)
        else:
            eval_dataset = None
            logger.info("No validation shard dir found at %s; skipping eval.", val_shard_dir)
    else:
        logger.info("No --shard_dir given; falling back to wikitext-2-raw-v1 from HF.")
        train_dataset, eval_dataset = load_hf_fallback_dataset(
            args.tokenizer_name, seq_len=args.seq_len
        )

    logger.info("Train samples: %d", len(train_dataset))
    if eval_dataset is not None:
        logger.info("Eval  samples: %d", len(eval_dataset))

    # ── training arguments ────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        fp16=args.fp16,
        bf16=args.bf16,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=args.eval_steps if eval_dataset is not None else None,
        save_total_limit=3,
        load_best_model_at_end=eval_dataset is not None,
        report_to=[] if args.disable_wandb else ["wandb"],
        dataloader_num_workers=args.dataloader_num_workers,
        seed=args.seed,
        remove_unused_columns=False,   # our dataset already returns the right keys
        ddp_find_unused_parameters=False,
    )

    # ── trainer ───────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        callbacks=[GradNormCallback()],
    )

    # ── train ─────────────────────────────────────────────────────────────────
    resume = args.resume_from_checkpoint
    if resume == "latest":
        # Auto-detect the latest checkpoint in output_dir.
        from transformers.trainer_utils import get_last_checkpoint
        resume = get_last_checkpoint(args.output_dir)
        if resume:
            logger.info("Resuming from checkpoint: %s", resume)
        else:
            logger.info("No checkpoint found in %s; starting from scratch.", args.output_dir)
            resume = None

    trainer.train(resume_from_checkpoint=resume)

    # ── save final model ──────────────────────────────────────────────────────
    logger.info("Saving final model to %s", args.output_dir)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
