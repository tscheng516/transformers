"""
Shard-based dataset utilities for offline, streaming-friendly training.

Shards are stored as uint16 .npy files plus an index.json that records
their paths and token counts.  The loader maps each file into memory so
only the pages that are actually accessed are loaded from disk.
"""

import json
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


# ── index helpers ──────────────────────────────────────────────────────────────

INDEX_FILENAME = "index.json"


def load_index(shard_dir: str) -> dict:
    """Load the JSON index from *shard_dir*."""
    index_path = os.path.join(shard_dir, INDEX_FILENAME)
    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"No index file found at {index_path}. "
            "Run tokenize_dataset.py first to create the shards."
        )
    with open(index_path) as f:
        return json.load(f)


def save_index(shard_dir: str, index: dict) -> None:
    """Write the JSON index to *shard_dir*."""
    os.makedirs(shard_dir, exist_ok=True)
    index_path = os.path.join(shard_dir, INDEX_FILENAME)
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)


# ── shard writers ──────────────────────────────────────────────────────────────

def write_shard(shard_path: str, token_ids: List[int]) -> None:
    """Save *token_ids* to a memory-mappable uint16 .npy file."""
    arr = np.array(token_ids, dtype=np.uint16)
    np.save(shard_path, arr)


def load_shard(shard_path: str) -> np.ndarray:
    """Memory-map a shard file, returning a uint16 array."""
    return np.load(shard_path, mmap_mode="r")


# ── PyTorch Dataset ────────────────────────────────────────────────────────────

class ShardedTokenDataset(Dataset):
    """
    A :class:`torch.utils.data.Dataset` that reads tokenized shards produced by
    ``tokenize_dataset.py``.

    Each call to ``__getitem__`` returns a 1-D :class:`torch.LongTensor` of
    length ``seq_len``.  Consecutive windows slide by ``stride`` tokens; if
    ``stride`` is ``None`` it defaults to ``seq_len`` (non-overlapping).

    Args:
        shard_dir: Directory that contains ``index.json`` and the ``.npy``
            shard files.
        seq_len: Context length for each training example.
        stride: Step between successive windows.  Defaults to ``seq_len``.
    """

    def __init__(
        self,
        shard_dir: str,
        seq_len: int = 2048,
        stride: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.stride = stride if stride is not None else seq_len

        index = load_index(shard_dir)
        self.shard_paths: List[str] = []
        self.shard_lengths: List[int] = []

        for entry in index["shards"]:
            path = entry["path"]
            if not os.path.isabs(path):
                path = os.path.join(shard_dir, path)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Shard file not found: {path}")
            self.shard_paths.append(path)
            self.shard_lengths.append(entry["num_tokens"])

        # Pre-compute per-shard window counts for fast index lookup.
        self._windows_per_shard: List[int] = []
        for length in self.shard_lengths:
            # Need seq_len + 1 tokens to build an (input, label) pair.
            # Max valid start: length - seq_len - 1; add 1 for the count.
            n_windows = max(0, (length - seq_len - 1) // self.stride + 1) if length >= seq_len + 1 else 0
            self._windows_per_shard.append(n_windows)

        self._cumulative = np.cumsum([0] + self._windows_per_shard)
        self._total = int(self._cumulative[-1])

        # Cache loaded shards to avoid redundant mmap calls.
        self._shard_cache: dict = {}

    def __len__(self) -> int:
        return self._total

    def _get_shard(self, shard_idx: int) -> np.ndarray:
        if shard_idx not in self._shard_cache:
            # Evict old entries to keep memory bounded (keep at most 2 shards).
            if len(self._shard_cache) >= 2:
                oldest = next(iter(self._shard_cache))
                del self._shard_cache[oldest]
            self._shard_cache[shard_idx] = load_shard(self.shard_paths[shard_idx])
        return self._shard_cache[shard_idx]

    def __getitem__(self, idx: int) -> dict:
        if idx < 0 or idx >= self._total:
            raise IndexError(f"Index {idx} out of range [0, {self._total})")

        # Binary-search for the shard that owns this window.
        shard_idx = int(np.searchsorted(self._cumulative[1:], idx, side="right"))
        window_in_shard = idx - int(self._cumulative[shard_idx])

        start = window_in_shard * self.stride
        end = start + self.seq_len + 1  # +1 for the label

        shard = self._get_shard(shard_idx)
        chunk = shard[start:end].astype(np.int64)

        input_ids = torch.from_numpy(chunk[:-1].copy())
        labels = torch.from_numpy(chunk[1:].copy())

        return {"input_ids": input_ids, "labels": labels}


# ── convenience factory ────────────────────────────────────────────────────────

def make_shard_dataset(
    shard_dir: str,
    seq_len: int = 2048,
    stride: Optional[int] = None,
) -> ShardedTokenDataset:
    """Factory wrapper for :class:`ShardedTokenDataset`."""
    return ShardedTokenDataset(shard_dir=shard_dir, seq_len=seq_len, stride=stride)


def print_dataset_stats(dataset: ShardedTokenDataset) -> None:
    """Print a short summary of a :class:`ShardedTokenDataset`."""
    total_tokens = sum(dataset.shard_lengths)
    print(f"Shards     : {len(dataset.shard_paths)}")
    print(f"Tokens     : {total_tokens:,}")
    print(f"Seq length : {dataset.seq_len}")
    print(f"Windows    : {len(dataset):,}")
