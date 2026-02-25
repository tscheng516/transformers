#!/usr/bin/env python3
"""
tokenize_dataset.py — offline tokenization into memory-mappable shards.

Reads raw text from a HuggingFace dataset (e.g. wikitext-2-raw-v1) or from
local text / jsonl files and writes uint16 .npy shards + an index.json so
that training can proceed without any network access.

Quick-start examples
--------------------
# HuggingFace dataset
python tokenize_dataset.py \\
    --tokenizer_name_or_path allenai/OLMo-2-1124-7B \\
    --hf_dataset wikitext \\
    --hf_config wikitext-2-raw-v1 \\
    --output_dir ./data/wikitext2_tokenized

# Local text file(s)
python tokenize_dataset.py \\
    --tokenizer_name_or_path allenai/OLMo-2-1124-7B \\
    --local_files ./data/train.txt ./data/val.txt \\
    --output_dir ./data/custom_tokenized

# Resume (existing shards are skipped automatically)
python tokenize_dataset.py ... --output_dir ./data/wikitext2_tokenized
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Iterator, List, Optional

import numpy as np
from transformers import AutoTokenizer

# Allow running as a top-level script from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.olmo3_custom.data import INDEX_FILENAME, save_index, write_shard  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ── text sources ───────────────────────────────────────────────────────────────

def iter_hf_dataset(dataset_name: str, config_name: Optional[str], split: str, text_column: str) -> Iterator[str]:
    """Yield raw text strings from a HuggingFace dataset."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Install the `datasets` library: pip install datasets")

    logger.info("Loading HF dataset %s / %s (split=%s) ...", dataset_name, config_name, split)
    ds = load_dataset(dataset_name, config_name, split=split, trust_remote_code=False)
    for example in ds:
        text = example.get(text_column, "")
        if text and text.strip():
            yield text


def iter_local_files(paths: List[str], text_column: Optional[str] = "text") -> Iterator[str]:
    """Yield raw text strings from local .txt or .jsonl / .json files."""
    for path in paths:
        ext = Path(path).suffix.lower()
        if ext == ".txt":
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        yield line
        elif ext in (".jsonl", ".json"):
            import json as _json
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    obj = _json.loads(line)
                    text = obj.get(text_column, "")
                    if text and text.strip():
                        yield text
        else:
            logger.warning("Skipping unsupported file type: %s", path)


# ── tokenisation + shard writing ──────────────────────────────────────────────

def tokenize_and_shard(
    texts: Iterator[str],
    tokenizer,
    output_dir: str,
    tokens_per_shard: int,
    existing_shards: List[dict],
) -> List[dict]:
    """
    Consume *texts*, tokenise them, and write shards of size *tokens_per_shard*.

    Already-existing shards (from a previous run) are passed in as
    *existing_shards* and are not overwritten; the function picks up where the
    previous run left off.

    Returns the full list of shard metadata dicts (old + new).
    """
    os.makedirs(output_dir, exist_ok=True)

    shards = list(existing_shards)   # copy so we can append
    shard_idx = len(shards)          # resume: start numbering from here
    buffer: List[int] = []

    for text in texts:
        ids = tokenizer.encode(text)
        buffer.extend(ids)

        while len(buffer) >= tokens_per_shard:
            chunk = buffer[:tokens_per_shard]
            buffer = buffer[tokens_per_shard:]

            shard_name = f"shard_{shard_idx:06d}.npy"
            shard_path = os.path.join(output_dir, shard_name)
            write_shard(shard_path, chunk)

            shards.append({"path": shard_name, "num_tokens": len(chunk)})
            logger.info("Wrote shard %d  →  %s  (%d tokens)", shard_idx, shard_path, len(chunk))
            shard_idx += 1

    # Write leftover tokens (possibly smaller than tokens_per_shard).
    if buffer:
        shard_name = f"shard_{shard_idx:06d}.npy"
        shard_path = os.path.join(output_dir, shard_name)
        write_shard(shard_path, buffer)
        shards.append({"path": shard_name, "num_tokens": len(buffer)})
        logger.info(
            "Wrote final shard %d  →  %s  (%d tokens)", shard_idx, shard_path, len(buffer)
        )

    return shards


# ── main ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tokenize a dataset into local .npy shards for offline training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Tokenizer
    parser.add_argument(
        "--tokenizer_name_or_path",
        required=True,
        help="HuggingFace tokenizer name or local path.",
    )
    parser.add_argument(
        "--tokenizer_use_fast",
        action="store_true",
        default=True,
        help="Use the fast tokenizer implementation.",
    )

    # Data source: HF dataset or local files (mutually exclusive).
    src_group = parser.add_mutually_exclusive_group(required=True)
    src_group.add_argument("--hf_dataset", help="HuggingFace dataset name, e.g. 'wikitext'.")
    src_group.add_argument(
        "--local_files",
        nargs="+",
        help="One or more local .txt / .jsonl / .json files.",
    )

    # HF dataset options (only used when --hf_dataset is set).
    parser.add_argument("--hf_config", default=None, help="HF dataset config, e.g. 'wikitext-2-raw-v1'.")
    parser.add_argument("--hf_split", default="train", help="Dataset split to tokenize.")
    parser.add_argument(
        "--text_column",
        default="text",
        help="Column/key in the dataset that contains raw text.",
    )

    # Sharding
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory where shards and index.json will be written.",
    )
    parser.add_argument(
        "--tokens_per_shard",
        type=int,
        default=1_000_000,
        help="Approximate number of tokens per shard file.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── tokenizer ──────────────────────────────────────────────────────────────
    logger.info("Loading tokenizer: %s", args.tokenizer_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name_or_path,
        use_fast=args.tokenizer_use_fast,
    )

    # ── resumability: load existing index (if any) ─────────────────────────────
    index_path = os.path.join(args.output_dir, INDEX_FILENAME)
    existing_shards: List[dict] = []
    if os.path.exists(index_path):
        with open(index_path) as fh:
            existing_index = json.load(fh)
        existing_shards = existing_index.get("shards", [])
        logger.info(
            "Resuming: found %d existing shards (%s tokens already tokenized).",
            len(existing_shards),
            sum(s["num_tokens"] for s in existing_shards),
        )
    else:
        logger.info("No existing index found; starting from scratch.")

    # ── text source ────────────────────────────────────────────────────────────
    if args.hf_dataset:
        texts = iter_hf_dataset(args.hf_dataset, args.hf_config, args.hf_split, args.text_column)
    else:
        texts = iter_local_files(args.local_files, args.text_column)

    # If resuming, skip the number of tokens already written.
    # Simple approach: skip the first N tokens from the stream.
    # (More sophisticated resume would require reproducible ordering.)
    tokens_already_written = sum(s["num_tokens"] for s in existing_shards)
    if tokens_already_written > 0:
        logger.info(
            "Skipping first %d tokens (already in existing shards).", tokens_already_written
        )

        def _skip_tokens(text_iter: Iterator[str], skip: int) -> Iterator[str]:
            skipped = 0
            for text in text_iter:
                ids = tokenizer.encode(text)
                if skipped + len(ids) <= skip:
                    skipped += len(ids)
                    continue
                # Partially skip this document.
                remaining_skip = skip - skipped
                skipped = skip
                # Re-decode the remaining part (crude but correct).
                partial_ids = ids[remaining_skip:]
                if partial_ids:
                    yield tokenizer.decode(partial_ids, skip_special_tokens=False)
                # From now on yield everything.
                yield from text_iter
                return

        texts = _skip_tokens(texts, tokens_already_written)

    # ── tokenise + shard ───────────────────────────────────────────────────────
    shards = tokenize_and_shard(
        texts=texts,
        tokenizer=tokenizer,
        output_dir=args.output_dir,
        tokens_per_shard=args.tokens_per_shard,
        existing_shards=existing_shards,
    )

    # ── write index ────────────────────────────────────────────────────────────
    index = {
        "tokenizer": args.tokenizer_name_or_path,
        "tokens_per_shard": args.tokens_per_shard,
        "total_tokens": sum(s["num_tokens"] for s in shards),
        "shards": shards,
    }
    save_index(args.output_dir, index)

    logger.info(
        "Done. %d shards, %d total tokens. Index: %s",
        len(shards),
        index["total_tokens"],
        index_path,
    )


if __name__ == "__main__":
    main()
