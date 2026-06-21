"""
precompute_bm25.py — Build BM25 sparse index for hybrid retrieval.

Requires candidate_meta.pkl (from precompute.py).
Outputs: bm25_index (bm25s directory format)

Tokenizer must stay in sync with ranking_pipeline.tokenize_bm25().
"""

import pickle
import shutil
from pathlib import Path

import bm25s
from tqdm import tqdm

from ranking_pipeline import tokenize_bm25

BASE = Path(__file__).parent


def main():
    meta_path = BASE / "candidate_meta.pkl"
    if not meta_path.exists():
        raise FileNotFoundError(
            "candidate_meta.pkl not found. Run precompute.py first."
        )

    print("Loading candidate_meta.pkl...")
    with open(meta_path, "rb") as f:
        metadata = pickle.load(f)

    print(f"Tokenizing {len(metadata):,} candidate documents for BM25...")
    corpus_texts = []
    candidate_ids = []
    for meta in tqdm(metadata, desc="BM25 tokenization"):
        doc = meta.get("doc_text") or ""
        corpus_texts.append(doc)
        candidate_ids.append(meta["candidate_id"])

    print("Building bm25s index...")
    # bm25s provides its own tokenization that handles batching well
    # We use token_pattern to match exactly our previous tokenize_bm25:
    corpus_tokens = bm25s.tokenize(
        corpus_texts,
        token_pattern=r"[a-z0-9][a-z0-9\-]*[a-z0-9]|[a-z0-9]",
        lower=True
    )

    bm25 = bm25s.BM25()
    bm25.index(corpus_tokens)

    out_path = BASE / "bm25_index"
    if out_path.exists():
        shutil.rmtree(out_path)

    bm25.save(str(out_path), corpus=candidate_ids)

    print(f"\nDone. Saved {out_path.name} directory ({len(candidate_ids):,} documents).")
    print("Now run: python rank.py")


if __name__ == "__main__":
    main()
