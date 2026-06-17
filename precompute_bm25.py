"""
precompute_bm25.py — Build BM25 sparse index for hybrid retrieval.

Requires candidate_meta.pkl (from precompute.py).
Outputs: bm25_index.pkl

Tokenizer must stay in sync with ranking_pipeline.tokenize_bm25().
"""

import pickle
from pathlib import Path

from rank_bm25 import BM25Okapi
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
    corpus_tokens = []
    candidate_ids = []
    for meta in tqdm(metadata, desc="BM25 tokenization"):
        doc = meta.get("doc_text") or ""
        corpus_tokens.append(tokenize_bm25(doc))
        candidate_ids.append(meta["candidate_id"])

    print("Building BM25Okapi index...")
    bm25 = BM25Okapi(corpus_tokens)

    out_path = BASE / "bm25_index.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(
            {"bm25": bm25, "candidate_ids": candidate_ids},
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    print(f"\nDone. Saved {out_path.name} ({len(candidate_ids):,} documents).")
    print("Now run: python rank.py")


if __name__ == "__main__":
    main()
