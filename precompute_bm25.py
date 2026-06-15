"""
precompute_bm25.py — Build a BM25 sparse retrieval index for all 100,000 candidates.
Runs in ~60-90 seconds. Saves bm25_index.pkl alongside candidate_db.pkl.

BM25 captures EXACT keyword matches (e.g. "FAISS", "RAG", "NDCG") that
dense embeddings sometimes miss. Combined with BGE via RRF, this gives
true hybrid search — the same architecture used by production search engines.

Run this ONCE. No GPU needed.
"""

import json
import pickle
import re
from pathlib import Path
from tqdm import tqdm
from rank_bm25 import BM25Okapi

BASE_DIR = Path(__file__).parent


def tokenize(text: str) -> list:
    """Simple whitespace + punctuation tokenizer, lowercase."""
    text = text.lower()
    # Keep alphanumeric and hyphens (important for terms like "bi-encoder")
    tokens = re.findall(r"[a-z0-9][a-z0-9\-]*[a-z0-9]|[a-z0-9]", text)
    return tokens


def build_doc(candidate: dict) -> str:
    """Build a BM25-friendly document from the candidate record.
    For BM25 we want keyword density, not semantic sentences.
    We repeat important sections to boost their weight.
    """
    profile = candidate.get("profile", {})
    skills = candidate.get("skills", [])
    career = candidate.get("career_history", [])

    parts = []

    # Headline + summary (1x)
    if profile.get("headline"):
        parts.append(profile["headline"])
    if profile.get("summary"):
        parts.append(profile["summary"])

    # Current title (2x — important signal)
    title = profile.get("current_title", "")
    if title:
        parts.append(title)
        parts.append(title)

    # All skills by name (2x for advanced/expert)
    for s in skills:
        name = s.get("name", "")
        if name:
            parts.append(name)
            if s.get("proficiency") in ("advanced", "expert"):
                parts.append(name)  # extra weight

    # Career descriptions (3x — richest signal)
    for exp in career:
        desc = exp.get("description", "")
        exp_title = exp.get("title", "")
        if exp_title:
            parts.append(exp_title)
        if desc:
            parts.append(desc)
            parts.append(desc)
            parts.append(desc)

    return " ".join(parts)


def main():
    print("Reading candidates.jsonl...")
    with open(BASE_DIR / "candidates.jsonl", "r", encoding="utf-8") as f:
        lines = f.readlines()

    print(f"Loaded {len(lines):,} candidates. Building BM25 documents...")

    candidate_ids = []
    tokenized_docs = []

    for line in tqdm(lines, desc="Tokenizing"):
        c = json.loads(line)
        doc = build_doc(c)
        tokens = tokenize(doc)
        candidate_ids.append(c["candidate_id"])
        tokenized_docs.append(tokens)

    print("Building BM25 index (this takes ~30-60 seconds)...")
    bm25 = BM25Okapi(tokenized_docs)

    print("Saving bm25_index.pkl...")
    with open(BASE_DIR / "bm25_index.pkl", "wb") as f:
        pickle.dump({
            "bm25": bm25,
            "candidate_ids": candidate_ids,
        }, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"\nDONE. bm25_index.pkl saved ({len(candidate_ids):,} documents).")
    print("Now update rank.py to use BM25 as a 3rd retrieval signal (3-way RRF).")


if __name__ == "__main__":
    main()
