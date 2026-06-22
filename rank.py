"""
rank.py — Final Ranking Script (runs in < 5 minutes at judging time)
Requires candidate_meta.pkl + faiss_index.bin produced by precompute.py.
Outputs: Team Rocket.csv + dashboard.html

Stage-3 judging: fully offline (no network). Models loaded from ./models/.
"""

import os

# Must be set BEFORE any HuggingFace import so no network calls are made.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import csv

import pickle
import sys
from pathlib import Path

import faiss
import yaml

from sentence_transformers import CrossEncoder, SentenceTransformer

from ranking_pipeline import RankingConfig, rank_candidates_core, tokenize_bm25
from core_scoring import reload_config

BASE = Path(__file__).parent


def rank_candidates():
    reload_config()
    print("Loading Job Description...")
    try:
        with open(BASE / "job_desc.txt", "r", encoding="utf-8") as f:
            jd_text = f.read()
    except FileNotFoundError:
        print("ERROR: job_desc.txt not found. Please extract text from job_description.docx.")
        return

    print("Loading Bi-Encoder...")
    bi_enc = SentenceTransformer(str(BASE / "models" / "bge-base-en-v1.5"))
    jd_emb = bi_enc.encode(jd_text, normalize_embeddings=True)
    del bi_enc
    import gc
    gc.collect()

    finetuned_ce_path = BASE / "models" / "finetuned-ce-model"
    if finetuned_ce_path.exists():
        cross_enc_path = str(finetuned_ce_path)
    else:
        cross_enc_path = str(BASE / "models" / "ms-marco-MiniLM-L-6-v2")

    print("Loading candidate_meta.pkl and FAISS index...")
    with open(BASE / "candidate_meta.pkl", "rb") as f:
        metadata = pickle.load(f)
    faiss_index = faiss.read_index(str(BASE / "faiss_index.bin"))

    bm25 = None
    bm25_candidate_ids = {}
    if os.path.exists(BASE / "bm25_index"):
        print("Loading BM25 index...")
        import bm25s
        bm25 = bm25s.BM25.load(str(BASE / "bm25_index"), load_corpus=True, mmap=True)
        bm25_candidate_ids = {cid["text"]: i for i, cid in enumerate(bm25.corpus)}
        print(f"  BM25 index loaded ({len(bm25_candidate_ids):,} documents).")
    else:
        print("[INFO] bm25_index directory not found — running without BM25 (2-way RRF only).")
        print("       Run precompute_bm25.py once to enable 3-way hybrid search.")

    lgb_model = None
    lgb_path = BASE / "lgbm_reranker.pkl"
    if os.path.exists(lgb_path):
        print("Loading LightGBM reranker...")
        import joblib
        lgb_model = joblib.load(lgb_path)
        if isinstance(lgb_model, dict) and "model" in lgb_model:
            lgb_model = lgb_model["model"]
    else:
        print("[INFO] lgbm_reranker.pkl not found — skipping LightGBM behavioral boost.")

    def _progress(frac, desc=""):
        print(f"  [{frac:.0%}] {desc}")

    try:
        top_100 = rank_candidates_core(
            jd_text=jd_text,
            jd_emb=jd_emb,
            metadata=metadata,
            faiss_index=faiss_index,
            cross_enc_path=cross_enc_path,
            bm25=bm25,
            bm25_candidate_ids=bm25_candidate_ids,
            lgb_model=lgb_model,
            config=RankingConfig(apply_mmr=False, top_n=100),
            progress=_progress,
        )
        
        # Free memory after ranking
        faiss_index = None
        bm25 = None
        bm25_data = None
        metadata = None
        gc.collect()
    except Exception as e:
        print(f"CRASH in ranking pipeline: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    meta_path = BASE / "submission_metadata.yaml"
    team_name = "submission"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as mf:
            submit_meta = yaml.safe_load(mf)
            if submit_meta:
                team_name = submit_meta.get("team_name", "submission").strip()
            team_name = "".join(c for c in team_name if c.isalnum() or c in (" ", "_", "-")).strip()

    out_file = BASE / f"{team_name}.csv"
    print(f"Writing {out_file.name}...")
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for c in top_100:
            writer.writerow([
                c["candidate_id"],
                c["rank"],
                f"{c['score']:.4f}",
                c["reasoning"],
            ])



    import json
    with open(BASE / "real_top_100.json", "w", encoding="utf-8") as f:
        json.dump(top_100, f)

    scores = [r["score"] for r in top_100]
    print(f"\nDONE. {out_file.name} written ({len(top_100)} rows).")
    print(f"  Score range (top 100): {min(scores):.2f} – {max(scores):.2f}")
    print("  Top 5 candidates:")
    for c in top_100[:5]:
        print(f"    [{c['candidate_id']}] score={c['score']:.2f}  ce={c['raw_ce']:.4f}")

    from validate_submission import validate_submission
    val_errors = validate_submission(out_file)
    if val_errors:
        print(f"\nVALIDATION FAILED ({len(val_errors)} issue(s)):")
        for e in val_errors:
            print(f"  - {e}")
        sys.exit(1)
    print(f"\nSubmission validated: {out_file.name}")

if __name__ == "__main__":
    rank_candidates()
