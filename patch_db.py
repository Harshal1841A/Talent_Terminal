"""
patch_db.py -- Inject Tier 1+2.1 features into existing candidate_db.pkl.

Instead of re-running precompute.py (~30-40 min), this script reads
candidates.jsonl and computes ONLY the new fields, then patches them
directly into the existing candidate_db.pkl.

New fields added:
  - location_score, location_score_city    (Tier 1.1)
  - ml_role_ratio, ml_role_months,         (Tier 1.2)
    total_months
  - research_founding_score (fixed logic)  (Tier 1.3)
  - title_chaser, avg_tenure_months        (Tier 1.4)
  - honeypot (strengthened to multi-signal) (Tier 2.1)

Usage:
    python patch_db.py
"""

import json
import pickle
import math
from datetime import date, datetime
from pathlib import Path
from tqdm import tqdm
from precompute import count_ml_signals

BASE = Path(__file__).parent

# ---- ML role title signals (Tier 1.2) -----------------------------------
ML_ROLE_TITLES = [
    "ml engineer", "machine learning", "ai engineer", "nlp engineer",
    "data scientist", "applied scientist", "search engineer",
    "recommendation", "ranking engineer", "research scientist",
    "deep learning", "computer vision", "ir engineer", "applied ml",
    "ml researcher", "ai researcher", "speech", "nlp",
]

# ---- Seniority rank for title-chaser (Tier 1.4) -------------------------
SENIORITY_RANK = {
    "intern": 0, "junior": 0, "associate": 0,
    "": 1,
    "senior": 2,
    "staff": 3, "lead": 3,
    "principal": 4, "director": 5, "vp": 5, "head": 4,
}

# ---- Preferred cities (Tier 1.1) -----------------------------------------
PUNE_NOIDA = ["pune", "noida"]
PREFERRED_CITIES = [
    "hyderabad", "mumbai", "delhi", "gurugram", "gurgaon",
    "bengaluru", "bangalore", "chennai", "kolkata",
]

DATASET_REF_DATE = date(2025, 6, 1)

# ---- ML signal lists (for research_founding_score fix) -------------------
PRODUCTION_ML_SIGNALS = [
    "recommendation system", "ranking", "retrieval", "search relevance",
    "faiss", "milvus", "qdrant", "pinecone", "weaviate", "vector db",
    "learning to rank", "cross-encoder", "bi-encoder", "embedding",
    "bert", "sbert", "sentence-transformers", "transformers", "huggingface",
    "pytorch", "tensorflow", "mlops", "sagemaker", "kubeflow", "mlflow",
    "tensorrt", "onnx", "triton", "llm", "rag", "retrieval augmented generation",
    "two-tower", "two tower", "hnsw", "ann index", "semantic search",
    "dense retrieval", "hybrid search", "bm25", "reranking", "re-ranking",
    "fine-tun", "deployed to production", "production deployment", "a/b test",
]


def get_seniority(title: str) -> int:
    t = title.lower()
    for kw, rank in sorted(SENIORITY_RANK.items(), key=lambda x: -x[1]):
        if kw and kw in t:
            return rank
    return 1


def compute_new_features(p: dict) -> dict:
    """Compute all Tier 1+2.1 new features from a raw candidate dict."""
    profile  = p.get("profile", {}) or {}
    signals  = p.get("redrob_signals", {}) or {}
    career   = p.get("career_history", []) or []
    skills   = p.get("skills", []) or []
    summary  = profile.get("summary", "") or ""

    # ---- Honeypot (Tier 2.1 strengthened) ----------------------------------
    expert_zero_count = sum(
        1 for s in skills
        if s.get("proficiency") == "expert" and (s.get("duration_months") or 0) == 0
    )
    timeline_impossible = False
    for exp in career:
        start_str = exp.get("start_date")
        end_str   = exp.get("end_date")
        dur       = exp.get("duration_months") or 0
        try:
            if start_str:
                s_d = datetime.strptime(start_str[:10], "%Y-%m-%d").date()
                if s_d > DATASET_REF_DATE:
                    timeline_impossible = True
                    break
            if start_str and end_str:
                s_d = datetime.strptime(start_str[:10], "%Y-%m-%d").date()
                e_d = datetime.strptime(end_str[:10], "%Y-%m-%d").date()
                if e_d < s_d:
                    timeline_impossible = True
                    break
                actual_months = (e_d.year - s_d.year) * 12 + (e_d.month - s_d.month)
                if dur > actual_months + 18:
                    timeline_impossible = True
                    break
        except Exception:
            pass

    honeypot = (
        expert_zero_count >= 3
        or (expert_zero_count >= 1 and timeline_impossible)
        or timeline_impossible
    )

    # ---- research_founding_score fix (Tier 1.3) ----------------------------
    all_titles_lower = [(exp.get("title", "") or "").lower() for exp in career]
    is_intern_only = len(all_titles_lower) > 0 and all(
        "intern" in t or "student" in t for t in all_titles_lower
    )
    research_founding_score = 0.0
    if is_intern_only:
        research_founding_score = -5.0
    else:
        ml_sig = count_ml_signals(career, summary)
        for t in all_titles_lower:
            if any(k in t for k in ["founder", "founding engineer", "cto"]):
                research_founding_score = 15.0
                break
            elif any(k in t for k in ["research scientist", "applied scientist"]):
                if ml_sig >= 0.3:
                    research_founding_score = 15.0
                elif ml_sig >= 0.1:
                    research_founding_score = 5.0
                else:
                    research_founding_score = 0.0
                break

    # ---- Location score (Tier 1.1) -----------------------------------------
    location_raw     = (profile.get("location", "") or "").lower()
    location_country = (profile.get("country", "") or "").lower()
    willing          = signals.get("willing_to_relocate", False)

    if any(c in location_raw for c in PUNE_NOIDA):
        location_score = 5.0
        location_score_city = next(c for c in PUNE_NOIDA if c in location_raw)
    elif any(c in location_raw for c in PREFERRED_CITIES):
        location_score = 3.0
        location_score_city = next(c for c in PREFERRED_CITIES if c in location_raw)
    elif "india" in location_country or "india" in location_raw:
        location_score = 3.0 if willing else 1.5
        location_score_city = "india"
    else:
        location_score = 1.0 if willing else 0.0
        location_score_city = "international"

    # ---- ML role ratio (Tier 1.2) ------------------------------------------
    ml_role_months = sum(
        (exp.get("duration_months") or 0)
        for exp in career
        if any(sig in (exp.get("title", "") or "").lower() for sig in ML_ROLE_TITLES)
    )
    total_months = sum((exp.get("duration_months") or 0) for exp in career)
    ml_role_ratio = ml_role_months / max(total_months, 1)

    # ---- Title chaser (Tier 1.4) -------------------------------------------
    non_current = [e for e in career if not e.get("is_current")]
    avg_tenure_months = (
        sum((e.get("duration_months") or 0) for e in non_current) / len(non_current)
        if non_current else 999.0
    )
    seniority_levels = [get_seniority(e.get("title", "")) for e in career]
    monotone_escalating = (
        len(seniority_levels) >= 3 and
        all(seniority_levels[i] >= seniority_levels[i + 1] for i in range(len(seniority_levels) - 1)) and
        seniority_levels[-1] < seniority_levels[0]
    )
    title_chaser = bool(avg_tenure_months < 18 and monotone_escalating)

    return {
        "honeypot":               honeypot,
        "research_founding_score": research_founding_score,
        "location_score":         location_score,
        "location_score_city":    location_score_city,
        "ml_role_ratio":          ml_role_ratio,
        "ml_role_months":         ml_role_months,
        "total_months":           total_months,
        "title_chaser":           title_chaser,
        "avg_tenure_months":      avg_tenure_months,
    }


# ============================================================
# MAIN
# ============================================================

print("Loading candidates.jsonl ...")
cid_to_raw: dict = {}
with open(BASE / "candidates.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        p = json.loads(line)
        cid_to_raw[p["candidate_id"]] = p

print(f"Loaded {len(cid_to_raw):,} candidates.")

print("Loading candidate_db.pkl ...")
with open(BASE / "candidate_db.pkl", "rb") as f:
    db = pickle.load(f)

print(f"DB has {len(db['metadata']):,} metadata entries.")

# Stats trackers
n_honeypot_new = 0
n_title_chaser = 0
n_location_5   = 0
n_location_0   = 0
n_research_fixed = 0

print("Patching metadata ...")
for m in tqdm(db["metadata"]):
    cid = m["candidate_id"]
    p = cid_to_raw.get(cid, {})
    if not p:
        # candidate not in jsonl -- fill with safe defaults
        m.setdefault("location_score", 1.5)
        m.setdefault("location_score_city", "unknown")
        m.setdefault("ml_role_ratio", 0.0)
        m.setdefault("ml_role_months", 0)
        m.setdefault("total_months", 0)
        m.setdefault("title_chaser", False)
        m.setdefault("avg_tenure_months", 999.0)
        continue

    new_feats = compute_new_features(p)
    m.update(new_feats)

    if new_feats["honeypot"] and not m.get("_old_honeypot"):
        n_honeypot_new += 1
    if new_feats["title_chaser"]:
        n_title_chaser += 1
    if new_feats["location_score"] >= 5.0:
        n_location_5 += 1
    if new_feats["location_score"] == 0.0:
        n_location_0 += 1

print("Saving patched candidate_db.pkl ...")
with open(BASE / "candidate_db.pkl", "wb") as f:
    pickle.dump(db, f, protocol=pickle.HIGHEST_PROTOCOL)

print("\n===== Patch Summary =====")
print(f"  Total candidates patched : {len(db['metadata']):,}")
print(f"  Honeypot (new detections): {n_honeypot_new:,}")
print(f"  Title-chasers flagged    : {n_title_chaser:,}")
print(f"  Pune/Noida location (5.0): {n_location_5:,}")
print(f"  International, no-reloc  : {n_location_0:,}")
print("\nDONE. candidate_db.pkl updated with Tier 1+2.1 features.")
