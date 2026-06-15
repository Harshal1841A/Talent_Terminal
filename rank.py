"""
rank.py — Final Ranking Script (runs in < 5 minutes at judging time)
Requires candidate_db.pkl produced by precompute.py.
Outputs: submission.csv

Stage-3 judging: fully offline (no network). Models loaded from ./models/.
"""

import os
# Must be set BEFORE any HuggingFace import so no network calls are made.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import pickle
import csv
import json
import re
import math
import html
import torch
import numpy as np
import yaml
import faiss
from jinja2 import Environment, FileSystemLoader
from pathlib import Path
from sentence_transformers import SentenceTransformer, CrossEncoder

BASE = Path(__file__).parent


def tokenize_bm25(text: str) -> list:
    """Same tokenizer as precompute_bm25.py — must match exactly."""
    text = text.lower()
    return re.findall(r"[a-z0-9][a-z0-9\-]*[a-z0-9]|[a-z0-9]", text)


from core_scoring import (
    W_SEMANTIC, W_COMPANY_TYPE, W_EDUCATION,
    HONEYPOT_PENALTY, WRONG_TITLE_PENALTY, CONSULTING_ONLY_PENALTY, TITLE_CHASER_PENALTY,
    RRF_K, RRF_WEIGHT,
    norm_semantic, score_experience, score_ml_signals, score_location,
    score_ml_role_ratio, score_saved_by_recruiters, score_profile_completeness,
    score_engagement, score_trust, score_behavioral, score_github, score_assessment,
    build_features, generate_reasoning
)
# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def rank_candidates():
    # 1. Load JD
    print("Loading Job Description...")
    try:
        with open(BASE / "job_desc.txt", "r", encoding="utf-8") as f:
            jd_text = f.read()
    except FileNotFoundError:
        print("ERROR: job_desc.txt not found. Please extract text from job_description.docx.")
        return

    # 2. Load models + embed JD (multi-phrase query expansion)
    print("Loading Bi-Encoder and Cross-Encoder models...")
    bi_enc = SentenceTransformer(str(BASE / "models" / "bge-base-en-v1.5"))
    cross_enc = CrossEncoder(str(BASE / "models" / "ms-marco-MiniLM-L-6-v2"))
    
    print("Loading LightGBM reranker...")
    lgb_path = BASE / "lgbm_reranker.pkl"
    if lgb_path.exists():
        import joblib
        lgb_model = joblib.load(lgb_path)
    else:
        lgb_model = None
    jd_phrases = [
        jd_text,  # Full JD (primary)
        "Senior ML engineer with experience in information retrieval, ranking, embeddings, FAISS, vector search, LLMs, NLP",
        "5-9 years experience product company applied machine learning search ranking recommendation",
        "Python PyTorch TensorFlow scikit-learn Elasticsearch Solr Spark MLOps production",
    ]
    print("  Computing multi-phrase JD embedding...")
    phrase_embeddings = bi_enc.encode(
        jd_phrases,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )
    # Average the phrase embeddings and re-normalize
    jd_embedding = phrase_embeddings.mean(dim=0)
    jd_embedding = jd_embedding / jd_embedding.norm()  # re-normalize
    print("  Multi-phrase JD embedding ready.")

    # 3. Load pre-computed candidate metadata and FAISS index
    print("Loading candidate_meta.pkl and FAISS index...")
    with open(BASE / "candidate_meta.pkl", "rb") as f:
        metadata = pickle.load(f)
        
    faiss_index = faiss.read_index(str(BASE / "faiss_index.bin"))

    # 4. Stage 1: Dense Retrieval (FAISS)
    k_retrieve = min(2000, len(metadata))
    print(f"Stage 1: Computing FAISS dense retrieval for top {k_retrieve} candidates...")
    
    jd_emb_np = jd_embedding.cpu().numpy().reshape(1, -1)
    # Using FAISS Inner Product Index returns cosine similarities since vectors are normalized
    cosine_scores, top_k_indices_array = faiss_index.search(jd_emb_np, k_retrieve)
    
    top_k_indices = top_k_indices_array[0].tolist()

    # 5. Load BM25 index (optional — degrades gracefully if not found)
    bm25 = None
    bm25_id_to_idx = {}
    if os.path.exists(BASE / "bm25_index.pkl"):
        print("Loading BM25 index...")
        with open(BASE / "bm25_index.pkl", "rb") as f:
            bm25_data = pickle.load(f)
        bm25 = bm25_data["bm25"]
        bm25_id_to_idx = {cid: i for i, cid in enumerate(bm25_data["candidate_ids"])}
        print(f"  BM25 index loaded ({len(bm25_id_to_idx):,} documents).")
    else:
        print("[INFO] bm25_index.pkl not found — running without BM25 (2-way RRF only).")
        print("       Run precompute_bm25.py once to enable 3-way hybrid search.")

    # 5. Cross-Encoder re-ranking
    print(f"Stage 2: Running Cross-Encoder re-ranking on Top {len(top_k_indices)} candidates...")
    cross_inputs = [[jd_text, metadata[idx]["doc_text"]] for idx in top_k_indices]
    # batch_size=64 caps peak RAM usage; default (full batch) can OOM on 16GB with 1000 pairs
    cross_scores = cross_enc.predict(cross_inputs, batch_size=64)
    cross_scores_arr = np.array(cross_scores)

    # 6b. LightGBM Reranker (Optional, generated by train_reranker.py)
    lgb_model = None
    lgb_path = BASE / "lgbm_reranker.pkl"
    if os.path.exists(lgb_path):
        print("Loading LightGBM reranker...")
        import joblib
        lgb_model = joblib.load(lgb_path)
        
        print("  Predicting LightGBM scores...")
        X_lgb = np.array([build_features(metadata[idx]) for idx in top_k_indices], dtype=np.float32)
        lgb_preds = lgb_model.predict(X_lgb)
    else:
        print("[INFO] lgbm_reranker.pkl not found — skipping LightGBM behavioral boost.")
        lgb_preds = np.zeros(len(top_k_indices))

    # Using imported norm_semantic from core_scoring.py

    # 7. Apply behavioral modifiers + 3-way RRF fusion
    print("Stage 3: Applying hybrid scoring with behavioral modifiers + RRF...")

    # BM25 retrieval: get scores for ALL candidates via JD keywords
    bm25_rank_lookup = {}  # candidate_id -> bm25 rank
    if bm25 is not None:
        jd_tokens = tokenize_bm25(jd_text)
        bm25_scores = bm25.get_scores(jd_tokens)  # array of length 100,000
        # Build rank lookup: lower rank = higher score
        bm25_ranked_indices = sorted(range(len(bm25_scores)), key=lambda i: -bm25_scores[i])
        for rank_pos, raw_idx in enumerate(bm25_ranked_indices, 1):
            bm25_rank_lookup[raw_idx] = rank_pos
    results = []

    # Build bi-encoder rank lookup (rank by cosine similarity order within top_k)
    bienc_rank_lookup = {idx: r + 1 for r, idx in enumerate(top_k_indices)}

    for i, original_idx in enumerate(top_k_indices):
        meta = metadata[original_idx]
        raw_ce = float(cross_scores[i])
        semantic = norm_semantic(raw_ce)

        # Cross-encoder rank is assigned after the scoring loop (see ce_sorted below)

        # Hard disqualifiers
        if meta["honeypot"]:
            final = HONEYPOT_PENALTY
        elif meta["wrong_title"]:
            final = WRONG_TITLE_PENALTY + semantic
        else:
            # Additive hybrid scoring
            exp_score       = score_experience(meta["years_exp"])
            company_score   = (
                W_COMPANY_TYPE if meta["has_product_company"] else 0.0
            ) + (CONSULTING_ONLY_PENALTY if meta["consulting_only"] else 0.0)
            ml_score        = score_ml_signals(meta["ml_signal_count"])
            behav_score     = score_behavioral(meta)
            gh_score        = score_github(meta["github_score"])
            assess_score    = score_assessment(meta["core_skill_score"], meta["avg_assessment"])
            edu_score       = W_EDUCATION if meta["edu_tier_1"] else 0.0
            saved_score     = score_saved_by_recruiters(meta.get("saved_by_recruiters", 0) or 0)
            complete_score  = score_profile_completeness(meta.get("profile_completeness", 50) or 50)
            engage_score    = score_engagement(meta)
            trust_score     = score_trust(meta)

            loc_score    = score_location(meta.get("location_score", 0.0))
            ml_ratio_score = score_ml_role_ratio(meta.get("ml_role_ratio", 0.5))

            final = (
                semantic
                + exp_score
                + company_score
                + ml_score
                + behav_score
                + gh_score
                + assess_score
                + edu_score
                + saved_score
                + complete_score
                + engage_score
                + trust_score
                + meta.get("research_founding_score", 0.0)
                + loc_score
                + ml_ratio_score
            )

            # Title-chaser soft penalty
            if meta.get("title_chaser"):
                final += TITLE_CHASER_PENALTY

            # Apply LightGBM boost (up to +30%)
            lgb_score = max(0.0, float(lgb_preds[i]))
            meta["lgbm_score"] = lgb_score
            final = final * (1.0 + 0.3 * lgb_score)

            if meta.get("skill_count", 0) > 80 and meta.get("years_exp", 0) < 3:
                final -= 50

        results.append({
            "candidate_id": meta["candidate_id"],
            "score": final,
            "raw_ce": raw_ce,
            "bienc_rank": bienc_rank_lookup[original_idx],
            "reasoning": "",
            "meta": meta,
            "breakdown": {
                "Semantic": semantic if not meta.get("honeypot") and not meta.get("wrong_title") else 0,
                "Location": loc_score if not meta.get("honeypot") and not meta.get("wrong_title") else 0,
                "ML Ratio": ml_ratio_score if not meta.get("honeypot") and not meta.get("wrong_title") else 0,
                "Experience": exp_score if not meta.get("honeypot") and not meta.get("wrong_title") else 0,
                "Company": company_score if not meta.get("honeypot") and not meta.get("wrong_title") else 0,
                "ML Signals": ml_score if not meta.get("honeypot") and not meta.get("wrong_title") else 0,
                "Behavioral": behav_score if not meta.get("honeypot") and not meta.get("wrong_title") else 0,
                "GitHub": gh_score if not meta.get("honeypot") and not meta.get("wrong_title") else 0,
                "Assessment": assess_score if not meta.get("honeypot") and not meta.get("wrong_title") else 0,
            }
        })

    # BUG-03 FIX: previous filter `score > WRONG_TITLE_PENALTY` included wrong_title
    # candidates (score ≈ -492 > -500) in the CE rank, so they received a valid CE rank
    # and could get up to +10 RRF pts, partially un-disqualifying them.
    ce_sorted = sorted(
        [r for r in results if not r["meta"].get("honeypot") and not r["meta"].get("wrong_title")],
        key=lambda x: -x["raw_ce"]
    )
    for ce_rank_pos, r in enumerate(ce_sorted, 1):
        r["ce_rank"] = ce_rank_pos

    # Assign ce_rank=9999 for disqualified
    for r in results:
        if "ce_rank" not in r:
            r["ce_rank"] = 9999

    # RRF score: sum of 1/(k+rank) across all available signals
    # BUG-03 FIX: only boost genuinely eligible candidates (no honeypots, no wrong-title)
    for r in results:
        if not r["meta"].get("honeypot") and not r["meta"].get("wrong_title"):
            rrf_bienc  = 1.0 / (RRF_K + r["bienc_rank"])
            rrf_ce     = 1.0 / (RRF_K + r["ce_rank"])

            if bm25 is not None and r["candidate_id"] in bm25_id_to_idx:
                raw_bm25_idx = bm25_id_to_idx[r["candidate_id"]]
                rrf_bm25 = 1.0 / (RRF_K + bm25_rank_lookup.get(raw_bm25_idx, 99999))
                rrf_score = rrf_bienc + rrf_ce + rrf_bm25
                rrf_max = 3.0 / (RRF_K + 1)
            else:
                rrf_score = rrf_bienc + rrf_ce
                rrf_max = 2.0 / (RRF_K + 1)

            # Properly scale to exactly RRF_WEIGHT for the theoretical #1 rank in all lists
            r["score"] += (rrf_score / rrf_max) * RRF_WEIGHT

    for r in results:
        r["reasoning"] = generate_reasoning(r["meta"], r["raw_ce"], r["score"])

    # 7. Sort: score DESC, then candidate_id ASC for tie-breaking
    print("Sorting and selecting top 100...")
    results.sort(key=lambda x: (-x["score"], x["candidate_id"]))
    top_100 = results[:100]

    # 8. Write submission.csv
    print("Writing submission.csv...")
    with open(BASE / "submission.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank_pos, c in enumerate(top_100, 1):
            writer.writerow([
                c["candidate_id"],
                rank_pos,
                f"{c['score']:.4f}",
                c["reasoning"],
            ])

    # 9. Generate dashboard.html
    print("Generating dashboard.html...")
    _write_dashboard(top_100, BASE)

    # 10. Print summary
    scores = [r["score"] for r in top_100]
    print(f"\nDONE. submission.csv written ({len(top_100)} rows).")
    print(f"  Score range (top 100): {min(scores):.2f} – {max(scores):.2f}")
    print(f"  Top 5 candidates:")
    for c in top_100[:5]:
        print(f"    [{c['candidate_id']}] score={c['score']:.2f}  ce={c['raw_ce']:.4f}")
    print("\nNow validate with: python validate_submission.py submission.csv")
    print("Open dashboard.html in your browser to view results visually.")


def _write_dashboard(top_100: list, base_dir: str):
    """Generate a beautiful HTML dashboard showing the top 100 ranked candidates."""
    # BUG-14 FIX: guard against empty top_100 to prevent IndexError on top_100[0] / top_100[-1]
    # BUG-04 FIX: use only positive scores for max_score so bars don't invert when
    # most candidates are disqualified and max_score would otherwise be negative.
    positive_scores = [c["score"] for c in top_100 if c["score"] > 0]
    max_score = max(positive_scores, default=1)

    cards = []
    for rank_pos, c in enumerate(top_100, 1):
        meta = c["meta"]
        score = c["score"]
        score_pct = max(0, min(100, (score / max_score) * 100))
        reasoning = html.escape(c["reasoning"])

        # Badge color based on rank
        if rank_pos <= 3:
            badge_class = "badge-gold"
        elif rank_pos <= 10:
            badge_class = "badge-silver"
        else:
            badge_class = "badge-bronze"

        # Score bar color
        if score_pct >= 75:
            bar_color = "#22c55e"
        elif score_pct >= 50:
            bar_color = "#3b82f6"
        elif score_pct >= 30:
            bar_color = "#f59e0b"
        else:
            bar_color = "#ef4444"

        # Signals badges
        signals = []
        if meta.get("open_to_work"):
            signals.append({"css_class": "signal-green", "text": "Open to Work"})
        if meta.get("has_product_company"):
            signals.append({"css_class": "signal-blue", "text": "Product Co."})
        if meta.get("edu_tier_1"):
            signals.append({"css_class": "signal-purple", "text": "Tier-1 Edu"})
        if meta.get("consulting_only"):
            signals.append({"css_class": "signal-red", "text": "Consulting Only"})
        nd = meta.get("notice_days", 90)
        if nd == 0:
            signals.append({"css_class": "signal-green", "text": "Immediate"})
        elif nd <= 30:
            signals.append({"css_class": "signal-green", "text": f"{nd}d Notice"})
        elif nd > 90:
            signals.append({"css_class": "signal-red", "text": f"{nd}d Notice"})
        gh = meta.get("github_score", -1)
        if gh >= 70:
            signals.append({"css_class": "signal-blue", "text": f"GitHub ★ {gh:.0f}"})
        ml = meta.get("ml_signal_count", 0)
        if ml >= 0.8:
            signals.append({"css_class": "signal-purple", "text": "Strong ML Signals"})
        elif ml >= 0.5:
            signals.append({"css_class": "signal-purple", "text": "Moderate ML Signals"})

        breakdown = {k: v for k, v in c.get('breakdown', {}).items() if v != 0}

        cards.append({
            "rank_pos": rank_pos,
            "badge_class": badge_class,
            "candidate_id": c['candidate_id'],
            "title": html.escape(meta.get('current_title', 'Unknown Role') or 'Unknown Role'),
            "company": html.escape(meta.get('current_company', '') or ''),
            "score": score,
            "score_pct": score_pct,
            "bar_color": bar_color,
            "yrs": meta.get('years_exp', 0),
            "bienc_r": c.get('bienc_rank', '?'),
            "ce_r": c.get('ce_rank', '?'),
            "breakdown": breakdown,
            "signals": signals,
            "reasoning": c['reasoning']
        })

    _top1_score  = f"{top_100[0]['score']:.1f}"  if top_100 else "N/A"
    _topN_score  = f"{top_100[-1]['score']:.1f}" if top_100 else "N/A"
    _n_open      = sum(1 for c in top_100 if c["meta"].get("open_to_work"))
    _n_product   = sum(1 for c in top_100 if c["meta"].get("has_product_company"))

    env = Environment(loader=FileSystemLoader(str(base_dir)))
    template = env.get_template("dashboard_template.html")
    
    html_content = template.render(
        num_candidates=len(top_100),
        top1_score=_top1_score,
        topN_score=_topN_score,
        n_open=_n_open,
        n_product=_n_product,
        cards=cards
    )

    out_path = Path(base_dir) / "dashboard.html"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"\nCreated static dashboard: {out_path.absolute()}")


if __name__ == "__main__":
    rank_candidates()
