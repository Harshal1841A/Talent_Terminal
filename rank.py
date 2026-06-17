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
import html
import pickle
import sys
from pathlib import Path

import faiss
import yaml
from jinja2 import Environment, FileSystemLoader
from sentence_transformers import CrossEncoder, SentenceTransformer

from ranking_pipeline import RankingConfig, rank_candidates_core
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

    print("Loading Bi-Encoder and Cross-Encoder models...")
    bi_enc = SentenceTransformer(str(BASE / "models" / "bge-base-en-v1.5"))
    finetuned_ce_path = BASE / "models" / "finetuned-ce-model"
    if finetuned_ce_path.exists():
        cross_enc = CrossEncoder(str(finetuned_ce_path))
        print("Using finetuned-ce-model cross-encoder.")
    else:
        cross_enc = CrossEncoder(str(BASE / "models" / "ms-marco-MiniLM-L-6-v2"))
        print("Using base ms-marco-MiniLM-L-6-v2 cross-encoder (not finetuned-ce-model).")

    print("Loading candidate_meta.pkl and FAISS index...")
    with open(BASE / "candidate_meta.pkl", "rb") as f:
        metadata = pickle.load(f)
    faiss_index = faiss.read_index(str(BASE / "faiss_index.bin"))

    bm25 = None
    bm25_candidate_ids = {}
    if os.path.exists(BASE / "bm25_index.pkl"):
        print("Loading BM25 index...")
        with open(BASE / "bm25_index.pkl", "rb") as f:
            bm25_data = pickle.load(f)
        bm25 = bm25_data["bm25"]
        bm25_candidate_ids = {cid: i for i, cid in enumerate(bm25_data["candidate_ids"])}
        print(f"  BM25 index loaded ({len(bm25_candidate_ids):,} documents).")
    else:
        print("[INFO] bm25_index.pkl not found — running without BM25 (2-way RRF only).")
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
            metadata=metadata,
            faiss_index=faiss_index,
            bi_enc=bi_enc,
            cross_enc=cross_enc,
            bm25=bm25,
            bm25_candidate_ids=bm25_candidate_ids,
            lgb_model=lgb_model,
            config=RankingConfig(apply_mmr=True, top_n=100),
            progress=_progress,
        )
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

    print("Generating dashboard.html...")
    _write_dashboard(top_100, BASE)

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
    print("Open dashboard.html in your browser to view results visually.")


def _write_dashboard(top_100: list, base_dir: Path):
    """Generate a beautiful HTML dashboard showing the top 100 ranked candidates."""
    positive_scores = [c["score"] for c in top_100 if c["score"] > 0]
    max_score = max(positive_scores, default=1)

    cards = []
    for rank_pos, c in enumerate(top_100, 1):
        meta = c["meta"]
        score = c["score"]
        score_pct = max(0, min(100, (score / max_score) * 100))
        reasoning = html.escape(c["reasoning"])

        if rank_pos <= 3:
            badge_class = "badge-gold"
        elif rank_pos <= 10:
            badge_class = "badge-silver"
        else:
            badge_class = "badge-bronze"

        if score_pct >= 75:
            bar_color = "#22c55e"
        elif score_pct >= 50:
            bar_color = "#3b82f6"
        elif score_pct >= 30:
            bar_color = "#f59e0b"
        else:
            bar_color = "#ef4444"

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

        breakdown = {k: v for k, v in c.get("breakdown", {}).items() if v != 0}

        cards.append({
            "rank_pos": rank_pos,
            "badge_class": badge_class,
            "candidate_id": c["candidate_id"],
            "title": html.escape(meta.get("current_title", "Unknown Role") or "Unknown Role"),
            "company": html.escape(meta.get("current_company", "") or ""),
            "score": score,
            "score_pct": score_pct,
            "bar_color": bar_color,
            "yrs": meta.get("years_exp", 0),
            "bienc_r": c.get("bienc_rank", "?"),
            "ce_r": c.get("ce_rank", "?"),
            "breakdown": breakdown,
            "signals": signals,
            "reasoning": c["reasoning"],
        })

    _top1_score = f"{top_100[0]['score']:.1f}" if top_100 else "N/A"
    _topN_score = f"{top_100[-1]['score']:.1f}" if top_100 else "N/A"
    _n_open = sum(1 for c in top_100 if c["meta"].get("open_to_work"))
    _n_product = sum(1 for c in top_100 if c["meta"].get("has_product_company"))

    env = Environment(loader=FileSystemLoader(str(base_dir)))
    template = env.get_template("dashboard_template.html")
    html_content = template.render(
        num_candidates=len(top_100),
        top1_score=_top1_score,
        topN_score=_topN_score,
        n_open=_n_open,
        n_product=_n_product,
        cards=cards,
    )

    out_path = base_dir / "dashboard.html"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"\nCreated static dashboard: {out_path.absolute()}")


if __name__ == "__main__":
    rank_candidates()
