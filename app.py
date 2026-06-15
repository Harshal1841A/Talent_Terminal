import math, os, tempfile
from pathlib import Path
from functools import lru_cache
from datetime import date, datetime
import html as _html

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import gradio as gr
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, CrossEncoder
import pickle
import re


BASE = Path(__file__).parent

from core_scoring import (
    W_SEMANTIC, W_EXPERIENCE, W_COMPANY_TYPE, W_ML_SIGNALS, W_BEHAVIORAL,
    W_SAVED_RECRUITERS, W_GITHUB, W_ASSESSMENT, W_PROFILE_COMPLETE,
    W_EDUCATION, W_ENGAGEMENT, W_TRUST, RRF_K, RRF_WEIGHT,
    HONEYPOT_PENALTY, WRONG_TITLE_PENALTY, CONSULTING_ONLY_PENALTY,
    norm_semantic, build_features, score_experience, score_ml_signals,
    score_saved_by_recruiters, score_github, score_assessment, score_profile_completeness,
    score_engagement, score_trust, score_behavioral, generate_reasoning
)

PREMIUM_CSS = """
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
        
        :root {
            --tangerine: #F58F20;
            --leaf-green: #467434;
            --sea-grey: #363636;
            --deep-bg: #1c1c1c;
            --hover-grey: #444444;
        }

        body, .gradio-container {
            background-color: var(--deep-bg) !important;
            color: #e8e1db !important;
            font-family: 'Inter', sans-serif !important;
        }

        h1, h2, h3, h4, h5, h6, p, span, label, .prose * {
            color: #e8e1db !important;
        }

        textarea, input, select {
            background-color: var(--sea-grey) !important;
            color: #ffffff !important;
            border-color: #555 !important;
        }


        .candidate-card {
            background: var(--sea-grey);
            border: 1px solid rgba(255,255,255,0.07);
            transition: all 0.15s ease-out;
            margin-bottom: 16px;
            padding: 20px;
            border-radius: 8px;
        }
        .candidate-card:hover {
            border-color: rgba(245,143,32,0.25);
            box-shadow: 0 0 12px rgba(245,143,32,0.1);
            transform: translateY(-1px);
        }

        .rank-badge-elite {
            background: linear-gradient(135deg, #F58F20, #ffb779);
            color: #1c1c1c;
            box-shadow: 0 0 10px rgba(245,143,32,0.3);
            padding: 2px 12px;
            border-radius: 9999px;
            font-size: 11px;
            font-weight: bold;
        }
        .rank-badge-strong {
            background: linear-gradient(135deg, #467434, #a1d489);
            color: #ffffff;
            box-shadow: 0 0 10px rgba(70,116,52,0.3);
            padding: 2px 12px;
            border-radius: 9999px;
            font-size: 11px;
            font-weight: bold;
        }
        .rank-badge-neutral {
            background: var(--hover-grey);
            color: #e8e1db;
            padding: 2px 12px;
            border-radius: 9999px;
            font-size: 11px;
            font-weight: bold;
        }

        .score-bar-track {
            background: #2a2a2a;
            height: 6px;
            border-radius: 3px;
            overflow: hidden;
            margin: 10px 0;
        }
        .score-bar-fill {
            height: 100%;
            background: linear-gradient(to right, var(--leaf-green), var(--tangerine));
            animation: fillBar 1s ease-out forwards;
        }

        .signal-chip {
            border: 1px solid rgba(255,255,255,0.1);
            padding: 2px 8px;
            border-radius: 9999px;
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-right: 6px;
            display: inline-block;
            margin-bottom: 6px;
        }

        .primary-btn {
            background: var(--tangerine);
            color: #1c1c1c;
            font-weight: 800;
            text-transform: uppercase;
            box-shadow: 0 4px 20px rgba(245,143,32,0.25);
            transition: all 0.2s;
        }
        .primary-btn:hover {
            opacity: 0.9;
            box-shadow: 0 4px 25px rgba(245,143,32,0.4);
        }

        @keyframes fillBar {
            0% { width: 0; }
            100% { width: var(--target); }
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.5; transform: scale(1.2); }
        }

        .logo-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: linear-gradient(to right, var(--leaf-green), var(--tangerine));
            animation: pulse 2s infinite ease-in-out;
            display: inline-block;
            margin-right: 8px;
        }

        .weight-slider {
            -webkit-appearance: none;
            width: 100%;
            height: 4px;
            background: #363636;
            border-radius: 2px;
            outline: none;
        }
        .weight-slider::-webkit-slider-thumb {
            -webkit-appearance: none;
            appearance: none;
            width: 12px;
            height: 12px;
            background: var(--tangerine);
            cursor: pointer;
            border-radius: 50%;
        }

        .card-header { display: flex; justify-content: space-between; align-items: flex-start; }
        .score-value { font-size: 24px; font-family: 'JetBrains Mono', monospace; font-weight: 600; color: var(--tangerine); }
        .id-text { font-family: 'JetBrains Mono', monospace; font-size: 13px; color: #a38d7c; }
        .reasoning-text { font-size: 13px; color: #94a3b8; margin-top: 10px; line-height: 1.5; border-top: 1px solid rgba(255,255,255,0.05); padding-top: 8px; }
"""

@lru_cache(maxsize=1)
def load_artifacts():
    db_path   = BASE / "candidate_meta.pkl"
    bm25_path = BASE / "bm25_index.pkl"
    faiss_path = BASE / "faiss_index.bin"
    if not db_path.exists():
        raise FileNotFoundError("candidate_meta.pkl missing. Run precompute.py first.")
    print("Loading candidate_meta.pkl...")
    with open(db_path, "rb") as f:
        metadata = pickle.load(f)
    bm25, bm25_ids = None, {}
    if bm25_path.exists():
        print("Loading bm25_index.pkl...")
        with open(bm25_path, "rb") as f:
            bd = pickle.load(f)
        bm25 = bd["bm25"]
        bm25_ids = {cid: i for i, cid in enumerate(bd["candidate_ids"])}
        
    index = None
    if faiss_path.exists():
        import faiss
        print("Loading faiss_index.bin...")
        index = faiss.read_index(str(faiss_path))
        
    return metadata, bm25, bm25_ids, index

@lru_cache(maxsize=1)
def load_models():
    print("Loading BGE bi-encoder...")
    bi  = SentenceTransformer("BAAI/bge-base-en-v1.5")
    print("Loading MiniLM cross-encoder...")
    ce  = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    
    lgb_model = None
    lgb_path = BASE / "lgbm_reranker.pkl"
    if lgb_path.exists():
        import joblib
        print("Loading LightGBM reranker...")
        lgb_model = joblib.load(lgb_path)
        
    return bi, ce, lgb_model



# Max chars of profile text fed to the cross-encoder.
# MiniLM-L6 has a hard 512-token limit; ~350 chars ≈ 400 tokens — fast & accurate.
_CE_DOC_MAX_CHARS  = 1500
_CE_JD_MAX_CHARS   = 1500   # Keep JD to key requirements only
_CE_POOL_SIZE      = 500   # Top-N bi-encoder results to re-rank (was 2000)
_CE_BATCH_SIZE     = 128   # Predict batch size

def run_pipeline(
    jd_text: str,
    top_n: int = 100,
    k_retrieve: int = 2000,
    w_semantic: float = 60.0,
    w_experience: float = 25.0,
    w_company: float = 18.0,
    w_ml: float = 20.0,
    w_behavioral: float = 10.0,
    w_saved: float = 12.0,
    w_github: float = 5.0,
    w_education: float = 3.0,
    progress=gr.Progress(),
):
    if not jd_text or len(jd_text.strip()) < 50:
        return "", "❌ JD too short — need at least 50 characters.", None, []
    try:
        progress(0.05, desc="Loading artifacts...")
        metadata, bm25, bm25_ids, index = load_artifacts()
        progress(0.10, desc="Loading models...")
        bi_enc, cross_enc, lgb_model = load_models()

        progress(0.20, desc="Stage 1: embedding JD...")
        jd_phrases = [
            jd_text,
            "Senior ML engineer information retrieval ranking embeddings FAISS vector search LLMs NLP",
            "5-9 years experience product company applied machine learning search ranking recommendation",
            "Python PyTorch TensorFlow scikit-learn Elasticsearch Spark MLOps production deployment",
        ]
        phrase_embs = bi_enc.encode(jd_phrases, convert_to_tensor=True, normalize_embeddings=True)
        jd_emb = phrase_embs.mean(dim=0)
        jd_emb = jd_emb / jd_emb.norm()

        progress(0.30, desc="Stage 1: dense retrieval over 100K candidates...")
        k = min(k_retrieve, len(metadata))
        jd_emb_np = jd_emb.numpy().reshape(1, -1)
        _, I = index.search(jd_emb_np, k)
        top_idx = I[0].tolist()

        bm25_rank_lookup = {}
        if bm25:
            tokens = re.findall(r"[a-z0-9][a-z0-9\-]*[a-z0-9]|[a-z0-9]", jd_text.lower())
            bm25_scores_arr = bm25.get_scores(tokens)
            for pos, ridx in enumerate(sorted(range(len(bm25_scores_arr)), key=lambda i: -bm25_scores_arr[i]), 1):
                bm25_rank_lookup[ridx] = pos

        bienc_rank = {idx: r+1 for r, idx in enumerate(top_idx)}

        # ── Stage 2: Cross-encoder re-ranking ─────────────────────────────────
        # Only re-rank the top _CE_POOL_SIZE from the bi-encoder — the rest are
        # scored with a sentinel value and rely on heuristics + RRF alone.
        # This is the single biggest speedup: 2000 → 500 pairs = 4x faster.
        ce_pool_idx  = top_idx[:_CE_POOL_SIZE]   # top bi-encoder candidates
        rest_idx     = top_idx[_CE_POOL_SIZE:]    # beyond CE pool

        # Truncate inputs — MiniLM has a hard 512-token limit.
        # Sending 2000-char docs wastes time on tokens that get clipped anyway.
        jd_short = jd_text[:_CE_JD_MAX_CHARS]
        ce_inputs = [
            [jd_short, (metadata[i]["doc_text"] or "")[:_CE_DOC_MAX_CHARS]]
            for i in ce_pool_idx
        ]

        n_ce = len(ce_inputs)
        progress(0.50, desc=f"Stage 2: cross-encoder over {n_ce} candidates (pool={_CE_POOL_SIZE})...")

        ce_scores_pool = []
        for start in range(0, n_ce, _CE_BATCH_SIZE):
            batch = ce_inputs[start : start + _CE_BATCH_SIZE]
            scores = cross_enc.predict(batch, show_progress_bar=False)
            if isinstance(scores, (list, np.ndarray)):
                ce_scores_pool.extend(float(s) for s in scores)
            else:
                ce_scores_pool.append(float(scores))
            progress(
                0.50 + 0.20 * (start + len(batch)) / n_ce,
                desc=f"Stage 2: cross-encoder ({min(start + len(batch), n_ce)}/{n_ce})...",
            )

        ce_scores_pool = np.array(ce_scores_pool, dtype=np.float32)

        # Assign sentinel score (min of pool) to candidates outside the CE pool
        ce_sentinel = float(ce_scores_pool.min()) if len(ce_scores_pool) else -9.0
        ce_score_map = {idx: float(ce_scores_pool[i]) for i, idx in enumerate(ce_pool_idx)}
        for idx in rest_idx:
            ce_score_map[idx] = ce_sentinel

        # Rebuild full arrays in original top_idx order
        ce_scores_arr = np.array([ce_score_map[idx] for idx in top_idx], dtype=np.float32)

        # Using imported norm_semantic from core_scoring.py

        if lgb_model is not None:
            progress(0.70, desc="Predicting LightGBM scores...")
            X_lgb = np.array([build_features(metadata[idx]) for idx in top_idx], dtype=np.float32)
            lgb_preds = lgb_model.predict(X_lgb)
        else:
            lgb_preds = np.zeros(len(top_idx))

        progress(0.80, desc="Stage 3: heuristic scoring + RRF...")
        results = []
        for i, oidx in enumerate(top_idx):
            meta = metadata[oidx]
            raw_ce = float(ce_scores_arr[i])
            semantic = norm_semantic(raw_ce, w_semantic)

            if meta.get("honeypot"):
                final = HONEYPOT_PENALTY
            elif meta.get("wrong_title"):
                final = WRONG_TITLE_PENALTY + semantic
            else:
                company_s = (w_company if meta.get("has_product_company") else 0.0) + \
                            (CONSULTING_ONLY_PENALTY if meta.get("consulting_only") else 0.0)
                final = (
                    semantic
                    + score_experience(meta.get("years_exp", 0) or 0, w_experience)
                    + company_s
                    + score_ml_signals(meta.get("ml_signal_count", 0) or 0, w_ml)
                    + score_behavioral(meta, w_behavioral)
                    + score_github(meta.get("github_score", -1) or -1, w_github)
                    + score_assessment(meta.get("core_skill_score", -1) or -1, meta.get("avg_assessment", -1) or -1)
                    + (w_education if meta.get("edu_tier_1") else 0.0)
                    + score_saved_by_recruiters(meta.get("saved_by_recruiters", 0) or 0, w_saved)
                    + score_profile_completeness(meta.get("profile_completeness", 50) or 50)
                    + score_engagement(meta)
                    + score_trust(meta)
                    + meta.get("research_founding_score", 0.0)
                )

                lgb_score = max(0.0, float(lgb_preds[i]))
                meta["lgbm_score"] = lgb_score
                final = final * (1.0 + 0.3 * lgb_score)

                if (meta.get("skill_count", 0) or 0) > 80 and (meta.get("years_exp", 0) or 0) < 3:
                    final -= 50

            results.append({
                "candidate_id": meta["candidate_id"],
                "score": final,
                "raw_ce": raw_ce,
                "bienc_rank": bienc_rank[oidx],
                "reasoning": "",  # generated post-RRF so text reflects true final score
                "meta": meta,
            })

        # BUG-03 FIX: old filter `score > WRONG_TITLE_PENALTY` included wrong_title candidates
        # (score ≈ -492 > -500), giving them valid CE ranks and up to +10 RRF boost.
        eligible = [r for r in results if not r["meta"].get("honeypot") and not r["meta"].get("wrong_title")]
        for pos, r in enumerate(sorted(eligible, key=lambda x: -x["raw_ce"]), 1):
            r["ce_rank"] = pos
        for r in results:
            r.setdefault("ce_rank", 9999)

        # BUG-03 FIX: same guard — only apply RRF boost to genuinely eligible candidates
        for r in results:
            if not r["meta"].get("honeypot") and not r["meta"].get("wrong_title"):
                rrf_b = 1.0 / (RRF_K + r["bienc_rank"])
                rrf_c = 1.0 / (RRF_K + r["ce_rank"])
                if bm25 and r["candidate_id"] in bm25_ids:
                    ridx = bm25_ids[r["candidate_id"]]
                    rrf_bm = 1.0 / (RRF_K + bm25_rank_lookup.get(ridx, 99999))
                    rrf_score = rrf_b + rrf_c + rrf_bm
                    rrf_max = 3.0 / (RRF_K + 1)
                else:
                    rrf_score = rrf_b + rrf_c
                    rrf_max = 2.0 / (RRF_K + 1)

                r["score"] += (rrf_score / rrf_max) * RRF_WEIGHT

        results.sort(key=lambda x: (-x["score"], x["candidate_id"]))
        top = results[:top_n]
        for rank, r in enumerate(top, 1):
            r["rank"] = rank

        # FIX: generate reasoning AFTER RRF so the text reflects the true displayed score
        for r in top:
            r["reasoning"] = generate_reasoning(r["meta"], r["raw_ce"], r["score"])

        # FIX: honest Recruiter Demand signal — normalized historical saves in this result set.
        # Removed the LR model (trained on saved_by_recruiters labels then scored on same
        # data — pure leakage). This shows actual recruiter market interest for each candidate.
        max_saves = max((r["meta"].get("saved_by_recruiters", 0) or 0 for r in top), default=1)
        max_saves = max(max_saves, 1)
        for r in top:
            saves = r["meta"].get("saved_by_recruiters", 0) or 0
            r["recruiter_demand"] = round((saves / max_saves) * 100.0)

        # BUG-04 FIX: max_score must never be negative. When most candidates are
        # disqualified, max() returns a negative, inverting score bars (worst candidate
        # gets 100% bar width). Use only positive scores for normalization.
        _pos_scores = [r["score"] for r in top if r["score"] > 0]
        max_score = max(_pos_scores, default=1)
        html_cards = []
        for r in top:
            rank = r["rank"]
            score = r["score"]
            meta = r["meta"]
            score_pct = max(0, min(100, (score / max_score) * 100))
            
            if rank <= 3:
                badge_cls = "rank-badge-elite"
            elif rank <= 10:
                badge_cls = "rank-badge-strong"
            else:
                badge_cls = "rank-badge-neutral"

            signals = []
            if meta.get("open_to_work"):
                signals.append('<span class="signal-chip">OPEN TO WORK</span>')
            if meta.get("has_product_company"):
                signals.append('<span class="signal-chip">PRODUCT CO</span>')
            if meta.get("edu_tier_1"):
                signals.append('<span class="signal-chip">TIER 1 EDU</span>')
            if meta.get("consulting_only"):
                signals.append('<span class="signal-chip" style="color: #ef4444;">CONSULTING ONLY</span>')
                
            recruiter_demand_html = ""
            if r.get("recruiter_demand", 0) > 0:
                _saves = meta.get("saved_by_recruiters", 0) or 0
                recruiter_demand_html = f'<span class="signal-chip" style="color: #a1d489; border-color: #467434;" title="{_saves} recruiters saved this candidate in the last 30 days">DEMAND: {r["recruiter_demand"]}%</span>'

            title = _html.escape(str(meta.get("current_title", "Unknown Role") or "Unknown Role"))
            company = _html.escape(str(meta.get("current_company", "") or ""))
            reasoning = _html.escape(str(r["reasoning"] or ""))

            html_cards.append(f'''
            <div class="candidate-card">
                <div class="card-header">
                    <div>
                        <span class="{badge_cls}">#{rank}</span>
                        <div style="margin-top: 8px;">
                            <div class="id-text">{r["candidate_id"]}</div>
                            <div style="font-size: 15px; font-weight: 600; margin-top: 4px;">{title}{' @ ' + company if company else ''}</div>
                        </div>
                    </div>
                    <div style="text-align: right;">
                        <div class="score-value">{score:.1f}</div>
                        <div style="font-size: 10px; color: #a38d7c; letter-spacing: 0.5px;">SCORE</div>
                    </div>
                </div>
                <div class="score-bar-track">
                    <div class="score-bar-fill" style="--target: {score_pct:.1f}%"></div>
                </div>
                <div style="margin-bottom: 12px;">
                    {(" ".join(signals) + " " + recruiter_demand_html)}
                </div>
                <div class="reasoning-text">
                    {reasoning}
                </div>
            </div>
            ''')
            
        full_html = f'''
        <style>{PREMIUM_CSS}</style>
        <div style="background: var(--deep-bg); padding: 20px; border-radius: 8px;">
            {''.join(html_cards)}
        </div>
        '''

        fig, ax = plt.subplots(figsize=(9,3))
        # BUG-16 FIX: clamp chart values to 0 — disqualified candidates have scores of
        # -9999 (honeypot) and -492 (wrong_title). Including them destroys the Y-axis
        # scale, making all positive scores look identical at the top.
        scores = [max(r["score"], 0) for r in top]
        ax.bar(range(1, len(scores)+1), scores, color="#F58F20", alpha=0.8, width=0.8)
        if scores:
            ax.axhline(sum(scores)/len(scores), color="#467434", linestyle="--", label="Mean")
        ax.set_xlabel("Rank"); ax.set_ylabel("Score"); ax.set_title("Score distribution")
        ax.legend(); plt.tight_layout()

        progress(1.0, desc="Done!")
        return full_html, f"✅ Ranked {len(top)} candidates from {k:,} retrieved. BGE + BM25 → MiniLM → RRF.", fig, top

    except Exception as e:
        import traceback
        return f"<div>Error: {e}</div>", f"\u274c Error: {e}\n{traceback.format_exc()}", None, []

def build_app():
    with gr.Blocks(
        title="TALENT_TERMINAL",
    ) as demo:

        gr.Markdown("# <span class='logo-dot'></span>TALENT_TERMINAL — Candidate Ranking Pipeline")
        gr.Markdown("BGE Dense + BM25 Sparse → MiniLM Cross-Encoder → Heuristic Modifiers + RRF")

        with gr.Row():
            with gr.Column(scale=2):
                jd_input = gr.Textbox(label="Job Description", lines=14, placeholder="Paste full JD here...")

                with gr.Accordion("⚙️ Weight Tuning", open=False):
                    with gr.Row():
                        w_sem = gr.Slider(20, 80, value=60, step=1, label="Semantic (Cross-Encoder)")
                        w_exp = gr.Slider(5, 40, value=25, step=1, label="Experience Years")
                    with gr.Row():
                        w_co  = gr.Slider(5, 30, value=18, step=1, label="Product Company")
                        w_ml  = gr.Slider(5, 30, value=20, step=1, label="Production ML Signals")
                    with gr.Row():
                        w_beh = gr.Slider(2, 20, value=10, step=1, label="Behavioral/Availability")
                        w_sav = gr.Slider(2, 20, value=12, step=1, label="Saved by Recruiters")
                    with gr.Row():
                        w_gh  = gr.Slider(0, 15, value=5,  step=1, label="GitHub Activity")
                        w_edu = gr.Slider(0, 10, value=3,  step=1, label="Tier-1 Education")
                    with gr.Row():
                        top_n = gr.Slider(10, 100, value=100, step=10, label="Results to return")
                        k_ret = gr.Slider(500, 5000, value=2000, step=500, label="Stage 1 pool size (CE scores top 500)")

                with gr.Row():
                    run_btn   = gr.Button("🚀 Rank Candidates", variant="primary", scale=3, elem_classes=["primary-btn"])
                    clear_btn = gr.Button("🗑️ Clear", scale=1)

                status = gr.Textbox(label="Status", interactive=False, lines=2)

            with gr.Column(scale=1):
                gr.Markdown("""
### How it works
**Stage 1** — BGE bi-encoder + BM25 → top 2,000

**Stage 2** — MiniLM cross-encoder reads JD + profile simultaneously

**Stage 3** — Gaussian experience curve, product company bonus, consulting penalty, behavioral availability, GitHub, assessments, RRF fusion

**Disqualifiers**
- 🚫 Honeypot profiles
- 🚫 Wrong current title
- ⚠️ Consulting-only (−25pts)
                """)

        with gr.Tabs():
            with gr.TabItem("📋 Results"):
                results_html = gr.HTML(label="Results")
            with gr.TabItem("📈 Score Distribution"):
                score_plot = gr.Plot()
            with gr.TabItem("📥 Download CSV"):
                dl_btn  = gr.Button("Generate submission.csv", variant="secondary")
                dl_file = gr.File(label="Download", visible=False)

        state = gr.State([])

        def on_download(res):
            if not res: return gr.update(visible=False)
            tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8", newline="")
            import csv as _csv
            w = _csv.writer(tmp)
            w.writerow(["candidate_id","rank","score","reasoning"])
            for r in res:
                w.writerow([r["candidate_id"], r["rank"], f"{r['score']:.4f}", r["reasoning"]])
            tmp.close()
            return gr.update(value=tmp.name, visible=True)

        def on_clear():
            return "", "<div></div>", "Cleared.", None, []

        run_btn.click(
            fn=run_pipeline,
            inputs=[jd_input, top_n, k_ret, w_sem, w_exp, w_co, w_ml, w_beh, w_sav, w_gh, w_edu],
            outputs=[results_html, status, score_plot, state],
        )
        dl_btn.click(fn=on_download, inputs=[state], outputs=[dl_file])
        clear_btn.click(fn=on_clear, inputs=[], outputs=[jd_input, results_html, status, score_plot, state])

    return demo

if __name__ == "__main__":
    custom_theme = gr.themes.Default(
        primary_hue=gr.themes.colors.orange,
        neutral_hue=gr.themes.colors.gray,
    ).set(
        body_background_fill="#1c1c1c",
        body_background_fill_dark="#1c1c1c",
        block_background_fill="#363636",
        block_background_fill_dark="#363636",
        block_border_color="#444444",
        block_border_color_dark="#444444",
        block_label_text_color="#e8e1db",
        block_label_text_color_dark="#e8e1db",
        body_text_color="#e8e1db",
        body_text_color_dark="#e8e1db"
    )
    app = build_app()
    app.queue(max_size=3)
    app.launch(server_name="0.0.0.0", show_error=True, theme=custom_theme, css=PREMIUM_CSS)
