import os
import tempfile
from pathlib import Path
from functools import lru_cache
import html as _html

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import gradio as gr
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, CrossEncoder
import pickle

BASE = Path(__file__).parent

from ranking_pipeline import RankingConfig, rank_candidates_core
from core_scoring import (
    reload_config,
    W_SEMANTIC, W_EXPERIENCE, W_COMPANY_TYPE, W_ML_SIGNALS,
    W_BEHAVIORAL, W_SAVED_RECRUITERS, W_GITHUB, W_EDUCATION,
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

_CE_POOL_SIZE = 500
_CE_BATCH_SIZE = 128


@lru_cache(maxsize=1)
def load_artifacts():
    db_path = BASE / "candidate_meta.pkl"
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
    else:
        print("WARNING: faiss_index.bin missing — ranking will fail until precompute.py is run.")

    return metadata, bm25, bm25_ids, index


@lru_cache(maxsize=1)
def load_models():
    local_bge = BASE / "models" / "bge-base-en-v1.5"
    local_ce = BASE / "models" / "ms-marco-MiniLM-L-6-v2"
    finetuned_ce = BASE / "models" / "finetuned-ce-model"

    if local_bge.exists():
        print("Loading BGE bi-encoder from ./models/...")
        bi = SentenceTransformer(str(local_bge))
    else:
        print("Loading BGE bi-encoder from HuggingFace...")
        bi = SentenceTransformer("BAAI/bge-base-en-v1.5")

    if finetuned_ce.exists():
        print("Loading fine-tuned cross-encoder from ./models/...")
        ce = CrossEncoder(str(finetuned_ce))
    elif local_ce.exists():
        print("Loading MiniLM cross-encoder from ./models/...")
        ce = CrossEncoder(str(local_ce))
    else:
        print("Loading MiniLM cross-encoder from HuggingFace...")
        ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    lgb_model = None
    lgb_path = BASE / "lgbm_reranker.pkl"
    if lgb_path.exists():
        import joblib
        print("Loading LightGBM reranker...")
        lgb_model = joblib.load(lgb_path)
        if isinstance(lgb_model, dict) and "model" in lgb_model:
            lgb_model = lgb_model["model"]

    return bi, ce, lgb_model


def _default_weights():
    """Slider defaults from config.yaml (via core_scoring)."""
    return {
        "w_semantic": W_SEMANTIC,
        "w_experience": W_EXPERIENCE,
        "w_company": W_COMPANY_TYPE,
        "w_ml": W_ML_SIGNALS,
        "w_behavioral": W_BEHAVIORAL,
        "w_saved": W_SAVED_RECRUITERS,
        "w_github": W_GITHUB,
        "w_education": W_EDUCATION,
    }


_DEFAULTS = _default_weights()

reload_config()
_DEFAULTS = _default_weights()


def run_pipeline(
    jd_text: str,
    top_n: int = 100,
    k_retrieve: int = 800,
    w_semantic: float = W_SEMANTIC,
    w_experience: float = W_EXPERIENCE,
    w_company: float = W_COMPANY_TYPE,
    w_ml: float = W_ML_SIGNALS,
    w_behavioral: float = W_BEHAVIORAL,
    w_saved: float = W_SAVED_RECRUITERS,
    w_github: float = W_GITHUB,
    w_education: float = W_EDUCATION,
    progress=gr.Progress(),
):
    if not jd_text or len(jd_text.strip()) < 50:
        return "", "JD too short — need at least 50 characters.", None, []
    try:
        progress(0.05, desc="Loading artifacts...")
        metadata, bm25, bm25_ids, index = load_artifacts()
        progress(0.10, desc="Loading models...")
        bi_enc, cross_enc, lgb_model = load_models()

        defaults = _DEFAULTS
        weights_match_defaults = (
            abs(w_semantic - defaults["w_semantic"]) < 0.01
            and abs(w_experience - defaults["w_experience"]) < 0.01
            and abs(w_company - defaults["w_company"]) < 0.01
            and abs(w_ml - defaults["w_ml"]) < 0.01
            and abs(w_behavioral - defaults["w_behavioral"]) < 0.01
            and abs(w_saved - defaults["w_saved"]) < 0.01
            and abs(w_github - defaults["w_github"]) < 0.01
            and abs(w_education - defaults["w_education"]) < 0.01
        )
        submission_like = top_n == 100 and k_retrieve == 800 and weights_match_defaults

        cfg = (
            RankingConfig(apply_mmr=True, top_n=100, k_retrieve=800)
            if submission_like
            else RankingConfig(
                k_retrieve=int(k_retrieve),
                ce_pool_size=_CE_POOL_SIZE,
                ce_batch_size=_CE_BATCH_SIZE,
                top_n=int(top_n),
                apply_mmr=top_n == 100,
                w_semantic=w_semantic,
                w_experience=w_experience,
                w_company=w_company,
                w_ml=w_ml,
                w_behavioral=w_behavioral,
                w_saved=w_saved,
                w_github=w_github,
                w_education=w_education,
            )
        )

        if index is None:
            return "", (
                "faiss_index.bin is missing. Run precompute.py locally, "
                "or upload precomputed artifacts to this Space."
            ), None, []

        def _progress(frac, desc=""):
            progress(frac, desc=desc)

        top = rank_candidates_core(
            jd_text=jd_text,
            metadata=metadata,
            faiss_index=index,
            bi_enc=bi_enc,
            cross_enc=cross_enc,
            bm25=bm25,
            bm25_candidate_ids=bm25_ids,
            lgb_model=lgb_model,
            config=cfg,
            progress=_progress,
        )

        max_saves = max((r["meta"].get("saved_by_recruiters", 0) or 0 for r in top), default=1)
        max_saves = max(max_saves, 1)
        for r in top:
            saves = r["meta"].get("saved_by_recruiters", 0) or 0
            r["recruiter_demand"] = round((saves / max_saves) * 100.0)

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
                recruiter_demand_html = (
                    f'<span class="signal-chip" style="color: #a1d489; border-color: #467434;" '
                    f'title="{_saves} recruiters saved this candidate in the last 30 days">'
                    f'DEMAND: {r["recruiter_demand"]}%</span>'
                )

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

        fig, ax = plt.subplots(figsize=(9, 3))
        scores = [max(r["score"], 0) for r in top]
        ax.bar(range(1, len(scores) + 1), scores, color="#F58F20", alpha=0.8, width=0.8)
        if scores:
            ax.axhline(sum(scores) / len(scores), color="#467434", linestyle="--", label="Mean")
        ax.set_xlabel("Rank")
        ax.set_ylabel("Score")
        ax.set_title("Score distribution")
        ax.legend()
        plt.tight_layout()

        mode = "submission-equivalent" if weights_match_defaults else "custom weights"
        status = (
            f"Ranked {len(top)} candidates from {min(k_retrieve, len(metadata)):,} retrieved "
            f"(BGE + BM25 -> MiniLM -> RRF, {mode})."
        )
        return full_html, status, fig, top

    except Exception as e:
        import traceback
        return f"<div>Error: {e}</div>", f"Error: {e}\n{traceback.format_exc()}", None, []


def build_app():
    with gr.Blocks(title="TALENT_TERMINAL") as demo:
        gr.Markdown("# <span class='logo-dot'></span>TALENT_TERMINAL — Candidate Ranking Pipeline")
        gr.Markdown("BGE Dense + BM25 Sparse → MiniLM Cross-Encoder → Heuristic Modifiers + RRF")

        with gr.Row():
            with gr.Column(scale=2):
                jd_input = gr.Textbox(label="Job Description", lines=14, placeholder="Paste full JD here...")

                with gr.Accordion("Weight Tuning", open=False):
                    with gr.Row():
                        w_sem = gr.Slider(20, 80, value=W_SEMANTIC, step=1, label="Semantic (Cross-Encoder)")
                        w_exp = gr.Slider(5, 40, value=W_EXPERIENCE, step=1, label="Experience Years")
                    with gr.Row():
                        w_co = gr.Slider(5, 30, value=W_COMPANY_TYPE, step=1, label="Product Company")
                        w_ml = gr.Slider(5, 30, value=W_ML_SIGNALS, step=1, label="Production ML Signals")
                    with gr.Row():
                        w_beh = gr.Slider(2, 20, value=W_BEHAVIORAL, step=1, label="Behavioral/Availability")
                        w_sav = gr.Slider(2, 20, value=W_SAVED_RECRUITERS, step=1, label="Saved by Recruiters")
                    with gr.Row():
                        w_gh = gr.Slider(0, 15, value=W_GITHUB, step=1, label="GitHub Activity")
                        w_edu = gr.Slider(0, 10, value=W_EDUCATION, step=1, label="Tier-1 Education")
                    with gr.Row():
                        top_n = gr.Slider(10, 100, value=100, step=10, label="Results to return")
                        k_ret = gr.Slider(500, 5000, value=800, step=100, label="Stage 1 pool size (CE scores top 500)")

                with gr.Row():
                    run_btn = gr.Button("Rank Candidates", variant="primary", scale=3, elem_classes=["primary-btn"])
                    clear_btn = gr.Button("Clear", scale=1)

                status = gr.Textbox(label="Status", interactive=False, lines=2)

            with gr.Column(scale=1):
                gr.Markdown("""
### How it works
**Stage 1** — BGE bi-encoder + BM25 → top 800

**Stage 2** — MiniLM cross-encoder reads JD + profile simultaneously

**Stage 3** — Gaussian experience curve, product company bonus, consulting penalty, behavioral availability, GitHub, assessments, RRF fusion

**Disqualifiers**
- Honeypot profiles
- Wrong current title
- Consulting-only (-25pts)
- Title-chasers (-12pts)

Default weights + top 100 uses the same MMR pass as `rank.py`.
                """)

        with gr.Tabs():
            with gr.TabItem("Results"):
                results_html = gr.HTML(label="Results")
            with gr.TabItem("Score Distribution"):
                score_plot = gr.Plot()
            with gr.TabItem("Download CSV"):
                dl_btn = gr.Button("Generate submission.csv", variant="secondary")
                dl_file = gr.File(label="Download", visible=False)

        state = gr.State([])

        def on_download(res):
            if not res:
                return gr.update(visible=False)
            tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8", newline="")
            import csv as _csv
            w = _csv.writer(tmp)
            w.writerow(["candidate_id", "rank", "score", "reasoning"])
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
        body_text_color_dark="#e8e1db",
    )
    app = build_app()
    app.queue(max_size=3)
    app.launch(server_name="0.0.0.0", show_error=True, theme=custom_theme, css=PREMIUM_CSS)
