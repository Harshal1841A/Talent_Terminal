"""
Unified ranking pipeline shared by rank.py (submission) and app.py (Gradio demo).

Both entry points must call rank_candidates_core() so submission CSV and live demo
produce identical rankings when using default config weights.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from core_scoring import (
    W_SEMANTIC,
    W_EXPERIENCE,
    W_COMPANY_TYPE,
    W_ML_SIGNALS,
    W_BEHAVIORAL,
    W_SAVED_RECRUITERS,
    W_GITHUB,
    W_ASSESSMENT,
    W_EDUCATION,
    W_LOCATION,
    W_ML_RATIO,
    HONEYPOT_PENALTY,
    WRONG_TITLE_PENALTY,
    CONSULTING_ONLY_PENALTY,
    LGBM_SCORE_MULTIPLIER,
    apply_availability_location_penalties,
    RRF_K,
    RRF_WEIGHT,
    norm_semantic,
    score_experience,
    score_ml_signals,
    score_location,
    score_ml_role_ratio,
    score_saved_by_recruiters,
    score_profile_completeness,
    score_engagement,
    score_trust,
    score_behavioral,
    score_recency,
    score_github,
    score_assessment,
    score_jd_term_bonus,
    score_elite_company_bonus,
    qualifies_for_product_bonus,
    build_features,
    generate_reasoning,
)

BASE = Path(__file__).parent

# Query-expansion phrases — only used when JD matches bundled job_desc.txt.
JD_EXPANSION_PHRASES = [
    "Senior ML engineer with experience in information retrieval, ranking, embeddings, FAISS, vector search, LLMs, NLP",
    "5-9 years experience product company applied machine learning search ranking recommendation",
    "Python PyTorch TensorFlow scikit-learn Elasticsearch Solr Spark MLOps production",
]

_CE_DOC_MAX_CHARS = 1500
_CE_JD_MAX_CHARS = 1500
_SCORE_TIE_EPS = 1e-4


def tokenize_bm25(text: str) -> list:
    """Tokenizer for BM25 — must match precompute_bm25.py exactly."""
    text = text.lower()
    return re.findall(r"[a-z0-9][a-z0-9\-]*[a-z0-9]|[a-z0-9]", text)


def _normalize_jd_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def is_bundled_jd(jd_text: str) -> bool:
    """True when the pasted JD is the challenge's bundled job_desc.txt."""
    jd_path = BASE / "job_desc.txt"
    if not jd_path.exists():
        return False
    bundled = jd_path.read_text(encoding="utf-8")
    a = _normalize_jd_text(jd_text)[:2000]
    b = _normalize_jd_text(bundled)[:2000]
    return a == b or (len(a) > 200 and a[:200] == b[:200])


def embed_jd(bi_enc, jd_text: str, *, use_expansion: bool = True):
    """Multi-phrase JD embedding when using the bundled challenge JD."""
    jd_phrases = [jd_text]
    if use_expansion:
        jd_phrases.extend(JD_EXPANSION_PHRASES)
    phrase_embeddings = bi_enc.encode(
        jd_phrases,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )
    jd_embedding = phrase_embeddings.mean(dim=0)
    jd_embedding = jd_embedding / jd_embedding.norm()
    return jd_embedding


@dataclass
class RankingConfig:
    """Tunable ranking parameters. Defaults match config.yaml / submission path."""

    k_retrieve: int = 800
    ce_pool_size: int = 500
    ce_batch_size: int = 64
    top_n: int = 100
    apply_mmr: bool = True
    use_jd_expansion: bool = False
    w_semantic: float = W_SEMANTIC
    w_experience: float = W_EXPERIENCE
    w_company: float = W_COMPANY_TYPE
    w_ml: float = W_ML_SIGNALS
    w_behavioral: float = W_BEHAVIORAL
    w_saved: float = W_SAVED_RECRUITERS
    w_github: float = W_GITHUB
    w_education: float = W_EDUCATION
    w_location: float = W_LOCATION
    w_ml_ratio: float = W_ML_RATIO
    rrf_k: float = RRF_K
    rrf_weight: float = RRF_WEIGHT


@dataclass
class RetrievalCache:
    """Stages 1–2 outputs reused for fast Stage-3 weight tuning."""

    top_k_indices: list[int]
    cross_scores: list[float]
    lgb_preds: np.ndarray
    bm25_rank_lookup: dict[int, int]


ProgressCallback = Optional[Callable[[float, str], None]]


def _report(progress: ProgressCallback, frac: float, desc: str) -> None:
    if progress is not None:
        progress(frac, desc)


def enforce_submission_scores(ranked: list[dict]) -> list[dict]:
    """
    Ensure non-increasing scores by rank with ascending candidate_id tie-break
    (validate_submission.py rules).
    """
    if not ranked:
        return ranked
    for i in range(1, len(ranked)):
        prev = ranked[i - 1]
        curr = ranked[i]
        if curr["score"] > prev["score"]:
            curr["score"] = prev["score"]
        elif abs(curr["score"] - prev["score"]) < 1e-12:
            if curr["candidate_id"] <= prev["candidate_id"]:
                curr["score"] = prev["score"] - _SCORE_TIE_EPS
    return ranked


def apply_mmr(candidates: list[dict], top_n: int) -> list[dict]:
    """Notice-period diversity re-ranking for the top_n slots."""
    remaining = sorted(candidates, key=lambda x: (-x["score"], x["candidate_id"]))
    selected: list[dict] = []
    selected_notice_categories: list[int] = []

    while len(selected) < top_n and remaining:
        best_idx = 0
        best_score = -999999.0
        best_cid = ""

        for i, r in enumerate(remaining):
            nd = r["meta"].get("notice_days", 60) or 60
            nd_cat = 0 if nd <= 15 else (1 if nd <= 45 else (2 if nd <= 90 else 3))

            penalty = 0.0
            same_cat_count = sum(1 for cat in selected_notice_categories if cat == nd_cat)
            if same_cat_count >= 5:
                penalty = r["score"] * 0.15

            current_score = r["score"] - penalty
            cid = r["candidate_id"]
            if current_score > best_score or (
                current_score == best_score and (not best_cid or cid < best_cid)
            ):
                best_score = current_score
                best_idx = i
                best_cid = cid

        best_cand = remaining.pop(best_idx)
        effective_score = best_score
        if selected:
            effective_score = min(best_score, selected[-1]["score"])
        best_cand["score"] = effective_score

        nd = best_cand["meta"].get("notice_days", 60) or 60
        nd_cat = 0 if nd <= 15 else (1 if nd <= 45 else (2 if nd <= 90 else 3))
        selected_notice_categories.append(nd_cat)
        selected.append(best_cand)

    return enforce_submission_scores(selected)


def build_retrieval_cache(
    *,
    jd_text: str,
    metadata: list,
    faiss_index,
    jd_emb,
    cross_enc_path,
    cross_enc=None,          # pre-loaded CrossEncoder; if given, cross_enc_path is ignored
    bm25=None,
    lgb_model=None,
    config: Optional[RankingConfig] = None,
    progress: ProgressCallback = None,
) -> RetrievalCache:
    """Run Stages 1–2 once; cache outputs for fast Stage-3 rescoring."""
    cfg = config or RankingConfig()
    use_expansion = cfg.use_jd_expansion and is_bundled_jd(jd_text)

    k = min(cfg.k_retrieve, len(metadata))
    _report(progress, 0.30, f"Stage 1: dense retrieval over {len(metadata):,} candidates...")
    jd_emb_np = jd_emb.cpu().numpy().reshape(1, -1) if hasattr(jd_emb, "cpu") else np.array(jd_emb).reshape(1, -1)
    _, top_k_indices_array = faiss_index.search(jd_emb_np, k)

    faiss_top_k = top_k_indices_array[0].tolist()
    ce_pool_idx = faiss_top_k[: cfg.ce_pool_size]
    top_k_indices = faiss_top_k

    bm25_rank_lookup: dict[int, int] = {}
    if bm25 is not None:
        jd_tokens = tokenize_bm25(jd_text)
        bm25_scores = bm25.get_scores(jd_tokens)
        bm25_ranked_indices = sorted(range(len(bm25_scores)), key=lambda i: -bm25_scores[i])
        for rank_pos, raw_idx in enumerate(bm25_ranked_indices, 1):
            bm25_rank_lookup[raw_idx] = rank_pos

    jd_short = jd_text[:_CE_JD_MAX_CHARS]
    cross_inputs = [
        [jd_short, (metadata[idx]["doc_text"] or "")[:_CE_DOC_MAX_CHARS]]
        for idx in ce_pool_idx
    ]

    n_ce = len(cross_inputs)
    _report(progress, 0.50, f"Stage 2: cross-encoder over {n_ce} candidates...")

    loaded_locally = False
    if cross_enc is None:
        print("Loading Cross-Encoder...")
        from sentence_transformers import CrossEncoder
        cross_enc = CrossEncoder(cross_enc_path)
        loaded_locally = True

    ce_scores_pool = cross_enc.predict(cross_inputs, batch_size=cfg.ce_batch_size)
    ce_scores_pool = np.array(ce_scores_pool, dtype=np.float32)
    
    if loaded_locally:
        del cross_enc
        import gc
        gc.collect()

    ce_sentinel = float(ce_scores_pool.min()) if len(ce_scores_pool) else -9.0
    ce_score_map = {idx: float(ce_scores_pool[i]) for i, idx in enumerate(ce_pool_idx)}
    for idx in faiss_top_k[cfg.ce_pool_size :]:
        ce_score_map[idx] = ce_sentinel

    cross_scores = [ce_score_map[idx] for idx in top_k_indices]

    if lgb_model is not None:
        _report(progress, 0.70, "Predicting LightGBM scores...")
        X_lgb = np.array([build_features(metadata[idx]) for idx in top_k_indices], dtype=np.float32)
        lgb_preds = lgb_model.predict(X_lgb)
    else:
        lgb_preds = np.zeros(len(top_k_indices))

    return RetrievalCache(
        top_k_indices=top_k_indices,
        cross_scores=cross_scores,
        lgb_preds=lgb_preds,
        bm25_rank_lookup=bm25_rank_lookup,
    )


def finalize_ranking(
    *,
    metadata: list,
    cache: RetrievalCache,
    bm25=None,
    bm25_candidate_ids: Optional[dict] = None,
    config: Optional[RankingConfig] = None,
    include_reasoning: bool = True,
) -> list[dict]:
    """Stage 3 + RRF + optional MMR from a precomputed retrieval cache."""
    cfg = config or RankingConfig()
    bm25_candidate_ids = bm25_candidate_ids or {}
    top_k_indices = cache.top_k_indices

    bienc_rank_lookup = {idx: r + 1 for r, idx in enumerate(top_k_indices)}
    results: list[dict] = []

    for i, original_idx in enumerate(top_k_indices):
        meta = metadata[original_idx]
        raw_ce = float(cache.cross_scores[i])
        semantic = norm_semantic(raw_ce, cfg.w_semantic)

        loc_score = 0.0
        ml_ratio_score = 0.0
        exp_score = 0.0
        company_score = 0.0
        ml_score = 0.0
        behav_score = 0.0
        gh_score = 0.0
        assess_score = 0.0
        recency_score = 0.0
        jd_bonus_score = 0.0
        elite_bonus_score = 0.0
        lgb_score = 0.0

        if meta.get("honeypot"):
            final = HONEYPOT_PENALTY
        elif meta.get("wrong_title"):
            final = WRONG_TITLE_PENALTY + semantic
        else:
            exp_score = score_experience(meta.get("years_exp", 0) or 0, cfg.w_experience)
            company_score = (
                cfg.w_company if qualifies_for_product_bonus(meta) else 0.0
            ) + (CONSULTING_ONLY_PENALTY if meta.get("consulting_only") else 0.0)
            ml_score = score_ml_signals(meta.get("ml_signal_count", 0) or 0, cfg.w_ml)
            behav_score = score_behavioral(meta, cfg.w_behavioral)
            recency_score = score_recency(meta)
            gh_score = score_github(meta.get("github_score", -1) or -1, cfg.w_github)
            assess_score = score_assessment(
                meta.get("core_skill_score", -1) or -1,
                meta.get("avg_assessment", -1) or -1,
                W_ASSESSMENT,
            )
            edu_score = cfg.w_education if meta.get("edu_tier_1") else 0.0
            saved_score = score_saved_by_recruiters(
                meta.get("saved_by_recruiters", 0) or 0, cfg.w_saved
            )
            complete_score = score_profile_completeness(
                meta.get("profile_completeness", 50) or 50
            )
            engage_score = score_engagement(meta)
            trust_score = score_trust(meta)
            loc_score = score_location(meta.get("location_score", 0.0), cfg.w_location)
            ml_ratio_score = score_ml_role_ratio(meta.get("ml_role_ratio", 0.5), cfg.w_ml_ratio)
            jd_bonus_score = score_jd_term_bonus(meta)
            elite_bonus_score = score_elite_company_bonus(meta)

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
                + float(meta.get("research_founding_score", 0.0) or 0.0)
                + loc_score
                + ml_ratio_score
            )

            final += apply_availability_location_penalties(meta)

            lgb_score = max(0.0, float(cache.lgb_preds[i]))
            final = final * (1.0 + LGBM_SCORE_MULTIPLIER * lgb_score)

        results.append({
            "candidate_id": meta["candidate_id"],
            "score": final,
            "raw_ce": raw_ce,
            "bienc_rank": bienc_rank_lookup[original_idx],
            "reasoning": "",
            "meta": meta,
            "lgbm_score": lgb_score,
            "breakdown": {
                "Semantic": semantic if not meta.get("honeypot") and not meta.get("wrong_title") else 0,
                "Location": loc_score if not meta.get("honeypot") and not meta.get("wrong_title") else 0,
                "ML Ratio": ml_ratio_score if not meta.get("honeypot") and not meta.get("wrong_title") else 0,
                "Experience": exp_score if not meta.get("honeypot") and not meta.get("wrong_title") else 0,
                "Company": company_score if not meta.get("honeypot") and not meta.get("wrong_title") else 0,
                "ML Signals": ml_score if not meta.get("honeypot") and not meta.get("wrong_title") else 0,
                "Behavioral": behav_score if not meta.get("honeypot") and not meta.get("wrong_title") else 0,
                "Recency": recency_score if not meta.get("honeypot") and not meta.get("wrong_title") else 0,
                "JD Terms": jd_bonus_score if not meta.get("honeypot") and not meta.get("wrong_title") else 0,
                "Elite Co": elite_bonus_score if not meta.get("honeypot") and not meta.get("wrong_title") else 0,
                "GitHub": gh_score if not meta.get("honeypot") and not meta.get("wrong_title") else 0,
                "Assessment": assess_score if not meta.get("honeypot") and not meta.get("wrong_title") else 0,
            },
        })

    ce_sorted = sorted(
        [r for r in results if not r["meta"].get("honeypot") and not r["meta"].get("wrong_title")],
        key=lambda x: -x["raw_ce"],
    )
    for ce_rank_pos, r in enumerate(ce_sorted, 1):
        r["ce_rank"] = ce_rank_pos
    for r in results:
        if "ce_rank" not in r:
            r["ce_rank"] = 9999

    for r in results:
        if not r["meta"].get("honeypot") and not r["meta"].get("wrong_title"):
            rrf_bienc = 1.0 / (cfg.rrf_k + r["bienc_rank"])
            rrf_ce = 1.0 / (cfg.rrf_k + r["ce_rank"])

            if bm25 is not None and r["candidate_id"] in bm25_candidate_ids:
                raw_bm25_idx = bm25_candidate_ids[r["candidate_id"]]
                rrf_bm25 = 1.0 / (cfg.rrf_k + cache.bm25_rank_lookup.get(raw_bm25_idx, 99999))
                rrf_score = rrf_bienc + rrf_ce + rrf_bm25
                rrf_max = 3.0 / (cfg.rrf_k + 1)
            else:
                rrf_score = rrf_bienc + rrf_ce
                rrf_max = 2.0 / (cfg.rrf_k + 1)

            r["score"] += (rrf_score / rrf_max) * cfg.rrf_weight

    if include_reasoning:
        for r in results:
            r["reasoning"] = generate_reasoning(r["meta"], r["raw_ce"], r["score"])

    if cfg.apply_mmr:
        ranked = apply_mmr(results, cfg.top_n)
    else:
        ranked = sorted(results, key=lambda x: (-x["score"], x["candidate_id"]))[: cfg.top_n]
        ranked = enforce_submission_scores(ranked)

    for rank_pos, r in enumerate(ranked, 1):
        r["rank"] = rank_pos

    return ranked


def rank_candidates_core(
    *,
    jd_text: str,
    metadata: list,
    faiss_index,
    jd_emb,
    cross_enc_path,
    cross_enc=None,          # pre-loaded CrossEncoder object (optional)
    bm25=None,
    bm25_candidate_ids: Optional[dict] = None,
    lgb_model=None,
    config: Optional[RankingConfig] = None,
    progress: ProgressCallback = None,
) -> list[dict]:
    """
    Run the full 3-stage ranking pipeline.

    Returns a list of result dicts sorted by final score (desc), each containing:
    candidate_id, score, raw_ce, bienc_rank, ce_rank, reasoning, meta, breakdown.
    """
    if faiss_index is None:
        raise FileNotFoundError(
            "faiss_index.bin is required. Run: python precompute.py"
        )

    cfg = config or RankingConfig()
    cache = build_retrieval_cache(
        jd_text=jd_text,
        metadata=metadata,
        faiss_index=faiss_index,
        jd_emb=jd_emb,
        cross_enc_path=cross_enc_path,
        cross_enc=cross_enc,
        bm25=bm25,
        lgb_model=lgb_model,
        config=cfg,
        progress=progress,
    )
    ranked = finalize_ranking(
        metadata=metadata,
        cache=cache,
        bm25=bm25,
        bm25_candidate_ids=bm25_candidate_ids,
        config=cfg,
        include_reasoning=True,
    )
    _report(progress, 1.0, "Done!")
    return ranked
