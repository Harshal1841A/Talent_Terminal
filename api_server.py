"""
api_server.py — FastAPI wrapper around the existing ranking pipeline.

Fixes applied:
  - CrossEncoder loaded ONCE at startup (Bug #1/#3)
  - Uses lifespan context instead of deprecated @on_event (Bug #5)
  - BM25 corpus id resolution handles plain string items (Bug #7)
  - Breakdown keys remapped to match frontend CandidateBreakdown interface (Bug #8)
  - Dead imports removed: gc, CrossEncoder at module level (Bug #2/#3)
  - Dead ProgressEvent model removed (Bug #4)
  - Health endpoint tracks a `_ready` flag (Bug #14)
  - CORS locked to GET/POST only (Bug #18)
  - ranking_pipeline.build_retrieval_cache now receives the pre-loaded cross_enc
    object via the new optional `cross_enc` parameter added to ranking_pipeline.py
"""

import os

# Must precede any HuggingFace import
# os.environ["HF_HUB_OFFLINE"] = "1"
# os.environ["TRANSFORMERS_OFFLINE"] = "1"

import pickle
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import faiss
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from core_scoring import reload_config
from ranking_pipeline import RankingConfig, rank_candidates_core

BASE = Path(__file__).parent

# ── State container ────────────────────────────────────────────────────────
_state: dict = {}
_ready: bool = False   # True only after ALL models are loaded successfully


def _load_models() -> None:
    global _ready
    _ready = False

    print("Loading candidate_meta.pkl and FAISS index...")
    with open(BASE / "candidate_meta.pkl", "rb") as f:
        _state["metadata"] = pickle.load(f)
    _state["faiss_index"] = faiss.read_index(str(BASE / "faiss_index.bin"))

    _state["bm25"] = None
    _state["bm25_candidate_ids"] = {}
    bm25_dir = BASE / "bm25_index"
    if bm25_dir.exists():
        print("Loading BM25 index...")
        import bm25s
        bm25 = bm25s.BM25.load(str(bm25_dir), load_corpus=True, mmap=True)
        _state["bm25"] = bm25

        # precompute_bm25.py saves corpus=candidate_ids (plain str list).
        # bm25s wraps each entry as {"text": <original_value>} on load.
        # The value stored IS the candidate_id string, not document text.
        corpus = bm25.corpus
        if corpus:
            sample = corpus[0]
            if isinstance(sample, dict):
                # {"text": "<candidate_id>"} — standard bm25s saved format
                _state["bm25_candidate_ids"] = {
                    item["text"]: i for i, item in enumerate(corpus)
                }
            else:
                # Bare strings (older bm25s version or custom save)
                _state["bm25_candidate_ids"] = {
                    str(item): i for i, item in enumerate(corpus)
                }
        print(f"  BM25 loaded ({len(_state['bm25_candidate_ids']):,} documents).")

    _state["lgb_model"] = None
    lgb_path = BASE / "lgbm_reranker.pkl"
    if lgb_path.exists():
        print("Loading LightGBM reranker...")
        import joblib
        lgb_model = joblib.load(lgb_path)
        if isinstance(lgb_model, dict) and "model" in lgb_model:
            lgb_model = lgb_model["model"]
        _state["lgb_model"] = lgb_model

    from sentence_transformers import SentenceTransformer, CrossEncoder
    print("Loading Bi-Encoder...")
    bi_enc_path = BASE / "models" / "bge-base-en-v1.5"
    if not bi_enc_path.exists():
        bi_enc_path = "BAAI/bge-base-en-v1.5"
    _state["bi_enc"] = SentenceTransformer(str(bi_enc_path))
    
    print("Loading Cross-Encoder...")
    finetuned_ce = BASE / "models" / "finetuned-ce-model"
    if finetuned_ce.exists():
        ce_path = str(finetuned_ce)
    else:
        ce_path = BASE / "models" / "ms-marco-MiniLM-L-6-v2"
        if not ce_path.exists():
            ce_path = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    _state["cross_enc_path"] = str(ce_path)
    _state["cross_enc"] = CrossEncoder(str(ce_path))


    reload_config()
    _ready = True
    print(f"API ready. {len(_state['metadata']):,} candidates loaded.")


# ── Lifespan (replaces deprecated @app.on_event) ──────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_models()
    yield
    # Teardown (if needed) goes here


app = FastAPI(title="Talent Terminal API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["GET", "POST"],        # only what we use
    allow_headers=["Content-Type"],
)


# ── Request / Response models ──────────────────────────────────────────────
class RankRequest(BaseModel):
    jd_text: str
    top_n: Optional[int] = 100


# Pipeline breakdown key → frontend interface key mapping.
# Must stay in sync with CandidateBreakdown in frontend/src/api.ts
_BREAKDOWN_MAP = {
    "Semantic":    "semantic",
    "Location":    "location",
    "ML Ratio":    "ml_ratio",
    "Experience":  "experience",
    "Company":     "company",
    "ML Signals":  "ml_signals",
    "Behavioral":  "behavioral",
    "Recency":     "recency",
    "JD Terms":    "jd_terms",
    "Elite Co":    "elite_co",
    "GitHub":      "github",
    "Assessment":  "assessment",
}


def _shape_result(r: dict, rank: int) -> dict:
    """Convert a raw pipeline result dict to the frontend-safe response shape."""
    meta = r["meta"]

    # Remap breakdown keys to what the TypeScript interface expects
    raw_bd = r.get("breakdown", {})
    breakdown = {
        _BREAKDOWN_MAP.get(k, k.lower().replace(" ", "_")): round(float(v), 3)
        for k, v in raw_bd.items()
    }

    return {
        "candidate_id": r["candidate_id"],
        "rank": rank,
        "score": round(float(r["score"]), 4),
        "reasoning": r.get("reasoning", ""),
        "breakdown": breakdown,
        "meta": {
            "current_title": meta.get("current_title") or "",
            "current_company": meta.get("current_company") or "",
            "years_exp": int(meta.get("years_exp") or 0),
            "edu_tier_1": bool(meta.get("edu_tier_1")),
            "open_to_work": bool(meta.get("open_to_work")),
            "willing_to_relocate": bool(meta.get("willing_to_relocate")),
            "github_score": float(meta.get("github_score") or -1),
            "avg_assessment": float(meta.get("avg_assessment") or -1),
            "saved_by_recruiters": int(meta.get("saved_by_recruiters") or 0),
            "linkedin_connected": bool(meta.get("linkedin_connected")),
            "verified_email": bool(meta.get("verified_email")),
            "verified_phone": bool(meta.get("verified_phone")),
            "notice_days": meta.get("notice_days"),    # may be None
            "preferred_work_mode": meta.get("preferred_work_mode") or "",
        },
    }


# ── Routes ─────────────────────────────────────────────────────────────────
import gc
@app.post("/api/rank")
def rank(req: RankRequest):
    if not _ready:
        raise HTTPException(status_code=503, detail="Models still loading, try again in a moment")

    if not req.jd_text or not req.jd_text.strip():
        raise HTTPException(status_code=400, detail="jd_text is required")

    # Using preloaded Bi-Encoder for speed
    jd_emb = _state["bi_enc"].encode(req.jd_text, normalize_embeddings=True)

    cfg = RankingConfig()
    cfg.top_n = req.top_n or 100

    try:
        results = rank_candidates_core(
            jd_text=req.jd_text,
            jd_emb=jd_emb,
            metadata=_state["metadata"],
            faiss_index=_state["faiss_index"],
            cross_enc=_state["cross_enc"],       # Use preloaded CrossEncoder
            cross_enc_path=_state["cross_enc_path"],              
            bm25=_state.get("bm25"),
            bm25_candidate_ids=_state.get("bm25_candidate_ids", {}),
            lgb_model=_state.get("lgb_model"),
            config=cfg,
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

    shaped = [_shape_result(r, i + 1) for i, r in enumerate(results)]
    return {"results": shaped, "count": len(shaped)}


@app.get("/api/health")
def health():
    return {
        "status": "ready" if _ready else "loading",
        "ready": _ready,
        "candidates_loaded": len(_state.get("metadata", [])),
    }

# ── Frontend SPA ───────────────────────────────────────────────────────────
frontend_dist = BASE / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/assets", StaticFiles(directory=frontend_dist / "assets"), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="Not Found")
        
        index_file = frontend_dist / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        return {"message": "Frontend not built yet. Run npm run build in frontend directory."}

