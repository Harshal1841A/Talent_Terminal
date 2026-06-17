"""
Fast weight tuning against proxy gold labels.

Runs Stages 1–2 once (~minutes), then random-searches Stage-3 weights (~seconds/trial).

Usage:
  python build_proxy_gold.py --min-relevance 1
  python tune_weights.py --trials 200
  python tune_weights.py --trials 50 --apply   # write best weights to config.yaml
"""

from __future__ import annotations

import argparse
import copy
import os
import pickle
import random
from pathlib import Path

import faiss
import yaml
from sentence_transformers import CrossEncoder, SentenceTransformer

from core_scoring import reload_config
from eval_metrics import evaluate_ranking
from ranking_pipeline import RankingConfig, build_retrieval_cache, finalize_ranking

BASE = Path(__file__).parent


def load_gold_map(path: Path) -> dict[str, int]:
    import csv
    gold = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rel = row.get("manual_relevance") or row.get("relevance")
            if rel is None or str(rel).strip() == "":
                continue
            gold[row["candidate_id"]] = int(float(rel))
    if not gold:
        raise ValueError(f"No labels found in {path}")
    return gold


def load_artifacts():
    with open(BASE / "candidate_meta.pkl", "rb") as f:
        metadata = pickle.load(f)
    faiss_index = faiss.read_index(str(BASE / "faiss_index.bin"))

    bm25, bm25_ids = None, {}
    bm25_path = BASE / "bm25_index.pkl"
    if bm25_path.exists():
        with open(bm25_path, "rb") as f:
            bd = pickle.load(f)
        bm25 = bd["bm25"]
        bm25_ids = {cid: i for i, cid in enumerate(bd["candidate_ids"])}

    lgb_model = None
    lgb_path = BASE / "lgbm_reranker.pkl"
    if lgb_path.exists():
        import joblib
        lgb_model = joblib.load(lgb_path)
        if isinstance(lgb_model, dict) and "model" in lgb_model:
            lgb_model = lgb_model["model"]

    return metadata, faiss_index, bm25, bm25_ids, lgb_model


def load_models(offline: bool):
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        bi = SentenceTransformer(str(BASE / "models" / "bge-base-en-v1.5"))
        ce = CrossEncoder(str(BASE / "models" / "ms-marco-MiniLM-L-6-v2"))
    else:
        bi = SentenceTransformer("BAAI/bge-base-en-v1.5")
        ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return bi, ce


def sample_config(rng: random.Random, base: RankingConfig) -> RankingConfig:
    return RankingConfig(
        k_retrieve=base.k_retrieve,
        ce_pool_size=base.ce_pool_size,
        ce_batch_size=base.ce_batch_size,
        top_n=100,
        apply_mmr=base.apply_mmr,
        w_semantic=rng.uniform(45, 75),
        w_experience=rng.uniform(15, 35),
        w_company=rng.uniform(10, 25),
        w_ml=rng.uniform(12, 28),
        w_behavioral=rng.uniform(6, 16),
        w_saved=rng.uniform(6, 18),
        w_github=rng.uniform(2, 10),
        w_education=rng.uniform(0, 6),
        w_location=rng.uniform(4, 15),
        w_ml_ratio=rng.uniform(8, 20),
        rrf_k=rng.choice([40, 50, 60, 80]),
        rrf_weight=rng.uniform(6, 14),
    )


def apply_config_to_yaml(cfg: RankingConfig, config_path: Path) -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    data["weights"].update({
        "W_SEMANTIC": round(cfg.w_semantic, 2),
        "W_EXPERIENCE": round(cfg.w_experience, 2),
        "W_COMPANY_TYPE": round(cfg.w_company, 2),
        "W_ML_SIGNALS": round(cfg.w_ml, 2),
        "W_BEHAVIORAL": round(cfg.w_behavioral, 2),
        "W_SAVED_RECRUITERS": round(cfg.w_saved, 2),
        "W_GITHUB": round(cfg.w_github, 2),
        "W_EDUCATION": round(cfg.w_education, 2),
        "W_LOCATION": round(cfg.w_location, 2),
        "W_ML_RATIO": round(cfg.w_ml_ratio, 2),
        "RRF_K": int(cfg.rrf_k),
        "RRF_WEIGHT": round(cfg.rrf_weight, 2),
    })

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def objective_score(metrics: dict[str, float], k: int, mode: str) -> float:
    ndcg = metrics[f"ndcg@{k}"]
    recall3 = metrics[f"recall@{k}_rel3"]
    if mode == "ndcg":
        return ndcg
    if mode == "recall3":
        return recall3
    return 0.6 * ndcg + 0.4 * recall3


def main():
    parser = argparse.ArgumentParser(description="Tune Stage-3 weights against proxy gold labels.")
    parser.add_argument("--gold", default="gold_labels_proxy.csv")
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--apply", action="store_true", help="Write best weights to config.yaml")
    parser.add_argument("--offline", action="store_true", help="Load models from ./models/")
    parser.add_argument("--no-mmr", action="store_true", help="Disable MMR during tuning")
    parser.add_argument(
        "--objective",
        default="composite",
        choices=["ndcg", "recall3", "composite"],
        help="Optimization target (default: 60%% NDCG + 40%% recall@K rel3)",
    )
    args = parser.parse_args()

    reload_config()

    gold_path = BASE / args.gold
    if not gold_path.exists():
        print(f"{gold_path.name} not found. Run: python build_proxy_gold.py --min-relevance 1")
        return

    gold_map = load_gold_map(gold_path)
    print(f"Loaded {len(gold_map):,} gold labels from {gold_path.name}")

    with open(BASE / "job_desc.txt", "r", encoding="utf-8") as f:
        jd_text = f.read()

    metadata, faiss_index, bm25, bm25_ids, lgb_model = load_artifacts()
    bi_enc, cross_enc = load_models(args.offline)

    base_cfg = RankingConfig(apply_mmr=not args.no_mmr)
    print("Building retrieval cache (Stages 1–2, one-time)...")
    cache = build_retrieval_cache(
        jd_text=jd_text,
        metadata=metadata,
        faiss_index=faiss_index,
        bi_enc=bi_enc,
        cross_enc=cross_enc,
        bm25=bm25,
        lgb_model=lgb_model,
        config=base_cfg,
    )

    rng = random.Random(args.seed)
    best_score = -1.0
    best_cfg = copy.deepcopy(base_cfg)
    best_metrics: dict[str, float] = {}

    baseline = finalize_ranking(
        metadata=metadata,
        cache=cache,
        bm25=bm25,
        bm25_candidate_ids=bm25_ids,
        config=base_cfg,
        include_reasoning=False,
    )
    baseline_metrics = evaluate_ranking(
        [r["candidate_id"] for r in baseline],
        gold_map,
        k=args.k,
    )
    print("\nBaseline (current config.yaml weights):")
    for key, val in baseline_metrics.items():
        print(f"  {key}: {val:.4f}")

    objective_key = args.objective
    best_score = objective_score(baseline_metrics, args.k, args.objective)
    best_metrics = baseline_metrics

    print(f"\nRunning {args.trials} random search trials (objective={args.objective})...")
    for trial in range(1, args.trials + 1):
        cfg = sample_config(rng, base_cfg)
        ranked = finalize_ranking(
            metadata=metadata,
            cache=cache,
            bm25=bm25,
            bm25_candidate_ids=bm25_ids,
            config=cfg,
            include_reasoning=False,
        )
        metrics = evaluate_ranking([r["candidate_id"] for r in ranked], gold_map, k=args.k)
        score = objective_score(metrics, args.k, args.objective)
        if score > best_score:
            best_score = score
            best_cfg = cfg
            best_metrics = metrics
            print(
                f"  trial {trial:4d}  NEW BEST {args.objective}={score:.4f}  "
                f"ndcg={metrics[f'ndcg@{args.k}']:.4f}  "
                f"recall_rel3={metrics[f'recall@{args.k}_rel3']:.4f}"
            )

    print("\n=== Best configuration ===")
    for key, val in best_metrics.items():
        print(f"  {key}: {val:.4f}")
    print("\nWeights:")
    for field in (
        "w_semantic", "w_experience", "w_company", "w_ml", "w_behavioral",
        "w_saved", "w_github", "w_education", "w_location", "w_ml_ratio",
        "rrf_k", "rrf_weight",
    ):
        print(f"  {field}: {getattr(best_cfg, field)}")

    import json
    best_path = BASE / "best_weights.json"
    weight_fields = (
        "w_semantic", "w_experience", "w_company", "w_ml", "w_behavioral",
        "w_saved", "w_github", "w_education", "w_location", "w_ml_ratio",
        "rrf_k", "rrf_weight",
    )
    payload = {"metrics": best_metrics}
    payload.update({field: getattr(best_cfg, field) for field in weight_fields})
    with open(best_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved best config to {best_path.name}")

    if args.apply:
        if best_score <= objective_score(baseline_metrics, args.k, args.objective):
            print("\nNo improvement over baseline — config.yaml unchanged.")
            print("Use: python apply_best_weights.py  (to apply a saved best_weights.json)")
        else:
            config_path = BASE / "config.yaml"
            apply_config_to_yaml(best_cfg, config_path)
            print(f"Applied best weights to {config_path.name}")
            print("Re-run: python rank.py")
    else:
        print("Dry run only. Re-run with --apply to write best weights to config.yaml")
        print("Or use: python apply_best_weights.py")


if __name__ == "__main__":
    main()
