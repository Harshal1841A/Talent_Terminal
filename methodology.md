# Ranking Methodology

## 1. Pipeline overview

Three-stage funnel shared by `rank.py` (submission) and `app.py` (demo) via `ranking_pipeline.py`:

1. **Stage 1 — Hybrid retrieval:** `BAAI/bge-base-en-v1.5` dense search (top 800) plus optional BM25 over 100K profiles. JD query expansion phrases are applied **only** when the input JD matches the bundled `job_desc.txt` (challenge mode).
2. **Stage 2 — Cross-encoder:** `ms-marco-MiniLM-L-6-v2` (or optional fine-tuned CE in `./models/finetuned-ce-model`) re-scores the top 500 dense hits.
3. **Stage 3 — Heuristic fusion:** Calibrated semantic score + experience, company type, ML signals, ML-role ratio, location, recency, JD-term bonus, elite-company bonus, behavioral signals, and penalties. Reciprocal Rank Fusion (RRF) blends dense, CE, and BM25 ranks. Optional MMR diversifies notice-period buckets in the final top 100.

## 2. Hard gates and penalties

| Signal | Effect |
|--------|--------|
| Honeypot (impossible credentials / timeline) | Score = −9999 |
| Wrong-title (marketing, HR, robotics-only, etc.) | Score = −500 + small semantic |
| Consulting-only career | −25 company penalty |
| Currently at consulting/services firm | −18 (e.g. Genpact, TCS) |
| Keyword stuffer (80+ skills, &lt;3 yr exp) | −45 |
| Notice &gt; 90 days | −18 (behavioral handles ≤90d) |
| International, no relocate | −28 |
| Offsite India, no relocate | −14 (location reward only for loc &gt; 1.5) |
| Title-chaser pattern | −12 |

`jd_term_bonus` (0–20) and `elite_company_bonus` (0 or 5) from precompute are added directly to the final score.

## 3. Evaluation

- **Proxy labels:** `build_proxy_gold.py` → `gold_labels_proxy.csv` (rule-based, useful for tuning — not a substitute for hidden judge labels).
- **Metrics:** `self_eval.py` reports NDCG@K, MAP, MRR, recall@K.
- **Manual review:** `sample_for_labeling.py` / `labeling_review_priority.csv` for human labels merged via `merge_manual_labels.py`.
- **Weight tuning:** `tune_weights.py` random-searches Stage-3 weights; `apply_best_weights.py` writes winners to `config.yaml`.

We do **not** claim fixed ablation percentages in docs — run `tune_weights.py` with components disabled to measure impact on your labels.

## 4. Case-study patterns (qualitative)

- **Keyword stuffer:** High BM25/semantic overlap but penalized −45 in Stage 3; often also caught by honeypot rules in precompute.
- **Domain pivot:** Low ML-role ratio suppresses candidates with many non-ML years despite “ML” keywords.
- **Quiet operator:** Needs to survive Stage-1 top 800; cross-encoder and JD-term bonus help surface depth over keyword lists.

## 5. Fairness notes

- Tier-1 education is a small additive weight (≤ ~0.75 pts).
- Names are not a separate scoring feature; embeddings use headline, summary, and career text only.
- Location scoring follows the JD’s Pune/Noida preference and relocation rules.

## 6. Submission validation

`rank.py` auto-runs `validate_submission.py` after writing the CSV (100 rows, monotonic scores, candidate_id tie-break).
