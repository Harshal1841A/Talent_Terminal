"""
tune_weights.py — Systematic weight search for the scoring function.

Runs rank.py scoring in isolation (no model needed — uses precomputed scores)
across a grid of weight combinations, then outputs the best config.

Strategy: We don't have ground-truth labels, so we maximize the SEPARATION
between the score distribution of the top-100 vs bottom-of-pool. A good
ranker should create a clear bimodal distribution — a distinct elite tier.

Metrics we maximize:
  1. Score gap (top_10_mean - top_100_mean) — elite separation
  2. Distribution skew (more candidates in top tier = better recall)
  3. Consulting penalty effectiveness (consulting-only should be far below)
"""

import pickle
import sys
import itertools
from typing import List, Dict
from pathlib import Path
from core_scoring import score_ml_signals

sys.stdout.reconfigure(encoding='utf-8')

BASE = Path(__file__).parent

# Load the pkl (already has cross-encoder scores from last run)
print("Loading candidate_db.pkl for weight grid search...")
with open(BASE / "candidate_db.pkl", "rb") as f:
    db = pickle.load(f)

metadata = db["metadata"]
n = len(metadata)
print(f"Loaded {n:,} candidates.\n")

ce_scores = {}
try:
    import csv
    with open(BASE / "submission.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            raw = float(row.get("score", 0))
            ce_scores[row["candidate_id"]] = min(raw * 0.38, 60.0)  # approx semantic portion
except Exception:
    pass  # graceful degradation

# ── Grid search ───────────────────────────────────────────────────────────────

import math

def score_exp(years, W_EXP):
    if years <= 0: return 0.0
    peak, sigma = 7.0, 3.0
    raw = math.exp(-0.5 * ((years - peak) / sigma) ** 2)
    if years < 3: raw *= 0.3
    elif years > 15: raw *= 0.6
    return raw * W_EXP


def score_saved(saved, W_SAVED):
    if saved <= 0: return 0.0
    return min(saved / 15.0, 1.0) * W_SAVED

def score_behavioral(meta, W_BEH):
    score = 0.0
    if meta["open_to_work"]: score += 3.0
    nd = meta["notice_days"]
    if nd == 0: score += 3.0
    elif nd <= 30: score += 2.0
    elif nd <= 60: score += 0.5
    elif nd <= 90: score -= 1.0
    else: score -= 3.0
    score += (meta["response_rate"] - 0.5) * W_BEH
    lad = meta["last_active_days"]
    if 0 <= lad <= 30: score += 2.0
    elif lad > 180: score -= 2.0
    score += (meta["interview_completion"] - 0.5) * 3.0
    if meta["offer_acceptance"] >= 0:
        score += (meta["offer_acceptance"] - 0.5) * 2.0
    return score

def compute_final(meta, W_EXP, W_ML, W_COMPANY, W_SAVED, W_BEH):
    if meta["honeypot"]: return -9999
    if meta["wrong_title"]: return -500
    semantic = ce_scores.get(meta["candidate_id"], 25.0)  # non-top candidates get 25 proxy
    company = (W_COMPANY if meta["has_product_company"] else 0.0) + \
              (-25 if meta["consulting_only"] else 0.0)
    return (semantic + score_exp(meta["years_exp"], W_EXP) + company +
            score_ml_signals(meta["ml_signal_count"], W_ML) +
            score_behavioral(meta, W_BEH) +
            score_saved(meta.get("saved_by_recruiters", 0), W_SAVED))

# Grid search parameters
W_EXP_vals    = [15, 20, 25]
W_ML_vals     = [10, 15, 20]
W_COMPANY_vals = [12, 15, 18]
W_SAVED_vals  = [8, 10, 12]
W_BEH_vals    = [8, 10]

best_score = -999
best_config = None
results = []

total_combos = len(W_EXP_vals) * len(W_ML_vals) * len(W_COMPANY_vals) * len(W_SAVED_vals) * len(W_BEH_vals)
print(f"Testing {total_combos} weight combinations...\n")

from tqdm import tqdm

for W_EXP, W_ML, W_COMPANY, W_SAVED, W_BEH in tqdm(
    itertools.product(W_EXP_vals, W_ML_vals, W_COMPANY_vals, W_SAVED_vals, W_BEH_vals),
    total=total_combos
):
    scores = []
    for m in metadata:
        s = compute_final(m, W_EXP, W_ML, W_COMPANY, W_SAVED, W_BEH)
        if s > -500:
            scores.append(s)

    scores.sort(reverse=True)
    if len(scores) < 100:
        continue

    top10_mean = sum(scores[:10]) / 10
    top100_mean = sum(scores[:100]) / 100
    gap = top10_mean - top100_mean

    # Consult check: consulting candidates should score significantly below top-100
    consulting_scores = []
    for m in metadata:
        if m["consulting_only"] and not m["honeypot"] and not m["wrong_title"]:
            consulting_scores.append(
                compute_final(m, W_EXP, W_ML, W_COMPANY, W_SAVED, W_BEH)
            )
    consult_mean = sum(consulting_scores[:20]) / max(len(consulting_scores[:20]), 1)
    consult_separation = top100_mean - consult_mean

    metric = gap * 0.6 + consult_separation * 0.4

    results.append({
        "W_EXP": W_EXP, "W_ML": W_ML, "W_COMPANY": W_COMPANY,
        "W_SAVED": W_SAVED, "W_BEH": W_BEH,
        "gap": gap, "consult_sep": consult_separation, "metric": metric,
        "top10_mean": top10_mean, "top100_mean": top100_mean
    })

    if metric > best_score:
        best_score = metric
        best_config = results[-1]

results.sort(key=lambda x: -x["metric"])

print("\n" + "="*60)
print("TOP 5 WEIGHT CONFIGURATIONS")
print("="*60)
for i, r in enumerate(results[:5]):
    print(f"\n#{i+1}  Metric={r['metric']:.2f}  Gap={r['gap']:.2f}  ConsultSep={r['consult_sep']:.2f}")
    print(f"     W_EXP={r['W_EXP']}  W_ML={r['W_ML']}  W_COMPANY={r['W_COMPANY']}  W_SAVED={r['W_SAVED']}  W_BEH={r['W_BEH']}")

print("\n" + "="*60)
print("RECOMMENDED WEIGHTS (copy into rank.py):")
print("="*60)
b = best_config
print(f"W_EXPERIENCE       = {b['W_EXP']}")
print(f"W_ML_SIGNALS       = {b['W_ML']}")
print(f"W_COMPANY_TYPE     = {b['W_COMPANY']}")
print(f"W_SAVED_RECRUITERS = {b['W_SAVED']}")
print(f"W_BEHAVIORAL       = {b['W_BEH']}")
