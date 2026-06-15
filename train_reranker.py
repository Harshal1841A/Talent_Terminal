"""
train_reranker.py — Train a LightGBM reranker on saved_by_recruiters labels.

This is the single biggest edge in the competition:
- Most submissions use hand-tuned heuristic weights.
- Ours learns weights from actual recruiter engagement data.

Run this ONCE after precompute.py:
    python train_reranker.py

Produces: lgbm_reranker.pkl (used at scoring time in rank.py / app.py)

Strategy:
  saved_by_recruiters_30d is the closest proxy to ground truth in this dataset.
  We use log(1 + saves) as the label to compress the heavy right tail.
  Features = all precomputed metadata (28 features).
  Model = LightGBM (fast, handles mixed types, interpretable via feature importances).
  Output = a [0, 1] relevance score used to scale the heuristic component by up to 30%.
"""

import pickle
import sys
import math
import numpy as np
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

BASE = Path(__file__).parent

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

print("Loading candidate_db.pkl...")
with open(BASE / "candidate_db.pkl", "rb") as f:
    db = pickle.load(f)

metadata = db["metadata"]
n_total = len(metadata)
print(f"Loaded {n_total:,} candidates.")

# Filter out deterministic disqualifiers — don't train on garbage
eligible = [m for m in metadata if not m.get("honeypot") and not m.get("wrong_title")]
n_eligible = len(eligible)
print(f"Eligible (non-honeypot, non-wrong-title): {n_eligible:,}")


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE BUILDER
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_NAMES = [
    "years_exp",
    "ml_signal_count",
    "jd_term_bonus",
    "elite_company_bonus",
    "has_product_company",
    "consulting_only",
    "github_score",
    "core_skill_score",
    "avg_assessment",
    "edu_tier_1",
    "has_external_validation",
    "response_rate",
    "notice_days_norm",      # normalized to [0, 1] — 0 days = 1.0, 180+ days = 0.0
    "open_to_work",
    "last_active_norm",      # normalized — 0 days ago = 1.0, 365+ = 0.0
    "interview_completion",
    "offer_acceptance",
    "profile_completeness",
    "applications_submitted",
    "search_appearance_30d",
    "avg_response_time_norm",  # normalized — 1h = 1.0, 999h = 0.0
    "endorsements_received",
    "linkedin_connected",
    "verified_email",
    "verified_phone",
    "research_founding_score",
    "skill_count_norm",       # normalized to [0, 1] by capping at 100 skills
    "willing_to_relocate",
]


def build_features(meta: dict) -> list:
    years_exp       = float(meta.get("years_exp", 0) or 0)
    ml_signal       = float(meta.get("ml_signal_count", 0) or 0)
    jd_bonus        = float(meta.get("jd_term_bonus", 0.0) or 0.0)
    elite_bonus     = float(meta.get("elite_company_bonus", 0.0) or 0.0)
    product_co      = 1.0 if meta.get("has_product_company") else 0.0
    consulting      = 1.0 if meta.get("consulting_only") else 0.0
    github          = float(meta.get("github_score", -1) or -1)
    github          = max(github, 0.0) / 100.0  # normalize to [0, 1]
    core_skill      = float(meta.get("core_skill_score", -1) or -1)
    core_skill      = max(core_skill, 0.0) / 100.0
    avg_assess      = float(meta.get("avg_assessment", -1) or -1)
    avg_assess      = max(avg_assess, 0.0) / 100.0
    edu_tier        = 1.0 if meta.get("edu_tier_1") else 0.0
    has_pub         = 1.0 if meta.get("has_external_validation") else 0.0
    resp_rate       = float(meta.get("response_rate", 0.5) or 0.5)
    notice          = float(meta.get("notice_days", 90) or 90)
    notice_norm     = max(0.0, 1.0 - notice / 180.0)  # 0 days = 1.0, 180+ = 0.0
    open_work       = 1.0 if meta.get("open_to_work") else 0.0
    last_active     = float(meta.get("last_active_days", 365) or 365)
    last_active_n   = max(0.0, 1.0 - last_active / 365.0)
    interview_comp  = float(meta.get("interview_completion", 0.5) or 0.5)
    offer_acc       = float(meta.get("offer_acceptance", -1) or -1)
    offer_acc       = max(offer_acc, 0.0) if offer_acc >= 0 else 0.5
    profile_pct     = float(meta.get("profile_completeness", 50) or 50) / 100.0
    apps_submitted  = float(meta.get("applications_submitted", 0) or 0)
    search_appear   = float(meta.get("search_appearance_30d", 0) or 0)
    resp_time       = float(meta.get("avg_response_time_hours", 999) or 999)
    resp_time_n     = max(0.0, 1.0 - resp_time / 999.0)
    endorsements    = float(meta.get("endorsements_received", 0) or 0)
    linkedin        = 1.0 if meta.get("linkedin_connected") else 0.0
    email_ver       = 1.0 if meta.get("verified_email") else 0.0
    phone_ver       = 1.0 if meta.get("verified_phone") else 0.0
    research        = float(meta.get("research_founding_score", 0) or 0)
    skill_count     = float(meta.get("skill_count", 0) or 0) / 100.0
    relocate        = 1.0 if meta.get("willing_to_relocate") else 0.0

    return [
        years_exp, ml_signal, jd_bonus, elite_bonus,
        product_co, consulting, github, core_skill, avg_assess,
        edu_tier, has_pub, resp_rate, notice_norm, open_work,
        last_active_n, interview_comp, offer_acc, profile_pct,
        apps_submitted, search_appear, resp_time_n, endorsements,
        linkedin, email_ver, phone_ver, research, skill_count, relocate,
    ]


# ─────────────────────────────────────────────────────────────────────────────
# BUILD TRAINING DATA
# ─────────────────────────────────────────────────────────────────────────────

print("Building feature matrix...")
X = np.array([build_features(m) for m in eligible], dtype=np.float32)
saves_raw = np.array(
    [float(m.get("saved_by_recruiters", 0) or 0) for m in eligible],
    dtype=np.float64
)
# Label: log(1 + saves) normalized to [0, 1]
# log compression handles heavy right tail (candidates with 50+ saves dominate otherwise)
log_saves = np.log1p(saves_raw)
y = log_saves / (log_saves.max() + 1e-9)

print(f"  Feature matrix shape: {X.shape}")
print(f"  Label range: [{y.min():.4f}, {y.max():.4f}]")
print(f"  Candidates with saves > 0: {(saves_raw > 0).sum():,} ({(saves_raw > 0).mean()*100:.1f}%)")
print(f"  Candidates with saves >= 16: {(saves_raw >= 16).sum():,} ({(saves_raw >= 16).mean()*100:.1f}%)")


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN LIGHTGBM
# ─────────────────────────────────────────────────────────────────────────────

try:
    import lightgbm as lgb
except ImportError:
    print("\nLightGBM not installed. Run: pip install lightgbm")
    sys.exit(1)

print("\nTraining LightGBM reranker...")
model = lgb.LGBMRegressor(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=6,
    num_leaves=63,
    min_child_samples=30,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)
model.fit(X, y)

print("Training complete.")

# ─────────────────────────────────────────────────────────────────────────────
# EVALUATE (pseudo — no held-out labels, use correlation with saves)
# ─────────────────────────────────────────────────────────────────────────────

preds = model.predict(X)
correlation = np.corrcoef(preds, y)[0, 1]
print(f"\nTrain correlation with log(1+saves): {correlation:.4f}")
print("(>0.5 = good, >0.7 = excellent for a proxy label like recruiter saves)")

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE IMPORTANCES
# ─────────────────────────────────────────────────────────────────────────────

print("\nFeature Importances (by gain):")
importances = model.feature_importances_
for name, imp in sorted(zip(FEATURE_NAMES, importances), key=lambda x: -x[1]):
    bar = "█" * int(imp / max(importances) * 30)
    print(f"  {name:<30} {bar}  ({imp:.1f})")

# ─────────────────────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────────────────────

import joblib
output_path = BASE / "lgbm_reranker.pkl"
joblib.dump({"model": model, "feature_names": FEATURE_NAMES}, output_path)
print(f"\n✓ Saved: {output_path}")
print("Now update rank.py and app.py to use this model (already wired in if you ran the full plan).")
print("Then run: python rank.py")
