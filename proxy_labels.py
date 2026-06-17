"""
JD-aligned proxy relevance labels (0–3) for offline evaluation and weight tuning.

These encode what the Redrob JD *means*, not keyword overlap.
Refine with sample_for_labeling.py + manual edits to gold_labels_proxy.csv.
"""

from __future__ import annotations


def compute_proxy_relevance(meta: dict) -> int:
    """
    Return relevance grade:
      3 = would interview (ideal profile)
      2 = strong maybe
      1 = weak / wrong band
      0 = hard no
    """
    if meta.get("honeypot") or meta.get("wrong_title"):
        return 0

    years = float(meta.get("years_exp", 0) or 0)
    ml_ratio = float(meta.get("ml_role_ratio", 0) or 0)
    ml_signals = float(meta.get("ml_signal_count", 0) or 0)
    loc = float(meta.get("location_score", 0) or 0)
    response_rate = float(meta.get("response_rate", 0.5) or 0.5)
    last_active = int(meta.get("last_active_days", 365) or 365)
    notice = int(meta.get("notice_days", 60) or 60)
    research = float(meta.get("research_founding_score", 0) or 0)
    skill_count = int(meta.get("skill_count", 0) or 0)

    score = 0.0

    # Hard negatives
    if meta.get("consulting_only"):
        score -= 3.0
    if meta.get("title_chaser"):
        score -= 2.0
    if skill_count > 80 and years < 3:
        score -= 3.0
    if research == -5.0:
        score -= 2.0
    if loc == 0.0 and not meta.get("willing_to_relocate"):
        score -= 4.0
        return 0
    if loc <= 1.5 and not meta.get("willing_to_relocate"):
        score -= 1.5
    if response_rate < 0.2 or last_active > 180:
        score -= 2.0
    if years < 3:
        score -= 2.0
    elif years > 14:
        score -= 1.5
    if ml_ratio < 0.15 and years > 3:
        score -= 2.5
    elif ml_ratio < 0.35:
        score -= 1.0

    # Positives — ideal candidate from JD closing paragraph
    if meta.get("has_product_company"):
        score += 1.5
    if 5 <= years <= 9:
        score += 2.0
    elif 4 <= years < 5 or 9 < years <= 11:
        score += 0.75
    if ml_ratio >= 0.65:
        score += 2.0
    elif ml_ratio >= 0.45:
        score += 1.0
    if ml_signals >= 0.7:
        score += 1.5
    elif ml_signals >= 0.4:
        score += 0.75
    if loc >= 4.0:
        score += 1.0
    elif loc >= 2.5 or meta.get("willing_to_relocate"):
        score += 0.5
    if meta.get("open_to_work"):
        score += 0.5
    if notice <= 30:
        score += 0.5
    elif notice > 90:
        score -= 2.0
    if response_rate >= 0.6 and last_active <= 60:
        score += 1.0
    if research == 15.0 and ml_signals >= 0.3:
        score += 1.0
    if (meta.get("saved_by_recruiters", 0) or 0) >= 10:
        score += 0.5

    if score >= 7.0:
        rel = 3
    elif score >= 4.0:
        rel = 2
    elif score >= 1.0:
        rel = 1
    else:
        rel = 0

    if rel >= 3 and loc <= 1.5 and not meta.get("willing_to_relocate"):
        rel = 2
    if rel >= 2 and notice > 90:
        rel = min(rel, 1)
    return rel


def bucket_label(meta: dict) -> str:
    """Stratification bucket for sampling candidates to review."""
    if meta.get("honeypot"):
        return "honeypot"
    if meta.get("wrong_title"):
        return "wrong_title"
    if meta.get("consulting_only"):
        return "consulting_only"
    if (meta.get("skill_count", 0) or 0) > 80 and (meta.get("years_exp", 0) or 0) < 3:
        return "keyword_stuffer"
    ml_ratio = float(meta.get("ml_role_ratio", 0) or 0)
    years = float(meta.get("years_exp", 0) or 0)
    if years >= 5 and ml_ratio < 0.25:
        return "domain_pivot"
    if compute_proxy_relevance(meta) >= 3:
        return "obvious_positive"
    if compute_proxy_relevance(meta) == 0:
        return "obvious_negative"
    return "borderline"
