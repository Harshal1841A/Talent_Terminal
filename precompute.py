"""
precompute.py — Offline Embedding & Feature Extraction
Run this ONCE before rank.py. Takes ~30-40 min on CPU.
Produces candidate_db.pkl with 768-dim embeddings + rich feature metadata.
"""

import json
import pickle
import re
import math
from datetime import date, datetime
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

BASE = Path(__file__).parent

import yaml
import faiss
import numpy as np
from schema import CandidateMetadata

# Load configuration
with open(BASE / "config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

heuristics = config["heuristics"]
ML_ROLE_TITLES = heuristics["ml_role_titles"]
SENIORITY_RANK = heuristics["seniority_rank"]
CONSULTING_FIRMS = heuristics["consulting_firms"]
WRONG_TITLE_SIGNALS = heuristics["wrong_title_signals"]
ML_SIGNALS_ELITE = heuristics["ml_signals_elite"]
ML_SIGNALS_STRONG = heuristics["ml_signals_strong"]
ML_SIGNALS_WEAK = heuristics["ml_signals_weak"]
PRODUCTION_ML_SIGNALS = ML_SIGNALS_ELITE + ML_SIGNALS_STRONG + ML_SIGNALS_WEAK
JD_EXACT_MATCH_TERMS = heuristics["jd_exact_match_terms"]
PUBLICATION_SIGNALS = heuristics["publication_signals"]
ELITE_TECH_COMPANIES = heuristics["elite_tech_companies"]

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def is_consulting(company_name: str) -> bool:
    name = company_name.lower().strip()
    return any(firm in name for firm in CONSULTING_FIRMS)

def detect_wrong_title(titles: list[str]) -> bool:
    """
    Returns True if the candidate's dominant titles suggest irrelevant domain.
    T2-B FIX: one past stint in a wrong-title role is NOT disqualifying.
    Requires current_title to match OR 2+ past titles to match.
    Call as: detect_wrong_title([current_title] + all_past_titles)
    """
    if not titles:
        return False
    # Current title (first element) — strict check
    current = titles[0].lower()
    if any(sig in current for sig in WRONG_TITLE_SIGNALS):
        return True
    # Past titles — only flag if 2+ roles match (prevents false positives from early-career stints)
    past_matches = sum(
        1 for t in titles[1:]
        if any(sig in t.lower() for sig in WRONG_TITLE_SIGNALS)
    )
    return past_matches >= 2


def get_seniority(title: str) -> int:
    """Map a job title to an integer seniority level (0=junior, 5=VP)."""
    t = title.lower()
    for kw, rank in sorted(SENIORITY_RANK.items(), key=lambda x: -x[1]):
        if kw and kw in t:
            return rank
    return 1  # base level

def get_recency_weight(years_ago: float) -> float:
    if years_ago <= 1.0: return 1.0
    if years_ago >= 5.0: return 0.1
    if years_ago <= 3.0:
        return 1.0 - 0.25 * (years_ago - 1.0)
    else:
        return 0.5 - 0.2 * (years_ago - 3.0)

def count_recency_weighted_ml_signals(career: list, summary: str) -> float:
    """
    T1-B: Tiered ML signal counting.
    - Elite signals (e.g. FAISS, HNSW, two-tower): weight 3.0×
    - Strong signals (e.g. production deployment, a/b test): weight 1.5×
    - Weak/noisy signals (e.g. transformer, llm): weight 0.3×
    All recency-weighted. Sigmoid-transformed to [0, 1].
    """
    total_score = 0.0
    today = date.today()
    summary_lower = summary.lower()

    # Score summary text (no recency — static document)
    for sig in ML_SIGNALS_ELITE:
        if re.search(r'\b' + re.escape(sig) + r'\b', summary_lower):
            total_score += 3.0
    for sig in ML_SIGNALS_STRONG:
        if re.search(r'\b' + re.escape(sig) + r'\b', summary_lower):
            total_score += 1.5
    for sig in ML_SIGNALS_WEAK:
        if re.search(r'\b' + re.escape(sig) + r'\b', summary_lower):
            total_score += 0.3

    # Score career history with recency weighting
    for exp in career:
        desc  = (exp.get("description", "") or "").lower()
        title = (exp.get("title", "") or "").lower()
        combined = desc + " " + title

        # Compute recency weight for this experience
        end_str = exp.get("end_date")
        years_ago = 0.0
        if end_str:
            try:
                d = datetime.strptime(end_str[:10], "%Y-%m-%d").date()
                years_ago = max(0.0, (today - d).days / 365.25)
            except (ValueError, TypeError):
                years_ago = 0.0
        rw = get_recency_weight(years_ago)

        for sig in ML_SIGNALS_ELITE:
            if re.search(r'\b' + re.escape(sig) + r'\b', combined):
                total_score += 3.0 * rw
        for sig in ML_SIGNALS_STRONG:
            if re.search(r'\b' + re.escape(sig) + r'\b', combined):
                total_score += 1.5 * rw
        for sig in ML_SIGNALS_WEAK:
            if re.search(r'\b' + re.escape(sig) + r'\b', combined):
                total_score += 0.3 * rw

    # BUG-10 FIX: zero signals must return exactly 0.0 — not 0.27
    if total_score == 0:
        return 0.0
    # Sigmoid re-centered at 5 to account for higher possible scores from 3-tier weights
    return float(1 / (1 + math.exp(-0.4 * (total_score - 5))))

def days_since(date_str: str | None) -> int:
    """Return days since the given date string (YYYY-MM-DD). -1 if missing."""
    if not date_str:
        return -1
    try:
        d = datetime.strptime(date_str[:10], "%Y-%m-%d").date()
        return (date.today() - d).days
    except (ValueError, TypeError):
        return -1

def build_weighted_doc(candidate: dict) -> str:
    """
    Build a semantic document weighted so that career descriptions (the richest
    signal) count 3× more than headlines/summaries.
    """
    profile = candidate.get("profile", {})
    headline = profile.get("headline", "")
    summary = profile.get("summary", "")

    career_parts = []
    for exp in candidate.get("career_history", []):
        title = exp.get("title", "")
        company = exp.get("company", "")
        desc = exp.get("description", "")
        career_parts.append(f"Title: {title} at {company}. {desc}")

    # Skills section (1×)
    skills_text = ", ".join(
        s.get("name", "") for s in candidate.get("skills", [])
        if s.get("proficiency") in ("advanced", "expert")
    )

    # Education (1×)
    edu_parts = []
    for edu in candidate.get("education", []):
        field = edu.get("field_of_study", "")
        degree = edu.get("degree", "")
        if field or degree:
            edu_parts.append(f"{degree} in {field}")

    doc = " | ".join([
        f"Headline: {headline}",
        f"Summary: {summary}",
        " | ".join(career_parts),
        f"Expert/Advanced Skills: {skills_text}",
        f"Education: {', '.join(edu_parts)}",
    ])
    return doc


def extract_features(candidate: dict) -> dict:
    """Extract all structured features needed for rule-based scoring."""
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])

    # ── Honeypot detection (Tier 2.1 — multi-signal) ────────────────────────
    # Signal 1: expert skills claimed with 0 months duration
    expert_zero_count = sum(
        1 for s in skills
        if s.get("proficiency") == "expert" and (s.get("duration_months") or 0) == 0
    )
    # Signal 2: timeline impossibilities in career history
    DATASET_REF_DATE = date(2025, 6, 1)  # approximate dataset snapshot date
    timeline_impossible = False
    for exp in career:
        start_str = exp.get("start_date")
        end_str = exp.get("end_date")
        dur = exp.get("duration_months") or 0
        try:
            if start_str:
                s_d = datetime.strptime(start_str[:10], "%Y-%m-%d").date()
                if s_d > DATASET_REF_DATE:
                    timeline_impossible = True
                    break
            if start_str and end_str:
                s_d = datetime.strptime(start_str[:10], "%Y-%m-%d").date()
                e_d = datetime.strptime(end_str[:10], "%Y-%m-%d").date()
                if e_d < s_d:  # end before start
                    timeline_impossible = True
                    break
                actual_months = (e_d.year - s_d.year) * 12 + (e_d.month - s_d.month)
                if dur > actual_months + 18:  # claimed >> actual span by >18 mo
                    timeline_impossible = True
                    break
        except (ValueError, TypeError):
            pass

    honeypot = (
        expert_zero_count >= 3        # 3+ expert skills with 0 duration
        or (expert_zero_count >= 1 and timeline_impossible)  # any timeline issue + expert-zero
        or timeline_impossible         # timeline impossibility alone is sufficient
    )

    # ── Experience years (use direct field — it's in the schema) ──────────
    years_exp = profile.get("years_of_experience", 0) or 0

    # ── Company type analysis ─────────────────────────────────────────────
    # BUG-05 FIX: filter blank company strings — empty string is not consulting
    # but also not a product company. Blank entries (freelance/self-employed)
    # were incorrectly granting has_product_company = True before this fix.
    companies = [exp.get("company", "").strip() for exp in career if exp.get("company", "").strip()]
    consulting_flags = [is_consulting(c) for c in companies]
    has_product_company = any(not f for f in consulting_flags)
    consulting_only = all(consulting_flags) and len(companies) > 0

    # ── Title check ───────────────────────────────────────────────────────
    current_title = profile.get("current_title", "") or ""
    # BUG-06 FIX: check full career history, not just current title.
    # A career marketer who recently changed title to 'ML Engineer' was slipping through.
    all_career_titles = [exp.get("title", "") or "" for exp in career]
    wrong_title = detect_wrong_title([current_title] + all_career_titles)

    # ── Production ML signals (tiered) ───────────────────────────────────────
    ml_signal_count = count_recency_weighted_ml_signals(career, profile.get("summary", "") or "")

    # ── T1-C: JD-specific exact-match bonus ──────────────────────────────────
    all_text_lower = (
        (profile.get("summary", "") or "") + " " +
        " ".join(
            (exp.get("description", "") or "") + " " + (exp.get("title", "") or "")
            for exp in career
        )
    ).lower()
    jd_term_bonus = 0.0
    for term, pts in JD_EXACT_MATCH_TERMS.items():
        if re.search(r'\b' + re.escape(term) + r'\b', all_text_lower):
            jd_term_bonus += pts
    jd_term_bonus = min(jd_term_bonus, 20.0)  # hard cap at 20 pts

    # ── External validation signal ────────────────────────────────────────────
    has_external_validation = any(re.search(r'\b' + re.escape(sig) + r'\b', all_text_lower) for sig in PUBLICATION_SIGNALS)

    # ── Elite company signal ──────────────────────────────────────────────────
    companies_lower = [(exp.get("company", "") or "").lower() for exp in career]
    elite_company_bonus = 5.0 if any(
        elite in company
        for company in companies_lower
        for elite in ELITE_TECH_COMPANIES
    ) else 0.0

    # ── Research/Founding DNA (Tier 1.3 fix) ─────────────────────────────
    research_founding_score = 0.0
    all_titles_lower = [(exp.get("title", "") or "").lower() for exp in career]
    is_intern_only = len(all_titles_lower) > 0 and all(
        "intern" in t or "student" in t for t in all_titles_lower
    )
    if is_intern_only:
        research_founding_score = -5.0
    else:
        for t in all_titles_lower:
            if any(k in t for k in ["founder", "founding engineer", "cto"]):
                research_founding_score = 15.0
                break
            elif any(k in t for k in ["research scientist", "applied scientist"]):
                if ml_signal_count >= 0.3:
                    research_founding_score = 15.0
                elif ml_signal_count >= 0.1:
                    research_founding_score = 5.0   # partial credit — uncertain
                else:
                    research_founding_score = 0.0   # pure research, no production signals
                break

    # ── Education tier ───────────────────────────────────────────────────────
    edu_tier_1 = any(
        e.get("tier") == "tier_1" for e in candidate.get("education", [])
    )

    # ── Location score (Tier 1.1) ─────────────────────────────────────────────
    location_raw = (profile.get("location", "") or "").lower()
    location_country = (profile.get("country", "") or "").lower()

    PUNE_NOIDA = ["pune", "noida"]
    PREFERRED_CITIES = [
        "hyderabad", "mumbai", "delhi", "gurugram", "gurgaon",
        "bengaluru", "bangalore", "chennai", "kolkata",
    ]

    if any(c in location_raw for c in PUNE_NOIDA):
        location_score = 5.0
        location_score_city = next(c for c in PUNE_NOIDA if c in location_raw)
    elif any(c in location_raw for c in PREFERRED_CITIES):
        location_score = 3.0
        location_score_city = next(c for c in PREFERRED_CITIES if c in location_raw)
    elif "india" in location_country or "india" in location_raw:
        willing_to_relocate = signals.get("willing_to_relocate", False)
        location_score = 3.0 if willing_to_relocate else 1.5
        location_score_city = "india"
    else:
        willing_to_relocate = signals.get("willing_to_relocate", False)
        location_score = 1.0 if willing_to_relocate else 0.0
        location_score_city = "international"

    # ── ML-role-years ratio (Tier 1.2) ─────────────────────────────────────────
    ml_role_months = sum(
        (exp.get("duration_months") or 0)
        for exp in career
        if any(sig in (exp.get("title", "") or "").lower() for sig in ML_ROLE_TITLES)
    )
    total_months = sum((exp.get("duration_months") or 0) for exp in career)
    ml_role_ratio = ml_role_months / max(total_months, 1)

    # ── Title-chaser detection (Tier 1.4) ────────────────────────────────────
    non_current_jobs = [e for e in career if not e.get("is_current")]
    avg_tenure_months = (
        sum((e.get("duration_months") or 0) for e in non_current_jobs) / len(non_current_jobs)
        if non_current_jobs else 999.0
    )
    seniority_levels = [get_seniority(e.get("title", "")) for e in career]
    monotone_escalating = (
        len(seniority_levels) >= 3 and
        all(seniority_levels[i] >= seniority_levels[i + 1] for i in range(len(seniority_levels) - 1)) and
        seniority_levels[-1] < seniority_levels[0]
    )
    title_chaser = bool(avg_tenure_months < 18 and monotone_escalating)

    # ── Redrob behavioral signals ─────────────────────────────────────────
    response_rate = signals.get("recruiter_response_rate", 0.5)
    notice_days = signals.get("notice_period_days", 60)
    open_to_work = signals.get("open_to_work_flag", False)
    github_score = signals.get("github_activity_score", -1)
    last_active_days = days_since(signals.get("last_active_date"))
    interview_completion = signals.get("interview_completion_rate", 0.5)
    offer_acceptance = signals.get("offer_acceptance_rate", -1)
    profile_completeness = signals.get("profile_completeness_score", 50)
    # BUG-15 FIX: .get() returns None when key exists with null value.
    # Use `or {}` to handle both missing key and null value safely.
    skill_assessments = signals.get("skill_assessment_scores") or {}
    avg_assessment = (
        sum(skill_assessments.values()) / len(skill_assessments)
        if skill_assessments else -1
    )
    willing_to_relocate = signals.get("willing_to_relocate", False)
    preferred_work_mode = signals.get("preferred_work_mode", "flexible")
    saved_by_recruiters = signals.get("saved_by_recruiters_30d", 0)
    applications_submitted = signals.get("applications_submitted_30d", 0)
    search_appearance_30d = signals.get("search_appearance_30d", 0)
    avg_response_time_hours = signals.get("avg_response_time_hours", 999)
    endorsements_received = signals.get("endorsements_received", 0)
    connection_count = signals.get("connection_count", 0)
    linkedin_connected = signals.get("linkedin_connected", False)
    verified_email = signals.get("verified_email", False)
    verified_phone = signals.get("verified_phone", False)

    # ── Skill assessment for core skills ──────────────────────────────────
    core_skill_score = -1
    for skill_name, score in skill_assessments.items():
        skill_lower = skill_name.lower()
        if any(kw in skill_lower for kw in ["python", "ml", "machine learning", "nlp", "retrieval", "ranking"]):
            core_skill_score = max(core_skill_score, score)

    raw_meta = {
        "candidate_id": candidate["candidate_id"],
        "current_title": current_title,
        "current_company": profile.get("current_company", ""),
        "years_exp": years_exp,
        "honeypot": honeypot,
        "has_product_company": has_product_company,
        "consulting_only": consulting_only,
        "wrong_title": wrong_title,
        "ml_signal_count": ml_signal_count,
        "jd_term_bonus": jd_term_bonus,               # NEW: T1-C
        "has_external_validation": has_external_validation,  # NEW: T2-D
        "elite_company_bonus": elite_company_bonus,    # NEW: T2-C
        "edu_tier_1": edu_tier_1,
        "response_rate": response_rate,
        "notice_days": notice_days,
        "open_to_work": open_to_work,
        "github_score": github_score,
        "last_active_days": last_active_days,
        "interview_completion": interview_completion,
        "offer_acceptance": offer_acceptance,
        "profile_completeness": profile_completeness,
        "avg_assessment": avg_assessment,
        "core_skill_score": core_skill_score,
        "willing_to_relocate": willing_to_relocate,
        "preferred_work_mode": preferred_work_mode,
        "saved_by_recruiters": saved_by_recruiters,
        "applications_submitted": applications_submitted,
        "search_appearance_30d": search_appearance_30d,
        "avg_response_time_hours": avg_response_time_hours,
        "endorsements_received": endorsements_received,
        "connection_count": connection_count,
        "linkedin_connected": linkedin_connected,
        "verified_email": verified_email,
        "verified_phone": verified_phone,
        "research_founding_score": research_founding_score,
        "skill_count": len(skills),
        "location_score": location_score,          # Tier 1.1
        "location_score_city": location_score_city,
        "ml_role_ratio": ml_role_ratio,            # Tier 1.2
        "ml_role_months": ml_role_months,
        "total_months": total_months,
        "title_chaser": title_chaser,              # Tier 1.4
        "avg_tenure_months": avg_tenure_months,
    }
    
    return CandidateMetadata(**raw_meta).model_dump()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("Loading SentenceTransformer model from ./models/ ...")
    model = SentenceTransformer(str(BASE / "models" / "bge-base-en-v1.5"))

    all_meta = []
    all_docs = []

    print("Reading and processing candidates.jsonl...")
    with open(BASE / "candidates.jsonl", "r", encoding="utf-8") as f:
        lines = f.readlines()

    print(f"Loaded {len(lines):,} candidates. Extracting features...")
    for line in tqdm(lines, desc="Feature extraction"):
        c = json.loads(line)
        meta = extract_features(c)
        doc = build_weighted_doc(c)
        meta["doc_text"] = doc
        all_meta.append(meta)
        all_docs.append(doc)

    print(f"\nComputing {len(all_docs):,} embeddings (batch_size=16)...")
    embeddings = model.encode(
        all_docs,
        batch_size=16,
        show_progress_bar=True,
        convert_to_tensor=True,
        normalize_embeddings=True,   # pre-normalize → cosine sim = dot product (fast)
    )

    print("Building FAISS index...")
    emb_np = embeddings.cpu().numpy()
    d = emb_np.shape[1]
    index = faiss.IndexFlatIP(d)  # Inner product since embeddings are normalized (cosine similarity)
    index.add(emb_np)

    print("Saving faiss_index.bin and candidate_meta.pkl...")
    faiss.write_index(index, str(BASE / "faiss_index.bin"))
    with open(BASE / "candidate_meta.pkl", "wb") as f:
        pickle.dump(all_meta, f, protocol=pickle.HIGHEST_PROTOCOL)

    honeypots = sum(1 for m in all_meta if m["honeypot"])
    wrong = sum(1 for m in all_meta if m["wrong_title"])
    consulting = sum(1 for m in all_meta if m["consulting_only"])
    print(f"\n✓ Done. faiss_index.bin and candidate_meta.pkl saved.")
    print(f"  Honeypots detected : {honeypots:,}")
    print(f"  Wrong-title flags  : {wrong:,}")
    print(f"  Consulting-only    : {consulting:,}")
    print(f"\nNow run: python rank.py")


if __name__ == "__main__":
    main()
