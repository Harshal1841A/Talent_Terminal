import math
from pathlib import Path
import yaml

BASE = Path(__file__).parent

_config: dict = {}
CONSULTING_FIRMS: list[str] = []

W_SEMANTIC = 0.0
W_EXPERIENCE = 0.0
W_COMPANY_TYPE = 0.0
W_ML_SIGNALS = 0.0
W_BEHAVIORAL = 0.0
W_SAVED_RECRUITERS = 0.0
W_GITHUB = 0.0
W_ASSESSMENT = 0.0
W_PROFILE_COMPLETE = 0.0
W_EDUCATION = 0.0
W_RECENCY = 0.0
W_ENGAGEMENT = 0.0
W_TRUST = 0.0
W_LOCATION = 0.0
W_ML_RATIO = 0.0
RRF_K = 50.0
RRF_WEIGHT = 0.0

HONEYPOT_PENALTY = 0.0
WRONG_TITLE_PENALTY = 0.0
CONSULTING_ONLY_PENALTY = 0.0
TITLE_CHASER_PENALTY = 0.0
LONG_NOTICE_PENALTY = 0.0
INTERNATIONAL_PENALTY = 0.0
OFFSITE_LOCATION_PENALTY = 0.0
KEYWORD_STUFFER_PENALTY = 0.0
CURRENT_CONSULTING_PENALTY = 0.0

LGBM_SCORE_MULTIPLIER = 0.15


def reload_config(config_path: Path | None = None) -> None:
    """Reload weights/penalties from config.yaml (call after apply_best_weights)."""
    global _config, CONSULTING_FIRMS
    global W_SEMANTIC, W_EXPERIENCE, W_COMPANY_TYPE, W_ML_SIGNALS, W_BEHAVIORAL
    global W_SAVED_RECRUITERS, W_GITHUB, W_ASSESSMENT, W_PROFILE_COMPLETE
    global W_EDUCATION, W_RECENCY, W_ENGAGEMENT, W_TRUST, W_LOCATION, W_ML_RATIO
    global RRF_K, RRF_WEIGHT
    global HONEYPOT_PENALTY, WRONG_TITLE_PENALTY, CONSULTING_ONLY_PENALTY
    global TITLE_CHASER_PENALTY, LONG_NOTICE_PENALTY, INTERNATIONAL_PENALTY
    global OFFSITE_LOCATION_PENALTY, KEYWORD_STUFFER_PENALTY, CURRENT_CONSULTING_PENALTY
    global LGBM_SCORE_MULTIPLIER

    path = config_path or (BASE / "config.yaml")
    with open(path, "r", encoding="utf-8") as f:
        _config = yaml.safe_load(f)

    weights = _config["weights"]
    W_SEMANTIC = weights["W_SEMANTIC"]
    W_EXPERIENCE = weights["W_EXPERIENCE"]
    W_COMPANY_TYPE = weights["W_COMPANY_TYPE"]
    W_ML_SIGNALS = weights["W_ML_SIGNALS"]
    W_BEHAVIORAL = weights["W_BEHAVIORAL"]
    W_SAVED_RECRUITERS = weights["W_SAVED_RECRUITERS"]
    W_GITHUB = weights["W_GITHUB"]
    W_ASSESSMENT = weights["W_ASSESSMENT"]
    W_PROFILE_COMPLETE = weights["W_PROFILE_COMPLETE"]
    W_EDUCATION = weights["W_EDUCATION"]
    W_RECENCY = weights["W_RECENCY"]
    W_ENGAGEMENT = weights["W_ENGAGEMENT"]
    W_TRUST = weights["W_TRUST"]
    W_LOCATION = weights["W_LOCATION"]
    W_ML_RATIO = weights["W_ML_RATIO"]
    RRF_K = weights["RRF_K"]
    RRF_WEIGHT = weights["RRF_WEIGHT"]

    penalties = _config["penalties"]
    HONEYPOT_PENALTY = penalties["HONEYPOT_PENALTY"]
    WRONG_TITLE_PENALTY = penalties["WRONG_TITLE_PENALTY"]
    CONSULTING_ONLY_PENALTY = penalties["CONSULTING_ONLY_PENALTY"]
    TITLE_CHASER_PENALTY = penalties["TITLE_CHASER_PENALTY"]
    LONG_NOTICE_PENALTY = penalties.get("LONG_NOTICE_PENALTY", -15)
    INTERNATIONAL_PENALTY = penalties.get("INTERNATIONAL_PENALTY", -25)
    OFFSITE_LOCATION_PENALTY = penalties.get("OFFSITE_LOCATION_PENALTY", -12)
    KEYWORD_STUFFER_PENALTY = penalties.get("KEYWORD_STUFFER_PENALTY", -45)
    CURRENT_CONSULTING_PENALTY = penalties.get("CURRENT_CONSULTING_PENALTY", -18)
    LGBM_SCORE_MULTIPLIER = float(_config.get("lgbm_score_multiplier", 0.15))

    CONSULTING_FIRMS = list(_config.get("heuristics", {}).get("consulting_firms", []))


reload_config()


def is_consulting_employer(company_name: str) -> bool:
    name = (company_name or "").lower().strip()
    if not name:
        return False
    return any(firm in name for firm in CONSULTING_FIRMS)


def is_current_consulting(meta: dict) -> bool:
    if meta.get("current_consulting") is not None:
        return bool(meta.get("current_consulting"))
    return is_consulting_employer(meta.get("current_company", "") or "")


def is_keyword_stuffer(meta: dict) -> bool:
    skill_count = int(meta.get("skill_count", 0) or 0)
    years = float(meta.get("years_exp", 0) or 0)
    return skill_count > 80 and years < 3


def apply_availability_location_penalties(meta: dict) -> float:
    """JD logistics penalties — notice/location/title-chaser only (no response double-count)."""
    penalty = 0.0
    notice = int(meta.get("notice_days", 60) or 60)
    loc = float(meta.get("location_score", 0) or 0)
    relocate = bool(meta.get("willing_to_relocate"))

    if notice > 90:
        penalty += LONG_NOTICE_PENALTY
    if loc == 0.0 and not relocate:
        penalty += INTERNATIONAL_PENALTY
    elif loc <= 1.5 and not relocate:
        penalty += OFFSITE_LOCATION_PENALTY

    if meta.get("title_chaser"):
        penalty += TITLE_CHASER_PENALTY

    if is_current_consulting(meta) and not meta.get("consulting_only"):
        penalty += CURRENT_CONSULTING_PENALTY

    if is_keyword_stuffer(meta):
        penalty += KEYWORD_STUFFER_PENALTY

    return penalty


def norm_semantic(raw_logit: float, w_semantic: float = W_SEMANTIC) -> float:
    return (1.0 / (1.0 + math.exp(-(raw_logit + 2.0)))) * w_semantic


def score_experience(years: float, w_experience: float = W_EXPERIENCE) -> float:
    if years <= 0:
        return 0.0
    peak = 7.0
    sigma = 3.0
    raw = math.exp(-0.5 * ((years - peak) / sigma) ** 2)
    if years < 3:
        raw *= 0.3
    elif years > 15:
        raw *= 0.6
    return raw * w_experience


def score_ml_signals(count: float, w_ml_signals: float = W_ML_SIGNALS) -> float:
    return min(count, 1.0) * w_ml_signals


def score_location(location_score: float, w_location: float = W_LOCATION) -> float:
    loc = float(location_score or 0.0)
    if loc <= 1.5:
        return 0.0
    return (loc / 5.0) * w_location


def score_ml_role_ratio(ratio: float, w_ml_ratio: float = W_ML_RATIO) -> float:
    ratio = float(ratio or 0.0)
    if ratio >= 0.5:
        return min(ratio, 1.0) * w_ml_ratio
    elif ratio >= 0.2:
        return ratio * w_ml_ratio * 0.5
    else:
        return max(-5.0, (ratio - 0.2) * 25)


def score_saved_by_recruiters(saved: int, w_saved_recruiters: float = W_SAVED_RECRUITERS) -> float:
    if saved <= 0:
        return 0.0
    return min(saved / 15.0, 1.0) * w_saved_recruiters


def score_profile_completeness(score: float, w_profile_complete: float = W_PROFILE_COMPLETE) -> float:
    if score <= 0:
        return 0.0
    if score >= 90:
        return w_profile_complete
    elif score >= 70:
        return (score - 70) / 20.0 * w_profile_complete
    elif score < 50:
        return -1.0
    return 0.0


def score_engagement(meta: dict, w_engagement: float = W_ENGAGEMENT) -> float:
    score = 0.0
    apps = meta.get("applications_submitted", 0) or 0
    if apps >= 5:
        score += 1.5
    elif apps >= 2:
        score += 0.8
    elif apps == 0:
        score -= 0.5

    searches = meta.get("search_appearance_30d", 0) or 0
    if searches >= 200:
        score += 1.0
    elif searches >= 50:
        score += 0.5

    resp_hrs = meta.get("avg_response_time_hours", 999) or 999
    if resp_hrs <= 4:
        score += 1.0
    elif resp_hrs <= 24:
        score += 0.5
    elif resp_hrs >= 168:
        score -= 1.0

    endorsements = meta.get("endorsements_received", 0) or 0
    if endorsements >= 50:
        score += 0.5

    if meta.get("linkedin_connected", False) or False:
        score += 0.5

    return min(score, w_engagement)


def score_trust(meta: dict, w_trust: float = W_TRUST) -> float:
    score = 0.0
    if meta.get("verified_email", False) or False:
        score += 1.0
    if meta.get("verified_phone", False) or False:
        score += 1.0
    return min(score, w_trust)


def score_recency(meta: dict, w_recency: float = W_RECENCY) -> float:
    """Platform activity recency — uses W_RECENCY from config."""
    lad = int(meta.get("last_active_days", 365) or 365)
    if 0 <= lad <= 30:
        return w_recency
    if lad <= 90:
        return w_recency * 0.4
    if lad <= 180:
        return 0.0
    return -w_recency * 0.5


def score_behavioral(meta: dict, w_behavioral: float = W_BEHAVIORAL) -> float:
    """Availability/engagement — notice >90d and inactivity handled elsewhere."""
    score = 0.0
    response_rate = meta.get("response_rate", 0.5) or 0.5
    notice_days = meta.get("notice_days", 60) or 60
    open_to_work = meta.get("open_to_work", False) or False
    interview_completion = meta.get("interview_completion", 0.5) or 0.5
    offer_acceptance = meta.get("offer_acceptance", -1) or -1
    willing_to_relocate = meta.get("willing_to_relocate", False) or False

    if open_to_work:
        score += 3.0

    if notice_days == 0:
        score += 3.0
    elif notice_days <= 30:
        score += 2.0
    elif notice_days <= 60:
        score += 0.5
    elif notice_days <= 90:
        score -= 1.0

    score += (response_rate - 0.5) * w_behavioral

    score += (interview_completion - 0.5) * 3.0

    if offer_acceptance >= 0:
        score += (offer_acceptance - 0.5) * 2.0

    if willing_to_relocate:
        score += 1.0

    return max(-w_behavioral, min(score, w_behavioral))


def score_github(github_score: float, w_github: float = W_GITHUB) -> float:
    if github_score < 0:
        return 0.0
    return (github_score / 100.0) * w_github


def score_assessment(core_skill_score: float, avg_assessment: float, w_assessment: float = W_ASSESSMENT) -> float:
    if core_skill_score >= 0:
        return (core_skill_score / 100.0) * w_assessment
    elif avg_assessment >= 0:
        return (avg_assessment / 100.0) * (w_assessment * 0.5)
    return 0.0


def score_jd_term_bonus(meta: dict) -> float:
    return min(float(meta.get("jd_term_bonus", 0.0) or 0.0), 20.0)


def score_elite_company_bonus(meta: dict) -> float:
    return float(meta.get("elite_company_bonus", 0.0) or 0.0)


def qualifies_for_product_bonus(meta: dict) -> bool:
    if meta.get("consulting_only"):
        return False
    if is_current_consulting(meta):
        return False
    return bool(meta.get("has_product_company"))


def build_features(meta: dict) -> list:
    years_exp = float(meta.get("years_exp", 0) or 0)
    ml_signal = float(meta.get("ml_signal_count", 0) or 0)
    jd_bonus = float(meta.get("jd_term_bonus", 0.0) or 0.0)
    elite_bonus = float(meta.get("elite_company_bonus", 0.0) or 0.0)
    product_co = 1.0 if qualifies_for_product_bonus(meta) else 0.0
    consulting = 1.0 if meta.get("consulting_only") else 0.0
    github = float(meta.get("github_score", -1) or -1)
    github = max(github, 0.0) / 100.0
    core_skill = float(meta.get("core_skill_score", -1) or -1)
    core_skill = max(core_skill, 0.0) / 100.0
    avg_assess = float(meta.get("avg_assessment", -1) or -1)
    avg_assess = max(avg_assess, 0.0) / 100.0
    edu_tier = 1.0 if meta.get("edu_tier_1") else 0.0
    has_pub = 1.0 if meta.get("has_external_validation") else 0.0
    resp_rate = float(meta.get("response_rate", 0.5) or 0.5)
    notice = float(meta.get("notice_days", 90) or 90)
    notice_norm = max(0.0, 1.0 - notice / 180.0)
    open_work = 1.0 if meta.get("open_to_work") else 0.0
    last_active = float(meta.get("last_active_days", 365) or 365)
    last_active_n = max(0.0, 1.0 - last_active / 365.0)
    interview_comp = float(meta.get("interview_completion", 0.5) or 0.5)
    offer_acc = float(meta.get("offer_acceptance", -1) or -1)
    offer_acc = max(offer_acc, 0.0) if offer_acc >= 0 else 0.5
    profile_pct = float(meta.get("profile_completeness", 50) or 50) / 100.0
    apps_submitted = float(meta.get("applications_submitted", 0) or 0)
    search_appear = float(meta.get("search_appearance_30d", 0) or 0)
    resp_time = float(meta.get("avg_response_time_hours", 999) or 999)
    resp_time_n = max(0.0, 1.0 - resp_time / 999.0)
    endorsements = float(meta.get("endorsements_received", 0) or 0)
    linkedin = 1.0 if meta.get("linkedin_connected") else 0.0
    email_ver = 1.0 if meta.get("verified_email") else 0.0
    phone_ver = 1.0 if meta.get("verified_phone") else 0.0
    research = float(meta.get("research_founding_score", 0) or 0)
    skill_count = float(meta.get("skill_count", 0) or 0) / 100.0
    relocate = 1.0 if meta.get("willing_to_relocate") else 0.0

    return [
        years_exp, ml_signal, jd_bonus, elite_bonus,
        product_co, consulting, github, core_skill, avg_assess,
        edu_tier, has_pub, resp_rate, notice_norm, open_work,
        last_active_n, interview_comp, offer_acc, profile_pct,
        apps_submitted, search_appear, resp_time_n, endorsements,
        linkedin, email_ver, phone_ver, research, skill_count, relocate,
    ]


def generate_reasoning(meta: dict, semantic_score: float, final_score: float) -> str:
    cid = meta.get("candidate_id", "")

    if meta.get("honeypot"):
        return (
            "Profile excluded: contains impossible credential claims "
            "(skills listed as expert-level with zero months of use). "
            "This pattern indicates a synthetic or deliberately inflated profile."
        )

    if meta.get("wrong_title"):
        ct = meta.get("current_title", "unknown")
        return (
            f"Excluded from shortlist: current role ('{ct}') falls outside the ML/IR engineering domain. "
            "JD explicitly requires a background in applied NLP, information retrieval, or production ML "
            "— not marketing, sales, HR, or embedded/robotics."
        )

    yrs = meta.get("years_exp", 0) or 0
    ml = meta.get("ml_signal_count", 0.0) or 0.0
    loc = meta.get("location_score", 0.0) or 0.0
    ratio = meta.get("ml_role_ratio", 0.0) or 0.0
    ml_mo = int(meta.get("ml_role_months", 0) or 0)
    tot_mo = int(meta.get("total_months", 1) or 1)
    saved = meta.get("saved_by_recruiters", 0) or 0
    nd = meta.get("notice_days", 60) or 60
    gh = meta.get("github_score", -1)
    rfs = meta.get("research_founding_score", 0.0) or 0.0
    chaser = meta.get("title_chaser", False)
    avg_t = meta.get("avg_tenure_months", 999) or 999
    company = meta.get("current_company", "") or ""
    lad = meta.get("last_active_days", 365) or 365
    rr = meta.get("response_rate", 0.5) or 0.5

    lead_parts = []
    secondary_parts = []
    concern_parts = []

    if loc >= 4.0:
        lead_parts.append(
            f"Location is a direct match for the JD's preferred cities (score: {loc:.0f}/5) — "
            "no relocation friction expected."
        )
    elif ratio >= 0.65 and ml_mo >= 36:
        yrs_ml = ml_mo // 12
        lead_parts.append(
            f"{yrs_ml} of their {yrs:.0f} total years ({ratio:.0%}) were spent in ML-titled roles — "
            "career composition closely matches the JD's '4-5 of 6-8 years in applied ML' target."
        )
    elif ml >= 0.8:
        lead_parts.append(
            "Strong production ML depth across their career — retrieval, ranking, or embedding "
            "systems detected in role descriptions."
        )
    elif saved >= 8:
        lead_parts.append(
            f"High recruiter demand signal: {saved} recruiters have bookmarked this profile "
            "in the last 30 days — already being actively competed for."
        )
    elif rfs == 15.0 and ml >= 0.3:
        lead_parts.append(
            "Research/founding background with production deployment signals — the research-plus-ship "
            "profile this JD values."
        )
    elif final_score >= 80:
        lead_parts.append(
            "Multiple signals align with what the JD is looking for — details below."
        )

    if 5 <= yrs <= 9:
        secondary_parts.append(f"{yrs:.0f} years total experience, squarely in the 5-9yr band.")
    elif 4 <= yrs < 5:
        secondary_parts.append(f"{yrs:.0f} years experience — just below the 5yr threshold, still in scope.")
    elif 9 < yrs <= 12:
        secondary_parts.append(f"{yrs:.0f} years experience — slightly senior, but within reasonable range.")
    elif yrs < 4:
        concern_parts.append(f"Only {yrs:.0f} years experience — below the 5yr minimum; junior risk.")
    else:
        concern_parts.append(
            f"{yrs:.0f} years experience — significantly senior; may be overqualified or above-budget."
        )

    if ratio >= 0.5 and not any("career composition" in p for p in lead_parts):
        yrs_ml = ml_mo // 12
        secondary_parts.append(f"{yrs_ml} of {yrs:.0f} years ({ratio:.0%}) in ML-titled roles.")
    elif ratio < 0.2 and tot_mo > 12:
        concern_parts.append(
            f"Only {ratio:.0%} of career in ML-titled roles — most experience was in other domains."
        )

    if meta.get("consulting_only"):
        concern_parts.append(
            f"Consulting-only background ({company}) — JD warns this is disqualifying without product experience."
        )
    elif is_current_consulting(meta):
        concern_parts.append(
            f"Currently at a services/consulting employer ({company}) — JD prefers product-company tenure."
        )
    elif qualifies_for_product_bonus(meta):
        secondary_parts.append(f"Product-company background confirmed ({company}).")

    if ml >= 0.5 and not any("production ML" in p.lower() for p in lead_parts):
        secondary_parts.append(f"Moderate-to-strong production ML signals (score: {ml:.2f}).")
    elif ml < 0.2:
        concern_parts.append(
            "No meaningful production ML/retrieval signals in role descriptions — skills may be self-reported."
        )

    if rfs == 15.0:
        if ml >= 0.3:
            secondary_parts.append("Research background with production ML signals.")
        else:
            concern_parts.append("Research Scientist title but minimal production deployment signals.")
    elif rfs == -5.0:
        concern_parts.append("Intern/student-only career history — insufficient professional experience.")

    if chaser:
        concern_parts.append(
            f"Rapid seniority escalation with avg {avg_t:.0f} months per role — title-chaser pattern."
        )

    if loc >= 4.0 and not any("Location" in p for p in lead_parts):
        secondary_parts.append("Location aligns with JD's Pune/Noida preference.")
    elif loc == 0.0:
        concern_parts.append("Based outside India with no relocation flag — JD offers no visa sponsorship.")
    elif loc <= 1.5:
        concern_parts.append("Outside JD's preferred cities (Pune/Noida) without a relocation signal.")

    if is_keyword_stuffer(meta):
        concern_parts.append(
            f"Profile lists {meta.get('skill_count')} skills with only {yrs:.0f} years experience — "
            "keyword inflation pattern."
        )

    if saved >= 10 and not any(str(saved) in p for p in lead_parts):
        secondary_parts.append(f"Bookmarked by {saved} recruiters in 30 days.")
    elif saved >= 5 and not any(str(saved) in p for p in lead_parts):
        secondary_parts.append(f"Bookmarked by {saved} recruiters recently.")

    if gh >= 70:
        secondary_parts.append(f"Active GitHub presence ({gh:.0f}/100).")
    elif gh >= 30:
        secondary_parts.append(f"Moderate GitHub activity ({gh:.0f}/100).")

    if meta.get("core_skill_score", -1) >= 80:
        secondary_parts.append(f"Core ML assessment: {meta['core_skill_score']:.0f}/100.")
    elif meta.get("avg_assessment", -1) >= 70:
        secondary_parts.append(f"Platform assessment average: {meta['avg_assessment']:.0f}/100.")

    if meta.get("edu_tier_1"):
        secondary_parts.append("Tier-1 institution background.")

    if float(meta.get("jd_term_bonus", 0) or 0) >= 8:
        secondary_parts.append("Strong JD term overlap (retrieval/ranking/vector search vocabulary).")

    avail = []
    if meta.get("open_to_work"):
        avail.append("open to work")
    if nd == 0:
        avail.append("immediate joiner")
    elif nd <= 30:
        avail.append(f"{nd}-day notice (buyable per JD)")
    elif nd > 90:
        concern_parts.append(f"{nd}-day notice period — long lead time.")
    if rr >= 0.75:
        avail.append(f"{rr:.0%} response rate")
    elif rr < 0.3:
        concern_parts.append(f"Low recruiter response rate ({rr:.0%}) — reachability risk.")
    if 0 <= lad <= 7:
        avail.append("active this week")
    elif lad > 180:
        concern_parts.append(f"Inactive on platform for {lad} days.")
    if avail:
        secondary_parts.append("Availability: " + ", ".join(avail) + ".")

    if final_score >= 50:
        all_parts = lead_parts + secondary_parts + concern_parts
    else:
        all_parts = concern_parts + lead_parts + secondary_parts

    if not all_parts:
        all_parts = [f"{yrs:.0f} years experience. ML signal: {ml:.2f}. Score: {final_score:.1f}."]

    return " ".join(all_parts)
