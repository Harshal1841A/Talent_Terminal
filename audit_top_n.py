"""
Audit top-N submission candidates against JD red flags and proxy relevance.

Usage:
  python audit_top_n.py --submission "Team Rocket.csv" --top 30
"""

import argparse
import csv
import pickle
from pathlib import Path

from proxy_labels import compute_proxy_relevance

BASE = Path(__file__).parent

FLAGS = [
    ("honeypot", "HONEYPOT — should not appear in top 100"),
    ("wrong_title", "WRONG TITLE — hard disqualifier"),
    ("consulting_only", "CONSULTING ONLY — JD warns against"),
    ("title_chaser", "TITLE CHASER — rapid seniority jumps"),
]


def audit_meta(rank: int, meta: dict) -> list[str]:
    issues = []
    for key, msg in FLAGS:
        if meta.get(key):
            issues.append(msg)

    years = float(meta.get("years_exp", 0) or 0)
    ml_ratio = float(meta.get("ml_role_ratio", 0) or 0)
    loc = float(meta.get("location_score", 0) or 0)
    rr = float(meta.get("response_rate", 0.5) or 0.5)
    lad = int(meta.get("last_active_days", 365) or 365)
    notice = int(meta.get("notice_days", 60) or 60)
    skill_count = int(meta.get("skill_count", 0) or 0)

    if years < 4:
        issues.append(f"LOW EXPERIENCE — {years:.0f}yr (JD targets 5-9)")
    elif years > 12:
        issues.append(f"VERY SENIOR — {years:.0f}yr (may be overqualified)")

    if ml_ratio < 0.25 and years >= 5:
        issues.append(f"DOMAIN PIVOT — only {ml_ratio:.0%} career in ML roles")

    if loc == 0.0 and not meta.get("willing_to_relocate"):
        issues.append("INTERNATIONAL / NO RELOCATION — JD offers no visa sponsorship")

    if loc <= 1.5 and not meta.get("willing_to_relocate"):
        issues.append("OUTSIDE PUNE/NOIDA — no relocation signal")

    if rr < 0.3:
        issues.append(f"LOW RESPONSE RATE — {rr:.0%}")

    if lad > 180:
        issues.append(f"INACTIVE — {lad} days since last activity")

    if notice > 90:
        issues.append(f"LONG NOTICE — {notice} days")

    if skill_count > 80 and years < 3:
        issues.append(f"KEYWORD STUFFER — {skill_count} skills, {years:.0f}yr exp")

    if not meta.get("has_product_company") and not meta.get("consulting_only"):
        issues.append("NO PRODUCT COMPANY — blank/freelance career history")

    proxy_rel = compute_proxy_relevance(meta)
    if rank <= 10 and proxy_rel < 2:
        issues.append(f"LOW PROXY RELEVANCE — auto-label={proxy_rel} in top 10")

    return issues


def main():
    parser = argparse.ArgumentParser(description="Audit submission top-N for JD red flags.")
    parser.add_argument("--submission", default="Team Rocket.csv")
    parser.add_argument("--meta", default="candidate_meta.pkl")
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args()

    sub_path = BASE / args.submission
    meta_path = BASE / args.meta

    if not sub_path.exists():
        raise FileNotFoundError(f"{sub_path} not found")
    if not meta_path.exists():
        raise FileNotFoundError(f"{meta_path} not found. Run precompute.py first.")

    with open(meta_path, "rb") as f:
        metadata = pickle.load(f)
    meta_by_id = {m["candidate_id"]: m for m in metadata}

    with open(sub_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))[: args.top]

    print(f"Auditing top {len(rows)} from {sub_path.name}\n")
    flagged = 0
    for row in rows:
        rank = int(row["rank"])
        cid = row["candidate_id"]
        meta = meta_by_id.get(cid)
        if not meta:
            print(f"#{rank:3d} {cid}  MISSING FROM METADATA")
            flagged += 1
            continue

        title = meta.get("current_title", "?")
        company = meta.get("current_company", "?")
        issues = audit_meta(rank, meta)
        rel = compute_proxy_relevance(meta)

        status = "OK" if not issues else "FLAG"
        if issues:
            flagged += 1
        print(f"#{rank:3d} {cid}  [{status}]  proxy_rel={rel}  {title} @ {company}")
        for issue in issues:
            print(f"       ! {issue}")
        if not issues:
            print(f"       + looks aligned with JD")

    print(f"\n{flagged}/{len(rows)} candidates flagged in top {len(rows)}")
    if flagged:
        print("Review flagged rows and re-tune weights before final submission.")


if __name__ == "__main__":
    main()
