"""
Pick the 50 highest-value rows from labeling_sample.csv for manual review.

Usage:
  python prioritize_labeling_review.py
  python prioritize_labeling_review.py --count 50 --output labeling_review_priority.csv
"""

import argparse
import csv
import pickle
from pathlib import Path

BASE = Path(__file__).parent.parent


def load_top100_ids() -> set[str]:
    sub = BASE / "Team Rocket.csv"
    if not sub.exists():
        return set()
    with open(sub, newline="", encoding="utf-8") as f:
        return {row["candidate_id"] for row in csv.DictReader(f)}


def review_priority(row: dict, top100: set[str]) -> tuple[int, str]:
    score = 0
    reasons = []
    bucket = row.get("bucket", "")
    cid = row["candidate_id"]
    auto = int(float(row.get("auto_relevance") or 0))
    loc = float(row.get("location_score") or 0)
    ml_ratio = float(row.get("ml_role_ratio") or 0)
    years = float(row.get("years_exp") or 0)
    rr = float(row.get("response_rate") or 0.5)

    if bucket == "borderline":
        score += 40
        reasons.append("borderline bucket")
    elif bucket == "domain_pivot":
        score += 35
        reasons.append("domain pivot")
    elif bucket in ("honeypot", "wrong_title", "keyword_stuffer"):
        score += 30
        reasons.append(bucket)
    elif bucket == "consulting_only":
        score += 25
        reasons.append("consulting only")

    if cid in top100:
        score += 50
        reasons.append("in current top 100")

    if auto == 2:
        score += 20
        reasons.append("auto=2 (strong maybe)")
    if auto == 1 and bucket != "borderline":
        score += 15
        reasons.append("auto=1")

    if loc <= 1.5 and auto >= 3:
        score += 25
        reasons.append("offsite but auto-labeled 3")
    if loc == 0.0:
        score += 30
        reasons.append("international")

    if ml_ratio < 0.35 and years >= 5:
        score += 20
        reasons.append("low ML ratio")

    if rr < 0.25:
        score += 15
        reasons.append("low response rate")

    if "consulting" in (row.get("company") or "").lower() or "tcs" in (row.get("company") or "").lower():
        score += 10
        reasons.append("consulting company name")

    title = (row.get("title") or "").lower()
    if any(k in title for k in ("vision", "cv ", "robotics", "speech")):
        score += 15
        reasons.append("non-NLP/IR title")

    return score, "; ".join(reasons) if reasons else "general review"


def main():
    parser = argparse.ArgumentParser(description="Prioritize labeling sample for manual review.")
    parser.add_argument("--input", default="labeling_sample.csv")
    parser.add_argument("--output", default="labeling_review_priority.csv")
    parser.add_argument("--count", type=int, default=50)
    args = parser.parse_args()

    src = BASE / args.input
    if not src.exists():
        raise FileNotFoundError(f"{src} not found. Run sample_for_labeling.py first.")

    with open(src, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    top100 = load_top100_ids()
    scored = []
    for row in rows:
        prio, reason = review_priority(row, top100)
        scored.append((prio, row, reason))

    scored.sort(key=lambda x: (-x[0], x[1]["candidate_id"]))
    picked = scored[: args.count]

    fieldnames = list(rows[0].keys()) + ["review_priority", "review_reason"]
    out = BASE / args.output
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for prio, row, reason in picked:
            row = dict(row)
            row["review_priority"] = prio
            row["review_reason"] = reason
            writer.writerow(row)

    print(f"Wrote top {len(picked)} priority rows to {out.name}")
    print("\nReview these first (highest priority):")
    for i, (prio, row, reason) in enumerate(picked[:10], 1):
        print(
            f"  {i:2d}. [{prio:3d}] {row['candidate_id']}  "
            f"{row.get('title', '?')} @ {row.get('company', '?')}  — {reason}"
        )
    print(f"\nFill manual_relevance (0-3) in {out.name}, then:")
    print("  copy manual_relevance back to labeling_sample.csv (or merge directly)")
    print("  python merge_manual_labels.py --manual labeling_review_priority.csv")


if __name__ == "__main__":
    main()
