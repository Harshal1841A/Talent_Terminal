"""
analyze_data.py — Data Distribution Analysis
Understand the candidate pool before tuning weights.
Outputs analysis to console + saves analysis_report.txt
"""
import json
import pickle
import sys
from collections import Counter
from datetime import date, datetime
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

BASE = Path(__file__).parent

def main():
    print("Loading candidate_meta.pkl...")
    with open(BASE / "candidate_meta.pkl", "rb") as f:
        metadata = pickle.load(f)

    n = len(metadata)
    print(f"Analyzing {n:,} candidates...\n")

    lines = []

    def section(title):
        s = f"\n{'='*60}\n{title}\n{'='*60}"
        print(s)
        lines.append(s)

    def log(msg):
        print(msg)
        lines.append(str(msg))

    # ── Experience distribution ───────────────────────────────────────
    section("EXPERIENCE DISTRIBUTION")
    exp_buckets = Counter()
    for m in metadata:
        y = m.get("years_exp", 0)
        if y < 2: exp_buckets["0-2yr"] += 1
        elif y < 4: exp_buckets["2-4yr"] += 1
        elif y < 5: exp_buckets["4-5yr"] += 1
        elif y <= 9: exp_buckets["5-9yr (TARGET)"] += 1
        elif y <= 12: exp_buckets["9-12yr"] += 1
        else: exp_buckets["12yr+"] += 1

    for bucket, count in sorted(exp_buckets.items()):
        pct = count / n * 100
        bar = "#" * int(pct / 2)
        log(f"  {bucket:<20} {count:>6,}  ({pct:5.1f}%)  {bar}")

    # ── Company type ─────────────────────────────────────────────────
    section("COMPANY TYPE")
    product = sum(1 for m in metadata if m.get("has_product_company"))
    consulting_only = sum(1 for m in metadata if m.get("consulting_only"))
    log(f"  Product company experience: {product:,} ({product/n*100:.1f}%)")
    log(f"  Consulting-only:            {consulting_only:,} ({consulting_only/n*100:.1f}%)")

    # ── Honeypot & disqualifiers ─────────────────────────────────────
    section("HONEYPOTS & DISQUALIFIERS")
    honeypots = sum(1 for m in metadata if m.get("honeypot"))
    wrong_title = sum(1 for m in metadata if m.get("wrong_title"))
    log(f"  Honeypots detected:  {honeypots:,} ({honeypots/n*100:.1f}%)")
    log(f"  Wrong title (disq):  {wrong_title:,} ({wrong_title/n*100:.1f}%)")
    log(f"  Clean candidates:    {n - honeypots - wrong_title:,}")

    # ── ML signals ───────────────────────────────────────────────────
    section("PRODUCTION ML SIGNALS")
    ml_buckets = Counter()
    for m in metadata:
        c = m.get("ml_signal_count", 0)
        if c == 0: ml_buckets["0 signals"] += 1
        elif c <= 2: ml_buckets["1-2 signals"] += 1
        elif c <= 4: ml_buckets["3-4 signals"] += 1
        else: ml_buckets["5+ signals (ideal)"] += 1

    for bucket, count in sorted(ml_buckets.items()):
        log(f"  {bucket:<25} {count:>6,}  ({count/n*100:.1f}%)")

    # ── Behavioral signals distribution ──────────────────────────────
    section("BEHAVIORAL SIGNALS")
    open_to_work = sum(1 for m in metadata if m.get("open_to_work"))
    immediate = sum(1 for m in metadata if m.get("notice_days", 90) == 0)
    short_notice = sum(1 for m in metadata if 0 < m.get("notice_days", 90) <= 30)
    long_notice = sum(1 for m in metadata if m.get("notice_days", 90) > 90)
    active_7d = sum(1 for m in metadata if 0 <= m.get("last_active_days", 90) <= 7)
    active_30d = sum(1 for m in metadata if 0 <= m.get("last_active_days", 90) <= 30)
    verified_both = sum(1 for m in metadata if m.get("verified_email") and m.get("verified_phone"))
    linkedin = sum(1 for m in metadata if m.get("linkedin_connected"))

    log(f"  Open to work:           {open_to_work:,} ({open_to_work/n*100:.1f}%)")
    log(f"  Immediate joiner:       {immediate:,} ({immediate/n*100:.1f}%)")
    log(f"  Notice <= 30 days:      {short_notice:,} ({short_notice/n*100:.1f}%)")
    log(f"  Notice > 90 days:       {long_notice:,} ({long_notice/n*100:.1f}%)")
    log(f"  Active last 7 days:     {active_7d:,} ({active_7d/n*100:.1f}%)")
    log(f"  Active last 30 days:    {active_30d:,} ({active_30d/n*100:.1f}%)")
    log(f"  Verified email+phone:   {verified_both:,} ({verified_both/n*100:.1f}%)")
    log(f"  LinkedIn connected:     {linkedin:,} ({linkedin/n*100:.1f}%)")

    # ── Saved by recruiters ──────────────────────────────────────────
    section("SAVED BY RECRUITERS (30d)")
    saved_buckets = Counter()
    for m in metadata:
        s = m.get("saved_by_recruiters", 0)
        if s == 0: saved_buckets["0 saves"] += 1
        elif s <= 3: saved_buckets["1-3 saves"] += 1
        elif s <= 7: saved_buckets["4-7 saves"] += 1
        elif s <= 15: saved_buckets["8-15 saves"] += 1
        else: saved_buckets["16+ saves"] += 1

    for bucket, count in sorted(saved_buckets.items()):
        log(f"  {bucket:<20} {count:>6,}  ({count/n*100:.1f}%)")

    # ── Education ────────────────────────────────────────────────────
    section("EDUCATION")
    tier1 = sum(1 for m in metadata if m.get("edu_tier_1"))
    log(f"  Tier-1 institution:     {tier1:,} ({tier1/n*100:.1f}%)")

    # ── GitHub ───────────────────────────────────────────────────────
    section("GITHUB ACTIVITY")
    no_github = sum(1 for m in metadata if m.get("github_score", -1) < 0)
    high_gh = sum(1 for m in metadata if m.get("github_score", -1) >= 70)
    log(f"  No GitHub linked:   {no_github:,} ({no_github/n*100:.1f}%)")
    log(f"  GitHub score >= 70: {high_gh:,} ({high_gh/n*100:.1f}%)")

    # ── Top 5 candidates in current submission ───────────────────────
    section("CURRENT TOP CANDIDATES (from submission.csv)")
    try:
        import csv
        with open(BASE / "submission.csv", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i >= 10: break
                log(f"  #{row['rank']:<4} {row['candidate_id']}  score={float(row['score']):.2f}")
                log(f"        {row['reasoning'][:100]}...")
    except Exception as e:
        log(f"  Could not read submission.csv: {e}")

    # Save report
    with open(BASE / "analysis_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\nSaved to analysis_report.txt")

if __name__ == "__main__":
    main()
