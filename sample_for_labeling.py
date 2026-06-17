"""
Export a stratified sample of candidates for manual relevance labeling.

Usage:
  python sample_for_labeling.py
  python sample_for_labeling.py --per-bucket 40 --output labeling_sample.csv

Edit the `manual_relevance` column (0–3), save, then merge into gold_labels_proxy.csv.
"""

import argparse
import csv
import pickle
import random
from collections import defaultdict
from pathlib import Path

from proxy_labels import bucket_label, compute_proxy_relevance

BASE = Path(__file__).parent

BUCKET_TARGETS = {
    "obvious_positive": 60,
    "obvious_negative": 60,
    "honeypot": 40,
    "wrong_title": 40,
    "keyword_stuffer": 30,
    "domain_pivot": 40,
    "consulting_only": 30,
    "borderline": 80,
}


def main():
    parser = argparse.ArgumentParser(description="Sample candidates for manual labeling.")
    parser.add_argument("--output", default="labeling_sample.csv")
    parser.add_argument("--per-bucket", type=int, default=0, help="Override per-bucket sample size")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--meta", default="candidate_meta.pkl")
    args = parser.parse_args()

    meta_path = BASE / args.meta
    if not meta_path.exists():
        raise FileNotFoundError(f"{meta_path} not found. Run precompute.py first.")

    with open(meta_path, "rb") as f:
        metadata = pickle.load(f)

    rng = random.Random(args.seed)
    by_bucket: dict[str, list] = defaultdict(list)
    for meta in metadata:
        by_bucket[bucket_label(meta)].append(meta)

    selected = []
    seen_ids = set()
    for bucket, target in BUCKET_TARGETS.items():
        n = args.per_bucket or target
        pool = by_bucket.get(bucket, [])
        rng.shuffle(pool)
        for meta in pool[:n]:
            cid = meta["candidate_id"]
            if cid in seen_ids:
                continue
            seen_ids.add(cid)
            selected.append(meta)

    out_path = BASE / args.output
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "candidate_id",
                "auto_relevance",
                "manual_relevance",
                "bucket",
                "title",
                "company",
                "years_exp",
                "ml_role_ratio",
                "ml_signal_count",
                "location_score",
                "response_rate",
                "last_active_days",
                "saved_by_recruiters",
                "notes",
            ],
        )
        writer.writeheader()
        for meta in selected:
            writer.writerow({
                "candidate_id": meta["candidate_id"],
                "auto_relevance": compute_proxy_relevance(meta),
                "manual_relevance": "",
                "bucket": bucket_label(meta),
                "title": meta.get("current_title", ""),
                "company": meta.get("current_company", ""),
                "years_exp": meta.get("years_exp", 0),
                "ml_role_ratio": round(float(meta.get("ml_role_ratio", 0) or 0), 2),
                "ml_signal_count": round(float(meta.get("ml_signal_count", 0) or 0), 2),
                "location_score": meta.get("location_score", 0),
                "response_rate": round(float(meta.get("response_rate", 0.5) or 0.5), 2),
                "last_active_days": meta.get("last_active_days", 365),
                "saved_by_recruiters": meta.get("saved_by_recruiters", 0),
                "notes": "",
            })

    print(f"Wrote {len(selected)} candidates to {out_path.name}")
    print("Fill in manual_relevance (0–3) for rows you review, then merge into gold_labels_proxy.csv")


if __name__ == "__main__":
    main()
