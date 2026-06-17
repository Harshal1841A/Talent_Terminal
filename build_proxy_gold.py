"""
Build gold_labels_proxy.csv from JD-aligned rules across the full candidate pool.

Usage:
  python build_proxy_gold.py
  python build_proxy_gold.py --min-relevance 1   # only non-zero labels
"""

import argparse
import csv
import pickle
from collections import Counter
from pathlib import Path

from proxy_labels import bucket_label, compute_proxy_relevance

BASE = Path(__file__).parent


def main():
    parser = argparse.ArgumentParser(description="Build proxy gold labels from candidate metadata.")
    parser.add_argument("--output", default="gold_labels_proxy.csv")
    parser.add_argument("--min-relevance", type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument("--meta", default="candidate_meta.pkl")
    args = parser.parse_args()

    meta_path = BASE / args.meta
    if not meta_path.exists():
        raise FileNotFoundError(f"{meta_path} not found. Run precompute.py first.")

    with open(meta_path, "rb") as f:
        metadata = pickle.load(f)

    rows = []
    dist = Counter()
    buckets = Counter()
    for meta in metadata:
        rel = compute_proxy_relevance(meta)
        if rel < args.min_relevance:
            continue
        rows.append({
            "candidate_id": meta["candidate_id"],
            "relevance": rel,
            "bucket": bucket_label(meta),
            "title": meta.get("current_title", ""),
            "company": meta.get("current_company", ""),
            "years_exp": meta.get("years_exp", 0),
            "ml_role_ratio": round(float(meta.get("ml_role_ratio", 0) or 0), 2),
            "ml_signal_count": round(float(meta.get("ml_signal_count", 0) or 0), 2),
            "location_score": meta.get("location_score", 0),
            "saved_by_recruiters": meta.get("saved_by_recruiters", 0),
        })
        dist[rel] += 1
        buckets[bucket_label(meta)] += 1

    out_path = BASE / args.output
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "candidate_id", "relevance", "bucket", "title", "company",
                "years_exp", "ml_role_ratio", "ml_signal_count",
                "location_score", "saved_by_recruiters",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows):,} labels to {out_path.name}")
    print("Relevance distribution:", dict(sorted(dist.items())))
    print("Bucket distribution:", dict(sorted(buckets.items(), key=lambda x: -x[1])[:12]))
    print("\nNext: review borderline rows, edit relevance column, then run tune_weights.py")


if __name__ == "__main__":
    main()
