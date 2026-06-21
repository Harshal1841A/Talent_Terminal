"""
Evaluate a ranking submission CSV against gold labels.

Usage:
  python build_proxy_gold.py --min-relevance 1
  python rank.py
  python self_eval.py --submission "Team Rocket.csv" --gold gold_labels_proxy.csv
"""

import argparse
import csv
import sys
from pathlib import Path

from eval_metrics import evaluate_ranking

BASE = Path(__file__).parent
ROOT = Path(__file__).parent.parent


def load_gold_map(path: Path) -> dict[str, int]:
    gold = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rel = row.get("manual_relevance") or row.get("relevance")
            if rel is None or str(rel).strip() == "":
                continue
            gold[row["candidate_id"]] = int(float(rel))
    return gold


def load_submission_ids(path: Path) -> list[str]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [row["candidate_id"] for row in reader]


def main():
    parser = argparse.ArgumentParser(description="Evaluate ranking output against gold labels.")
    parser.add_argument("--submission", default=str(ROOT / "Team Rocket.csv"))
    parser.add_argument("--gold", default=str(ROOT / "gold_labels_proxy.csv"))
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    sub_path = BASE / args.submission
    gold_path = BASE / args.gold

    if not sub_path.exists():
        print(f"Error: submission file not found: {sub_path}")
        print("Run rank.py first.")
        sys.exit(1)

    if not gold_path.exists():
        print(f"Error: gold file not found: {gold_path}")
        print("Run: python build_proxy_gold.py --min-relevance 1")
        print("Or:  python sample_for_labeling.py  (for manual labeling)")
        sys.exit(1)

    gold_map = load_gold_map(gold_path)
    if not gold_map:
        print(f"Error: no labels in {gold_path}")
        sys.exit(1)

    ranked_ids = load_submission_ids(sub_path)
    metrics = evaluate_ranking(ranked_ids, gold_map, k=args.k)

    print(f"Submission: {sub_path.name} ({len(ranked_ids)} rows)")
    print(f"Gold labels: {gold_path.name} ({len(gold_map):,} labeled)")
    print("\n=== Evaluation Results ===")
    for key, val in metrics.items():
        print(f"{key}: {val:.4f}")
    print("==========================")


if __name__ == "__main__":
    main()
