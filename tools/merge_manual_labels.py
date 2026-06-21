"""
Merge manual relevance edits from labeling_sample.csv into gold_labels_proxy.csv.

Usage:
  python merge_manual_labels.py
  python merge_manual_labels.py --manual labeling_sample.csv --gold gold_labels_proxy.csv
"""

import argparse
import csv
from pathlib import Path

BASE = Path(__file__).parent
ROOT = Path(__file__).parent.parent


def load_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_gold(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Merge manual labels into proxy gold file.")
    parser.add_argument("--manual", default=str(ROOT / "labeling_sample.csv"))
    parser.add_argument("--gold", default=str(ROOT / "gold_labels_proxy.csv"))
    args = parser.parse_args()

    manual_path = BASE / args.manual
    gold_path = BASE / args.gold

    if not manual_path.exists():
        raise FileNotFoundError(f"{manual_path} not found. Run sample_for_labeling.py first.")
    if not gold_path.exists():
        raise FileNotFoundError(f"{gold_path} not found. Run build_proxy_gold.py first.")

    manual_rows = load_csv(manual_path)
    gold_rows = load_csv(gold_path)
    gold_by_id = {r["candidate_id"]: r for r in gold_rows}

    updated = 0
    added = 0
    for row in manual_rows:
        manual = row.get("manual_relevance", "").strip()
        if not manual:
            continue
        rel = int(float(manual))
        cid = row["candidate_id"]
        if cid in gold_by_id:
            if int(gold_by_id[cid]["relevance"]) != rel:
                gold_by_id[cid]["relevance"] = rel
                updated += 1
        else:
            new_row = {
                "candidate_id": cid,
                "relevance": rel,
                "bucket": row.get("bucket", "manual"),
                "title": row.get("title", ""),
                "company": row.get("company", ""),
                "years_exp": row.get("years_exp", ""),
                "ml_role_ratio": row.get("ml_role_ratio", ""),
                "ml_signal_count": row.get("ml_signal_count", ""),
                "location_score": row.get("location_score", ""),
                "saved_by_recruiters": row.get("saved_by_recruiters", ""),
            }
            gold_rows.append(new_row)
            gold_by_id[cid] = new_row
            added += 1

    fieldnames = list(gold_rows[0].keys()) if gold_rows else [
        "candidate_id", "relevance", "bucket", "title", "company",
        "years_exp", "ml_role_ratio", "ml_signal_count",
        "location_score", "saved_by_recruiters",
    ]
    write_gold(gold_path, gold_rows, fieldnames)

    print(f"Merged manual labels into {gold_path.name}")
    print(f"  updated: {updated}")
    print(f"  added:   {added}")
    print("Next: python self_eval.py  or  python tune_weights.py --trials 200")


if __name__ == "__main__":
    main()
