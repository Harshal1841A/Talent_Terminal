"""
Pre-submission checklist — verify artifacts, models, and submission format.

Usage:
  python judge_bundle_check.py
  python judge_bundle_check.py --submission "Team Rocket.csv"
"""

import argparse
import sys
from pathlib import Path

BASE = Path(__file__).parent

REQUIRED_CODE = [
    "rank.py",
    "app.py",
    "ranking_pipeline.py",
    "core_scoring.py",
    "config.yaml",
    "precompute.py",
    "precompute_bm25.py",
    "requirements.txt",
    "job_desc.txt",
    "dashboard_template.html",
]

REQUIRED_FOR_RANK = [
    ("candidate_meta.pkl", "Run: python precompute.py"),
    ("faiss_index.bin", "Run: python precompute.py"),
]

OPTIONAL_BUT_RECOMMENDED = [
    ("bm25_index.pkl", "Run: python precompute_bm25.py"),
    ("models/bge-base-en-v1.5", "Run: python download_models.py"),
    ("models/ms-marco-MiniLM-L-6-v2", "Run: python download_models.py"),
    ("lgbm_reranker.pkl", "Run: python train_reranker.py (optional)"),
]

MODEL_FILES = [
    "models/bge-base-en-v1.5",
    "models/ms-marco-MiniLM-L-6-v2",
]


def check_path(rel: str) -> bool:
    return (BASE / rel).exists()


def main():
    parser = argparse.ArgumentParser(description="Verify judge bundle readiness.")
    parser.add_argument("--submission", default="Team Rocket.csv")
    args = parser.parse_args()

    errors = []
    warnings = []

    print("=== Talent Terminal — Judge Bundle Check ===\n")

    print("Code files:")
    for name in REQUIRED_CODE:
        ok = check_path(name)
        print(f"  [{'OK' if ok else 'MISSING'}] {name}")
        if not ok:
            errors.append(f"Missing code file: {name}")

    print("\nPrecomputed artifacts (required for rank.py):")
    for name, hint in REQUIRED_FOR_RANK:
        ok = check_path(name)
        print(f"  [{'OK' if ok else 'MISSING'}] {name}")
        if not ok:
            errors.append(f"{name} — {hint}")

    print("\nOptional artifacts:")
    for name, hint in OPTIONAL_BUT_RECOMMENDED:
        ok = check_path(name)
        print(f"  [{'OK' if ok else 'WARN'}] {name}")
        if not ok:
            warnings.append(f"{name} — {hint}")

    sub_path = BASE / args.submission
    if sub_path.exists():
        from validate_submission import validate_submission
        val_errors = validate_submission(sub_path)
        print(f"\nSubmission ({args.submission}):")
        if val_errors:
            for e in val_errors:
                print(f"  [INVALID] {e}")
            errors.extend(val_errors)
        else:
            print("  [OK] CSV format valid")
    else:
        print(f"\nSubmission ({args.submission}): MISSING")
        warnings.append(f"No submission CSV — run rank.py")

    gold_path = BASE / "gold_labels_proxy.csv"
    if gold_path.exists():
        print("\nProxy eval: gold_labels_proxy.csv present")
    else:
        print("\nProxy eval: gold_labels_proxy.csv missing (run build_proxy_gold.py)")

    print("\n=== Summary ===")
    if warnings:
        print(f"Warnings ({len(warnings)}):")
        for w in warnings:
            print(f"  - {w}")
    if errors:
        print(f"\nErrors ({len(errors)}) — fix before submission:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)

    print("\nAll critical checks passed.")
    if warnings:
        print("Address warnings for best offline judging experience.")


if __name__ == "__main__":
    main()
