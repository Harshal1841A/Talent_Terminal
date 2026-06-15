import os
import argparse
import pandas as pd
import numpy as np

def dcg_at_k(r, k):
    """Discounted Cumulative Gain at K"""
    r = np.asfarray(r)[:k]
    if r.size:
        return np.sum((2**r - 1) / np.log2(np.arange(2, r.size + 2)))
    return 0.

def ndcg_at_k(r, k, ground_truth):
    """Normalized Discounted Cumulative Gain at K"""
    dcg_max = dcg_at_k(sorted(ground_truth, reverse=True), k)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k) / dcg_max

def average_precision(r):
    """Average Precision"""
    r = np.asarray(r) != 0
    out = [np.mean(r[:i+1]) for i in range(r.size) if r[i]]
    if not out:
        return 0.
    return np.mean(out)

def calculate_metrics(submission_file, gold_file, k=10):
    print(f"Loading submission: {submission_file}")
    sub_df = pd.read_csv(submission_file)
    
    if not os.path.exists(gold_file):
        print(f"Gold file '{gold_file}' not found. Generating a synthetic one for demonstration...")
        # Synthesize a gold file for demonstration (random subset of the top candidates)
        gold_df = pd.DataFrame({
            'candidate_id': sub_df['candidate_id'].sample(n=min(50, len(sub_df)), random_state=42),
            'relevance': np.random.choice([1, 2, 3], size=min(50, len(sub_df)), p=[0.5, 0.3, 0.2])
        })
        gold_df.to_csv(gold_file, index=False)
        print(f"Saved synthetic gold labels to {gold_file}.")
    else:
        gold_df = pd.read_csv(gold_file)
    
    gold_map = dict(zip(gold_df['candidate_id'], gold_df['relevance']))
    
    # Map ranked candidates to their relevance scores
    ranked_relevance = [gold_map.get(cid, 0) for cid in sub_df['candidate_id']]
    
    # All non-zero relevance scores in the gold set
    all_relevance_scores = [rel for rel in gold_map.values() if rel > 0]
    
    ndcg = ndcg_at_k(ranked_relevance, k, all_relevance_scores)
    
    # Binarize relevance for MAP and MRR (relevance > 0 is relevant)
    binary_relevance = [1 if r > 0 else 0 for r in ranked_relevance]
    ap = average_precision(binary_relevance[:k])
    
    # Mean Reciprocal Rank
    try:
        first_relevant_idx = binary_relevance.index(1) + 1
        mrr = 1.0 / first_relevant_idx
    except ValueError:
        mrr = 0.0
        
    print("\n=== Evaluation Results ===")
    print(f"NDCG@{k}: {ndcg:.4f}")
    print(f"MAP@{k}:  {ap:.4f}")
    print(f"MRR:     {mrr:.4f}")
    print("==========================")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ranking pipeline output against a gold set.")
    parser.add_argument("--submission", default="submission.csv", help="Path to the ranking submission CSV.")
    parser.add_argument("--gold", default="gold_labels.csv", help="Path to the gold labels CSV.")
    parser.add_argument("--k", type=int, default=10, help="Cutoff K for metrics (e.g., NDCG@K).")
    args = parser.parse_args()
    
    if not os.path.exists(args.submission):
        print(f"Error: Submission file '{args.submission}' not found. Please run rank.py first.")
    else:
        calculate_metrics(args.submission, args.gold, args.k)
