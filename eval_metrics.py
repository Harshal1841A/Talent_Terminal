"""Ranking evaluation metrics for proxy gold labels and submission CSVs."""

from __future__ import annotations

import numpy as np


def dcg_at_k(relevances: list[float], k: int) -> float:
    r = np.asarray(relevances, dtype=float)[:k]
    if r.size == 0:
        return 0.0
    return float(np.sum((2**r - 1) / np.log2(np.arange(2, r.size + 2))))


def ndcg_at_k(ranked_relevances: list[float], k: int, all_positive_relevances: list[float]) -> float:
    dcg_max = dcg_at_k(sorted(all_positive_relevances, reverse=True), k)
    if dcg_max == 0:
        return 0.0
    return dcg_at_k(ranked_relevances, k) / dcg_max


def average_precision(binary_relevance: list[int], k: int) -> float:
    r = np.asarray(binary_relevance[:k]) != 0
    if r.size == 0:
        return 0.0
    hits = [np.mean(r[: i + 1]) for i in range(r.size) if r[i]]
    return float(np.mean(hits)) if hits else 0.0


def mean_reciprocal_rank(binary_relevance: list[int]) -> float:
    try:
        first = binary_relevance.index(1) + 1
        return 1.0 / first
    except ValueError:
        return 0.0


def recall_at_k(ranked_ids: list[str], gold_positives: set[str], k: int) -> float:
    if not gold_positives:
        return 0.0
    hits = sum(1 for cid in ranked_ids[:k] if cid in gold_positives)
    return hits / len(gold_positives)


def evaluate_ranking(
    ranked_candidate_ids: list[str],
    gold_map: dict[str, int],
    k: int = 10,
) -> dict[str, float]:
    """
    Evaluate a ranked list against a gold relevance map (0–3 scale).
    Unlabeled candidates are treated as relevance 0.
    """
    ranked_relevance = [float(gold_map.get(cid, 0)) for cid in ranked_candidate_ids]
    all_positive = [float(v) for v in gold_map.values() if v > 0]
    binary = [1 if r > 0 else 0 for r in ranked_relevance]
    gold_3 = {cid for cid, rel in gold_map.items() if rel >= 3}
    gold_2plus = {cid for cid, rel in gold_map.items() if rel >= 2}

    return {
        f"ndcg@{k}": ndcg_at_k(ranked_relevance, k, all_positive),
        f"map@{k}": average_precision(binary, k),
        "mrr": mean_reciprocal_rank(binary),
        f"recall@{k}_rel3": recall_at_k(ranked_candidate_ids, gold_3, k),
        f"recall@{k}_rel2plus": recall_at_k(ranked_candidate_ids, gold_2plus, k),
    }
