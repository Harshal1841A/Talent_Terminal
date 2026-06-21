import os
import pickle
import random
from pathlib import Path

import torch
from sentence_transformers import CrossEncoder, InputExample
from torch.utils.data import DataLoader

BASE = Path(__file__).parent.parent

MIN_TRAIN_PAIRS = 5000


def finetune_cross_encoder():
    """
    Optional CE fine-tune — guarded.

    Requires CUDA and >= 5000 JD–candidate pairs. Synthetic heuristic labels
    do not replace MS MARCO relevance; skipping is the default safe path.
    Submission uses base ms-marco-MiniLM-L-6-v2 (see rank.py / app.py).
    """
    meta_path = BASE / "candidate_meta.pkl"
    if not meta_path.exists():
        print(f"Error: {meta_path} not found.")
        return

    jd_path = BASE / "job_desc.txt"
    if not jd_path.exists():
        print(f"Error: {jd_path} not found.")
        return

    if not torch.cuda.is_available():
        print(
            "Skipping CE fine-tune: no CUDA GPU. "
            "Base ms-marco-MiniLM-L-6-v2 (39M MS MARCO pairs) is stronger than "
            "fine-tuning on synthetic heuristic labels on CPU."
        )
        return

    with open(jd_path, "r", encoding="utf-8") as f:
        jd_text = f.read()

    print("Loading candidate metadata...")
    with open(meta_path, "rb") as f:
        metadata = pickle.load(f)

    random.seed(42)
    train_examples = []
    jd_short = jd_text[:1500]

    for meta in metadata:
        cand_text = meta.get("doc_text", "")
        if not cand_text:
            continue
        train_examples.append(
            InputExample(texts=[jd_short, cand_text[:1500]], label=0.0)
        )

    n_pairs = len(train_examples)
    if n_pairs < MIN_TRAIN_PAIRS:
        print(
            f"Skipping CE fine-tune: only {n_pairs} pairs (need >= {MIN_TRAIN_PAIRS}). "
            "Use base ms-marco for submission."
        )
        return

    print(
        "WARNING: finetune_ce.py uses placeholder labels (0.0). "
        "For real lift you need human/LLM relevance labels per JD–candidate pair. "
        "Proceeding only because GPU + pair-count guards passed."
    )

    train_dataloader = DataLoader(train_examples[:MIN_TRAIN_PAIRS], shuffle=True, batch_size=16)

    model_name = str(BASE / "models" / "ms-marco-MiniLM-L-6-v2")
    if not os.path.exists(model_name):
        model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    print(f"Loading base CrossEncoder: {model_name}")
    model = CrossEncoder(model_name, num_labels=1)

    print("Starting Fine-Tuning (1 epoch)...")
    epochs = 1
    warmup_steps = int(len(train_dataloader) * epochs * 0.1)

    output_path = str(BASE / "models" / "finetuned-ce-model")
    os.makedirs(str(BASE / "checkpoints"), exist_ok=True)
    model.fit(
        train_dataloader=train_dataloader,
        epochs=epochs,
        warmup_steps=warmup_steps,
        output_path=output_path,
        use_amp=True,
    )

    print(f"Fine-tuning complete. Model saved to {output_path}")


if __name__ == "__main__":
    finetune_cross_encoder()
