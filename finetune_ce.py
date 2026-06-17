import os
import pickle
import numpy as np
from pathlib import Path
import torch
from sentence_transformers import CrossEncoder, InputExample
from torch.utils.data import DataLoader
import random

BASE = Path(__file__).parent

# Import create_synthetic_target from train_reranker
import sys
sys.path.append(str(BASE))
from train_reranker import create_synthetic_target

def finetune_cross_encoder():
    meta_path = BASE / "candidate_meta.pkl"
    if not meta_path.exists():
        print(f"Error: {meta_path} not found.")
        return

    jd_path = BASE / "job_desc.txt"
    if not jd_path.exists():
        print(f"Error: {jd_path} not found.")
        return
        
    with open(jd_path, "r", encoding="utf-8") as f:
        jd_text = f.read()

    print("Loading candidate metadata...")
    with open(meta_path, "rb") as f:
        metadata = pickle.load(f)

    print("Generating training pairs using synthetic target scores...")
    train_examples = []
    
    random.seed(42)
    
    # We only take up to a limited number of items to fit in a short time if GPU is not available.
    sample_size = 2000 if torch.cuda.is_available() else 100
    
    # Sort candidates by synthetic score to ensure diverse sampling
    scored_cands = [(meta, create_synthetic_target(meta)) for meta in metadata]
    scored_cands.sort(key=lambda x: x[1], reverse=True)
    
    # Take top 50% and random 50% from the rest
    top_cands = scored_cands[:sample_size//2]
    bottom_cands = random.sample(scored_cands[sample_size//2:], min(sample_size//2, len(scored_cands) - sample_size//2))
    
    training_set = top_cands + bottom_cands
    
    for meta, target_score in training_set:
        cand_text = meta.get("doc_text", "")
        if not cand_text:
            continue
            
        cand_short = cand_text[:1500]
        jd_short = jd_text[:1500]
        
        train_examples.append(InputExample(texts=[jd_short, cand_short], label=target_score))

    print(f"Prepared {len(train_examples)} pairs.")
    
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=8)
    
    model_name = str(BASE / "models" / "ms-marco-MiniLM-L-6-v2")
    if not os.path.exists(model_name):
        model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        
    print(f"Loading base CrossEncoder: {model_name}")
    # Using num_labels=1 since we are doing regression
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
        use_amp=torch.cuda.is_available()
    )
    
    print(f"Fine-tuning complete. Model saved to {output_path}")

if __name__ == "__main__":
    finetune_cross_encoder()
