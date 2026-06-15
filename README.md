---
title: Talent Terminal
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

# 🚀 Talent Terminal

A ruthless, highly-optimized candidate ranking pipeline for the India Runs Data and AI Challenge. Filters and ranks 100K+ profiles using dense retrieval, cross-encoder re-ranking, and strict behavioral heuristics.

**Performance Constraints:** ≤5min execution, ≤16GB RAM, CPU-only. Fully offline.

## ⚙️ Architecture

1. **Retrieval (`BAAI/bge-base-en-v1.5`):** Embeds JD and filters top 800 candidates via cosine similarity against precomputed embeddings.
2. **Re-ranking (`cross-encoder/ms-marco-MiniLM-L-6-v2`):** Deep semantic matching on top candidates (batch size 128 to prevent OOM).
3. **Heuristics & Fusion:** Punishes "title chasers", scores location matching, applies ML behavioral signals, and fuses scores additively.

## 🚀 Usage

### 1. Precompute (Offline Generation)
```bash
python precompute.py
```
*(Takes up to 24h depending on hardware. Generates required `candidate_db.pkl` & `bm25_index.pkl`)*

### 2. Rank (Production)
```bash
python rank.py
```
*(Executes the pipeline in <5 mins, outputs `submission.csv`)*

### 3. Deploy (Hugging Face Spaces)
The repository intentionally excludes all models, `*.pkl`, and `.csv` artifacts to prevent bloat. Do not push large models to GitHub. 

To deploy:
1. Push only the code.
2. Download models (`models/`) and run `precompute.py` on the target server.
3. Hugging Face Spaces will automatically launch the Gradio app (`app.py`). Requires 16GB+ RAM.

## 🧪 Evaluation
```bash
python self_eval.py --submission submission.csv --gold gold_labels.csv --k 10
```
