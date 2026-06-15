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

# 🚀 Talent Terminal: Candidate Ranking Pipeline

Talent Terminal is a high-performance candidate-ranking system designed to filter and rank 100K+ profiles for specialized roles (e.g., Senior AI Engineer) under strict performance constraints (≤5min execution, ≤16GB RAM, CPU-only).

## 🛠 Pipeline Architecture

The ranking pipeline executes through three heavily optimized stages:

1. **Stage 1 (Dense Retrieval):** 
   - Uses `BAAI/bge-base-en-v1.5` (offline) to embed the Job Description (JD) and compute cosine similarities against precomputed candidate embeddings.
   - Retrieves the top 800 candidates to balance high recall with latency limits.

2. **Stage 2 (Cross-Encoder Re-ranking):** 
   - Uses `cross-encoder/ms-marco-MiniLM-L-6-v2` (offline) to deeply analyze the semantic match between the JD and the top candidate profiles.
   - Batch size is optimized (128) to prevent OOM errors on 16GB systems.

3. **Stage 3 (Behavioral Heuristics & Fusion):** 
   - Applies domain-specific expert heuristics: `location_score`, `ml_role_ratio`, `title_chaser` penalty, and honeypot detection.
   - Fuses semantic scores, heuristic scores, and optional LightGBM behavioral scores using an additive approach with proper scaling constraints.
   - Final output: Generates a `submission.csv` (Top 100) and a `dashboard.html` visual report.

## 🚀 Execution & Deployment

### Local Offline Execution (Sandbox/Air-gapped)

The system is configured to run fully offline without any network dependency.

1. **Prerequisites:**
   Ensure `BAAI/bge-base-en-v1.5` and `cross-encoder/ms-marco-MiniLM-L-6-v2` are downloaded to the local `./models` directory (using the offline model downloader script if necessary).

2. **Run the Precompute (One-time):**
   ```bash
   python precompute.py
   ```
   *Note: This generates `candidate_db.pkl` and `bm25_index.pkl`.*

3. **Run the Ranking (Production):**
   ```bash
   python rank.py
   ```
   *Guaranteed to finish in ≤5 minutes on standard hardware.*

### Hugging Face Spaces Deployment

To deploy the Gradio UI (`app.py`) to Hugging Face Spaces:

1. **Artifact Exclusion:**
   Artifacts and models are extremely large. They are intentionally ignored via `.gitignore` to prevent repository bloat. DO NOT attempt to push them via Git LFS to standard GitHub as they exceed quotas.
   You must regenerate the data (`python precompute.py`) or download the models on your deployment server directly.

2. **Hardware Requirements:**
   - **RAM:** Minimum 16GB required. If Stage 2 runs out of memory (OOM), request a CPU-upgrade tier (32GB RAM).
   - **Startup Time:** Initial startup may take ~90 seconds due to memory mapping of the 1GB+ artifact file (`candidate_db.pkl`).

3. **Launch:**
   Once pushed, Hugging Face will automatically detect `app.py` and launch the Gradio dashboard interface for interactive candidate filtering.

## 🧪 Evaluation

To measure the effectiveness of the ranking against a known gold-standard subset:
```bash
python self_eval.py --submission submission.csv --gold gold_labels.csv --k 10
```
This computes NDCG@10, MAP@10, and MRR. If a gold labels file isn't provided, it will generate a synthetic one for demonstration purposes.
