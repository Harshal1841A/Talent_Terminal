"""
download_models.py — One-time model download (run with network access).

Downloads both models to local ./models/ subdirectories so that rank.py
can run fully offline in the Stage-3 judging sandbox (HF_HUB_OFFLINE=1).

Run once:
    python download_models.py

After this script completes, the ./models/ directory will contain:
    models/bge-base-en-v1.5/     (~440 MB)
    models/ms-marco-MiniLM-L-6-v2/  (~85 MB)

Total on disk: ~525 MB (well within the 5 GB artifact budget).
"""

import os
from pathlib import Path

BASE = Path(__file__).parent
MODELS_DIR = BASE / "models"
MODELS_DIR.mkdir(exist_ok=True)

BGE_LOCAL = str(MODELS_DIR / "bge-base-en-v1.5")
MINIML_LOCAL = str(MODELS_DIR / "ms-marco-MiniLM-L-6-v2")

print("=" * 60)
print("Talent Terminal — Model Downloader")
print("=" * 60)

# -- Download BGE Bi-Encoder -------------------------------------
if Path(BGE_LOCAL).exists() and any(Path(BGE_LOCAL).iterdir()):
    print(f"\n[1/2] BGE model already exists at {BGE_LOCAL} — skipping.")
else:
    print(f"\n[1/2] Downloading BAAI/bge-base-en-v1.5 -> {BGE_LOCAL}")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("BAAI/bge-base-en-v1.5")
    model.save(BGE_LOCAL)
    print(f"      Saved to {BGE_LOCAL}")

# -- Download CrossEncoder ---------------------------------------
if Path(MINIML_LOCAL).exists() and any(Path(MINIML_LOCAL).iterdir()):
    print(f"\n[2/2] CrossEncoder already exists at {MINIML_LOCAL} — skipping.")
else:
    print(f"\n[2/2] Downloading cross-encoder/ms-marco-MiniLM-L-6-v2 -> {MINIML_LOCAL}")
    from sentence_transformers import CrossEncoder
    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    ce.save(MINIML_LOCAL)
    print(f"      Saved to {MINIML_LOCAL}")

# -- Size check -------------------------------------------------
print("\n" + "=" * 60)
print("Disk size report:")

def dir_size_mb(path: str) -> float:
    total = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
    return total / (1024 ** 2)

for label, path in [("BGE model", BGE_LOCAL), ("MiniLM model", MINIML_LOCAL)]:
    if Path(path).exists():
        mb = dir_size_mb(path)
        print(f"  {label}: {mb:.1f} MB  ({path})")

# Check total artifact budget
pkl_files = ["candidate_meta.pkl", "faiss_index.bin", "bm25_index.pkl", "lgbm_reranker.pkl"]
pkl_total = sum(
    (BASE / f).stat().st_size for f in pkl_files if (BASE / f).exists()
) / (1024 ** 2)
model_total = dir_size_mb(str(MODELS_DIR)) if MODELS_DIR.exists() else 0
grand_total = pkl_total + model_total
print(f"\n  PKL artifacts: {pkl_total:.1f} MB")
print(f"  Models dir:    {model_total:.1f} MB")
print(f"  TOTAL:         {grand_total:.1f} MB  ({'[OK] under 5GB budget' if grand_total < 5000 else '[WARNING] exceeds 5GB!'})")
print("\n[SUCCESS] Models ready. rank.py will load them from ./models/ with HF_HUB_OFFLINE=1.")
