import pickle
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import faiss
import numpy as np

# Mocking the SentenceTransformers for fast testing without huge models loaded
class MockEncoder:
    def encode(self, texts, **kwargs):
        # Return random embeddings
        return torch.randn(1, 384)
        
    def predict(self, pairs, **kwargs):
        # Return random scores
        return [0.5] * len(pairs)

import torch
from ranking_pipeline import rank_candidates_core, RankingConfig

app = FastAPI(title="Multi-Agent AI Recruiter")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE = Path(__file__).parent

print("Loading metadata...")
with open(BASE / "candidate_meta.pkl", "rb") as f:
    metadata_dict = pickle.load(f)
metadata = list(metadata_dict.values())

print("Skipping Faiss index to save memory.")
faiss_index = None

bi_enc = MockEncoder()
cross_enc = MockEncoder()

class RankRequest(BaseModel):
    jd_text: str

@app.post("/api/rank")
def rank_api(req: RankRequest):
    results = []
    for i, m in enumerate(metadata[:100]):
        results.append({
            "candidate_id": m["candidate_id"],
            "rank": i + 1,
            "score": round(np.random.uniform(0.7, 0.95), 3),
            "reasoning": f"Dummy reasoning for candidate {i}",
            "meta": m,
            "breakdown": {
                "semantic": 0.8,
                "location": 0.9,
                "ml_ratio": 0.7,
                "experience": 0.85,
                "company": 0.9,
                "ml_signals": 0.8,
                "behavioral": 0.85,
                "recency": 0.95,
                "jd_terms": 0.88,
                "elite_co": 0.8,
                "github": 0.9,
                "assessment": 0.85
            }
        })
        
    return {"status": "success", "results": results}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
