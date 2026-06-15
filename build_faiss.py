import pickle
import torch
import faiss
from pathlib import Path
BASE = Path(__file__).parent

print("Loading candidate_db.pkl...")
with open(BASE / "candidate_db.pkl", "rb") as f:
    db = pickle.load(f)
embeddings = db["embeddings"]
emb_np = embeddings.cpu().numpy()

print("Building FAISS index...")
d = emb_np.shape[1]
index = faiss.IndexFlatIP(d)
index.add(emb_np)

faiss.write_index(index, str(BASE / "faiss_index.bin"))
print("Done!")
