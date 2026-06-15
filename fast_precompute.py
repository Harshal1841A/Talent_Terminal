import json, pickle
from pathlib import Path
from tqdm import tqdm
from precompute import extract_features, build_weighted_doc

BASE = Path(__file__).parent

print("Reading candidates.jsonl...")
with open(BASE / "candidates.jsonl", "r", encoding="utf-8") as f:
    lines = f.readlines()

print("Loading existing embeddings from candidate_db.pkl...")
with open(BASE / "candidate_db.pkl", "rb") as f:
    db = pickle.load(f)

print(f"Extracting features for {len(lines)} candidates...")
all_meta = []
for line in tqdm(lines, desc="Feature extraction"):
    c = json.loads(line)
    meta = extract_features(c)
    doc = build_weighted_doc(c)
    meta["doc_text"] = doc
    all_meta.append(meta)

db["metadata"] = all_meta

print("Saving candidate_db.pkl...")
with open(BASE / "candidate_db.pkl", "wb") as f:
    pickle.dump(db, f, protocol=pickle.HIGHEST_PROTOCOL)

print("Done! Updated metadata without recomputing 100K embeddings.")
