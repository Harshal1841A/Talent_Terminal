import pickle
import json
import faiss
from pathlib import Path
from precompute import extract_features, build_weighted_doc

BASE = Path(__file__).parent

def main():
    print("Reading candidates.jsonl to regenerate metadata...")
    all_meta = []
    with open(BASE / "candidates.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            meta = extract_features(c)
            doc = build_weighted_doc(c)
            meta["doc_text"] = doc
            all_meta.append(meta)
    
    print("Saving candidate_meta.pkl...")
    with open(BASE / "candidate_meta.pkl", "wb") as f:
        pickle.dump(all_meta, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    print("Loading candidate_db.pkl to get existing embeddings...")
    with open(BASE / "candidate_db.pkl", "rb") as f:
        db = pickle.load(f)
        
    embeddings = db["embeddings"]
    emb_np = embeddings.cpu().numpy()
    
    print("Building FAISS index...")
    d = emb_np.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(emb_np)
    
    faiss.write_index(index, str(BASE / "faiss_index.bin"))
    print("Migration complete!")

if __name__ == "__main__":
    main()
