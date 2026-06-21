import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
from core_scoring import build_features

BASE = Path(__file__).parent
META_PATH = BASE / "candidate_meta.pkl"
MODEL_DIR = BASE / "models"
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "neural_scorer.pth"

def train():
    from neural_scorer import DimensionAwareScorer
    from jd_parser import get_dimension_weights
    
    print("Loading metadata...")
    with open(META_PATH, "rb") as f:
        metadata = pickle.load(f)
        
    print("Building dataset...")
    X = []
    y = []
    
    weights_dict = get_dimension_weights()
    jd_weights = [
        weights_dict["technical_depth"],
        weights_dict["product_shipping"],
        weights_dict["startup_fitness"],
        weights_dict["experience_band"],
        weights_dict["location"]
    ]
    
    for meta in tqdm(metadata):
        if meta.get("honeypot") or meta.get("wrong_title"):
            continue
        feats = build_features(meta)
        X.append(feats)
        
        # Soft target derived from LLM features for demo purposes
        # (Assuming build_features returns [..., llm_overall, llm_tech, llm_prod, llm_start])
        # Indices for llm features: 28, 29, 30, 31
        llm_tech = feats[29]
        llm_prod = feats[30]
        llm_start = feats[31]
        
        target = (llm_tech * jd_weights[0] + 
                  llm_prod * jd_weights[1] + 
                  llm_start * jd_weights[2])
        # scale up roughly to ranking range
        y.append(target * 20.0)
        
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    jd_w = torch.tensor([jd_weights], dtype=torch.float32).repeat(X.shape[0], 1)
    
    model = DimensionAwareScorer(n_features=X.shape[1], n_dims=5)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    print("Training neural scorer...")
    dataset = torch.utils.data.TensorDataset(X, jd_w, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=True)
    
    model.train()
    for epoch in range(5):
        epoch_loss = 0.0
        for batch_X, batch_jd, batch_y in loader:
            optimizer.zero_grad()
            preds = model(batch_X, batch_jd)
            loss = criterion(preds.unsqueeze(1), batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1} Loss: {epoch_loss/len(loader):.4f}")
        
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train()
