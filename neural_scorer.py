import torch
import torch.nn as nn
from pathlib import Path

class DimensionAwareScorer(nn.Module):
    def __init__(self, n_features=32, n_dims=5):
        super().__init__()
        # Projects classical features into JD dimensions
        self.dim_projector = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, n_dims)
        )
        # Learns importance of each dimension
        # Note: n_dims must be divisible by num_heads. So n_dims=5 and num_heads=1 (or n_dims=4 or pad n_dims)
        # Let's pad n_dims to 8 for attention, or just use 1 head if n_dims=5
        self.attention = nn.MultiheadAttention(embed_dim=n_dims, num_heads=1, batch_first=True)
        self.score_head = nn.Sequential(
            nn.Linear(n_dims, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, features, jd_dim_weights):
        # features: [batch, n_features]
        # jd_dim_weights: [batch, n_dims]
        candidate_dims = self.dim_projector(features)  # [batch, 5]
        
        # Cross-attention: queries=candidate_dims, keys/values=jd_dim_weights
        attended, _ = self.attention(
            candidate_dims.unsqueeze(1), 
            jd_dim_weights.unsqueeze(1),
            jd_dim_weights.unsqueeze(1)
        )
        score = self.score_head(attended.squeeze(1)).squeeze(-1)
        return score

def load_model(model_path=None):
    if model_path is None:
        model_path = Path(__file__).parent / "models" / "neural_scorer.pth"
    model = DimensionAwareScorer()
    if Path(model_path).exists():
        model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()
    return model
