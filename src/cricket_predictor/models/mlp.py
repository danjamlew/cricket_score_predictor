"""MLP with categorical embeddings that predicts remaining runs (scaled)."""
import torch
from torch import nn


class ScorePredictor(nn.Module):
    def __init__(
        self, cardinalities: list[int], num_features: int,
        embed_dim: int = 8, hidden_dims: tuple[int, ...] = (128, 64, 32), dropout: float = 0.1,
    ):
        super().__init__()
        self.embeddings = nn.ModuleList(nn.Embedding(c, embed_dim) for c in cardinalities)
        layers: list[nn.Module] = []
        width = embed_dim * len(cardinalities) + num_features
        for h in hidden_dims:
            layers += [nn.Linear(width, h), nn.ReLU(), nn.Dropout(dropout)]
            width = h
        layers.append(nn.Linear(width, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, cats: torch.Tensor, num: torch.Tensor) -> torch.Tensor:
        emb = [e(cats[:, i]) for i, e in enumerate(self.embeddings)]
        x = torch.cat(emb + [num], dim=1)
        # Softplus keeps predicted remaining runs non-negative (final >= current score).
        return nn.functional.softplus(self.mlp(x)).squeeze(1)
