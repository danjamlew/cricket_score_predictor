"""PyTorch dataset and loader factory."""
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .features import FeatureEncoder


class InningsStateDataset(Dataset):
    """One item per (match, ball): features plus the remaining runs to be scored."""

    def __init__(
        self, df: pd.DataFrame, encoder: FeatureEncoder, target_scale: float,
        training: bool = False,
    ):
        cats, num = encoder.transform(df, training=training)
        self.cats = torch.from_numpy(cats)
        self.num = torch.from_numpy(num)
        remaining = (df["final"] - df["runs"]).to_numpy(dtype="float32")
        self.target = torch.from_numpy(remaining / target_scale)

    def __len__(self) -> int:
        return len(self.target)

    def __getitem__(self, i: int):
        return self.cats[i], self.num[i], self.target[i]


def make_loader(
    df: pd.DataFrame, encoder: FeatureEncoder, batch_size: int, target_scale: float,
    shuffle: bool, training: bool = False,
) -> DataLoader:
    """`training=True` is for the encoder's own training matches (leave-one-match-out stats)."""
    return DataLoader(
        InningsStateDataset(df, encoder, target_scale, training), batch_size=batch_size,
        shuffle=shuffle,
    )
