"""Chronological train/val/test split by match (all rows of a match stay together)."""
import pandas as pd


def split_by_year(
    df: pd.DataFrame, train_until: int, val_year: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df[df["year"] <= train_until]
    val = df[df["year"] == val_year]
    test = df[df["year"] > val_year]
    if train_until >= val_year:
        raise ValueError("train_until must be earlier than val_year")
    return train, val, test
