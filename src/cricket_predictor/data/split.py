"""Train/val/test splits by match (all rows of a match always stay together)."""
import numpy as np
import pandas as pd


def split_by_year(
    df: pd.DataFrame, train_until: int, val_year: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological: train <= train_until, validate on val_year, test on later years."""
    if train_until >= val_year:
        raise ValueError("train_until must be earlier than val_year")
    return df[df["year"] <= train_until], df[df["year"] == val_year], df[df["year"] > val_year]


def split_stratified_by_year(
    df: pd.DataFrame, val_frac: float, test_frac: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Random split of matches, done separately within each year so every season is
    represented in train, validation and test in the same proportions."""
    if not 0 < val_frac + test_frac < 1:
        raise ValueError("val_frac + test_frac must be between 0 and 1")
    rng = np.random.default_rng(seed)
    val_ids: list[str] = []
    test_ids: list[str] = []
    for _, year_df in df.groupby("year"):
        ids = np.sort(year_df["match_id"].unique())  # sorted so the result ignores row order
        rng.shuffle(ids)
        n_val, n_test = round(len(ids) * val_frac), round(len(ids) * test_frac)
        val_ids += list(ids[:n_val])
        test_ids += list(ids[n_val : n_val + n_test])
    is_val, is_test = df["match_id"].isin(val_ids), df["match_id"].isin(test_ids)
    return df[~is_val & ~is_test], df[is_val], df[is_test]


def split_data(
    df: pd.DataFrame, split_cfg: dict
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Dispatch on split.mode ('chronological' if absent, as in experiments recorded earlier)."""
    mode = split_cfg.get("mode", "chronological")
    if mode == "chronological":
        return split_by_year(df, split_cfg["train_until"], split_cfg["val_year"])
    if mode == "stratified_year":
        return split_stratified_by_year(
            df, split_cfg["val_frac"], split_cfg["test_frac"], split_cfg["seed"]
        )
    raise ValueError(f"Unknown split.mode '{mode}'")
