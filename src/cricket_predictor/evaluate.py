"""Test-set evaluation against simple baselines, overall and by stage of the innings."""
import numpy as np
import pandas as pd

from .data.split import split_data
from .predict import predict_final_scores
from .train import load_checkpoint, load_table

BUCKETS = [(0, 0, "ball 0"), (1, 30, "overs 1-5"), (31, 60, "overs 6-10"),
           (61, 90, "overs 11-15"), (91, 119, "overs 16-19")]


def _metrics(final: pd.Series, pred: pd.Series) -> dict:
    err = pred - final
    return {"MAE": err.abs().mean(), "RMSE": float(np.sqrt((err**2).mean()))}


def method_predictions(
    model, encoder, ckpt_cfg: dict, train_df: pd.DataFrame, df: pd.DataFrame
) -> dict[str, pd.Series]:
    """Final-total predictions for each method, indexed like `df`."""
    totals = train_df.groupby("match_id").agg(venue=("venue", "first"), final=("final", "first"))
    global_mean = totals["final"].mean()
    venue_mean = df["venue"].map(totals.groupby("venue")["final"].mean()).fillna(global_mean)
    crr = np.where(df["balls"] > 0, df["runs"] / df["balls"].clip(lower=1) * 120, global_mean)
    return {
        "network": predict_final_scores(model, encoder, df, ckpt_cfg["train"]["target_scale"]),
        "run-rate x 20": pd.Series(crr, index=df.index),
        "venue mean": venue_mean,
    }


def evaluate(cfg: dict, checkpoint: str) -> pd.DataFrame:
    model, encoder, ckpt_cfg = load_checkpoint(checkpoint)
    df = load_table(cfg)
    train_df, _, test_df = split_data(df, ckpt_cfg["split"])  # the split the model was trained with
    if test_df.empty:
        raise ValueError("Test split is empty; adjust the split settings")
    preds = method_predictions(model, encoder, ckpt_cfg, train_df, test_df)

    rows = []
    for lo, hi, label in BUCKETS + [(0, 119, "all (balls 0-119)")]:
        mask = test_df["balls"].between(lo, hi)
        for name, pred in preds.items():
            rows.append({"stage": label, "model": name,
                         **_metrics(test_df.loc[mask, "final"], pred[mask])})
    return pd.DataFrame(rows)
