"""Inference from a checkpoint."""
import pandas as pd
import torch

from .data.normalise import canonical_team, canonical_venue
from .models.mlp import ScorePredictor
from .data.features import FeatureEncoder


@torch.no_grad()
def predict_final_scores(
    model: ScorePredictor, encoder: FeatureEncoder, df: pd.DataFrame, target_scale: float
) -> pd.Series:
    """Predicted final totals for a frame of innings states (needs the parse.py columns)."""
    cats, num = encoder.transform(df)
    remaining = model(torch.from_numpy(cats), torch.from_numpy(num)).numpy() * target_scale
    return pd.Series(df["runs"].to_numpy() + remaining, index=df.index)


def predict_state(
    model: ScorePredictor, encoder: FeatureEncoder, cfg: dict, *, runs: int, wickets: int,
    balls: int, venue: str, bat_team: str, bowl_team: str, year: int = 2026,
    runs_last12: int | None = None, runs_last30: int | None = None,
    wkts_last30: int | None = None,
) -> float:
    """Predict the final total from a single match state.

    Recent-form inputs default to what the run rate so far implies if not supplied.
    """
    rate = runs / balls if balls else 0.0
    row = pd.DataFrame([{
        "venue": canonical_venue(venue), "bat_team": canonical_team(bat_team),
        "bowl_team": canonical_team(bowl_team), "year": year, "balls": balls, "runs": runs,
        "wickets": wickets,
        "runs_last12": runs_last12 if runs_last12 is not None else round(rate * min(balls, 12)),
        "runs_last30": runs_last30 if runs_last30 is not None else round(rate * min(balls, 30)),
        "wkts_last30": wkts_last30 if wkts_last30 is not None else round(
            wickets * min(balls, 30) / balls) if balls else 0,
    }])
    return float(predict_final_scores(model, encoder, row, cfg["train"]["target_scale"]).iloc[0])
