"""Feature construction: categorical encoders and a numeric scaler fit on train only."""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

CAT_COLS = ["venue", "bat_team", "bowl_team"]
NUM_COLS = [
    "runs", "wickets", "balls", "balls_left", "run_rate",
    "runs_last12", "runs_last30", "wkts_last30", "year_offset",
]
TOTAL_BALLS = 120
BASE_YEAR = 2008


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Add balls_left, run_rate and year_offset. Works on a full table or a one-row frame."""
    df = df.copy()
    df["balls_left"] = TOTAL_BALLS - df["balls"]
    df["run_rate"] = np.where(df["balls"] > 0, df["runs"] / df["balls"].clip(lower=1) * 6, 0.0)
    df["year_offset"] = df["year"] - BASE_YEAR
    return df


@dataclass
class FeatureEncoder:
    """Index 0 of every categorical vocabulary is reserved for unseen values."""

    vocabs: dict[str, dict[str, int]] = field(default_factory=dict)
    mean: np.ndarray | None = None
    std: np.ndarray | None = None

    def fit(self, df: pd.DataFrame) -> "FeatureEncoder":
        for col in CAT_COLS:
            self.vocabs[col] = {v: i + 1 for i, v in enumerate(sorted(df[col].unique()))}
        num = add_derived(df)[NUM_COLS].to_numpy(dtype=np.float32)
        self.mean = num.mean(axis=0)
        self.std = num.std(axis=0) + 1e-6
        return self

    @property
    def cardinalities(self) -> list[int]:
        return [len(self.vocabs[c]) + 1 for c in CAT_COLS]

    def transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return (categorical int64 [n, 3], numeric float32 [n, len(NUM_COLS)])."""
        df = add_derived(df)
        cats = np.stack(
            [df[c].map(self.vocabs[c]).fillna(0).astype(np.int64).to_numpy() for c in CAT_COLS],
            axis=1,
        )
        num = (df[NUM_COLS].to_numpy(dtype=np.float32) - self.mean) / self.std
        return cats, num.astype(np.float32)

    def state_dict(self) -> dict:
        return {"vocabs": self.vocabs, "mean": self.mean, "std": self.std}

    @classmethod
    def from_state_dict(cls, state: dict) -> "FeatureEncoder":
        return cls(vocabs=state["vocabs"], mean=state["mean"], std=state["std"])
