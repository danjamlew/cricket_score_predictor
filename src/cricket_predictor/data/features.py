"""Feature construction: categorical encoders, optional feature groups, numeric scaler.

Everything that is learned from data (vocabularies, scaler, venue statistics, player career
rates) is fit on training matches only. Venue statistics for training rows are "ordered": a
training match only sees the matches at its venue played before it (`transform(...,
training=True)`). Leave-one-out would be wrong here: with the venue embedding identifying the
venue, the network could recover each match's own final total from the leave-one-out mean.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .players import PlayerStats

CAT_COLS = ["venue", "bat_team", "bowl_team"]
BASE_NUM_COLS = [
    "runs", "wickets", "balls", "balls_left", "run_rate",
    "runs_last12", "runs_last30", "wkts_last30", "year_offset",
]
GROUP_NUM_COLS = {
    "state_extras": [
        "is_powerplay", "is_death", "resources", "balls_since_wkt", "runs_since_wkt",
        "boundaries_last12", "boundaries_last30", "dots_last12", "dots_last30", "extras_last30",
    ],
    "venue_era": [
        "impact_era", "toss_won_by_batting", "venue_mean_final", "venue_par_runs",
        "runs_vs_par", "prev_season_mean",
    ],
    "players": ["striker_sr", "nonstriker_sr", "yet_to_bat_sr", "bowler_econ", "attack_econ"],
}
TOTAL_BALLS = 120
BASE_YEAR = 2008
IMPACT_PLAYER_YEAR = 2023
SHRINK = 10.0  # pseudo-matches of league-average mixed into venue statistics
LATEST = "9999-12-31"  # query date meaning "use all history" (inference)


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Add columns that are functions of the state alone. Works on one row or a full table."""
    df = df.copy()
    df["balls_left"] = TOTAL_BALLS - df["balls"]
    df["run_rate"] = np.where(df["balls"] > 0, df["runs"] / df["balls"].clip(lower=1) * 6, 0.0)
    df["year_offset"] = df["year"] - BASE_YEAR
    df["is_powerplay"] = (df["balls"] < 36).astype(float)
    df["is_death"] = (df["balls"] >= 96).astype(float)
    df["resources"] = (10 - df["wickets"]) * df["balls_left"] / 1200
    df["impact_era"] = (df["year"] >= IMPACT_PLAYER_YEAR).astype(float)
    return df


def _ids(s: str) -> list[str]:
    return [p for p in s.split("|") if p]


@dataclass
class FeatureEncoder:
    """Index 0 of every categorical vocabulary is reserved for unseen values."""

    vocabs: dict[str, dict[str, int]] = field(default_factory=dict)
    mean: np.ndarray | None = None
    std: np.ndarray | None = None
    groups: list[str] = field(default_factory=list)
    venue: dict | None = None  # venue_era statistics
    _ordered: dict | None = field(default=None, repr=False)  # fit-time only, not persisted
    player_stats: PlayerStats | None = None

    @property
    def num_cols(self) -> list[str]:
        return BASE_NUM_COLS + [c for g in self.groups for c in GROUP_NUM_COLS[g]]

    @property
    def cardinalities(self) -> list[int]:
        return [len(self.vocabs[c]) + 1 for c in CAT_COLS]

    # ---- fitting ----
    def fit(
        self, df: pd.DataFrame, groups: tuple[str, ...] | list[str] = (),
        player_table: pd.DataFrame | None = None,
    ) -> "FeatureEncoder":
        unknown = set(groups) - set(GROUP_NUM_COLS)
        if unknown:
            raise ValueError(f"Unknown feature group(s) {sorted(unknown)}; "
                             f"choose from {sorted(GROUP_NUM_COLS)}")
        self.groups = list(groups)
        for col in CAT_COLS:
            self.vocabs[col] = {v: i + 1 for i, v in enumerate(sorted(df[col].unique()))}
        if "venue_era" in self.groups:
            self._fit_venue(df)
        if "players" in self.groups:
            if player_table is None:
                raise ValueError("The 'players' group needs the player table; run `prepare`")
            train_ids = set(df["match_id"].unique())
            self.player_stats = PlayerStats(player_table[player_table["match_id"].isin(train_ids)])
        num = self._numeric(df, training=True)
        self.mean = np.nanmean(num, axis=0).astype(np.float32)
        self.std = (np.nanstd(num, axis=0) + 1e-6).astype(np.float32)
        return self

    def _fit_venue(self, df: pd.DataFrame) -> None:
        matches = df.groupby("match_id").agg(
            venue=("venue", "first"), year=("year", "first"), final=("final", "first"),
            date=("date", "first"))
        card = len(self.vocabs["venue"]) + 1
        vi = matches["venue"].map(self.vocabs["venue"]).to_numpy()
        final_sum, final_n = np.zeros(card), np.zeros(card)
        np.add.at(final_sum, vi, matches["final"].to_numpy(float))
        np.add.at(final_n, vi, 1)
        # runs after b balls at each venue, over innings that reached ball b
        row_vi = df["venue"].map(self.vocabs["venue"]).to_numpy()
        balls = np.minimum(df["balls"].to_numpy(int), TOTAL_BALLS)  # one innings has 121
        runs_sum, runs_n = np.zeros((card, TOTAL_BALLS + 1)), np.zeros((card, TOTAL_BALLS + 1))
        np.add.at(runs_sum, (row_vi, balls), df["runs"].to_numpy(float))
        np.add.at(runs_n, (row_vi, balls), 1)
        n_by_ball = runs_n.sum(0)  # balls with no observations are filled from the previous ball
        ball_mean = pd.Series(np.where(n_by_ball > 0, runs_sum.sum(0) / np.maximum(n_by_ball, 1),
                                       np.nan))
        season = matches.groupby("year")["final"].mean()
        self._fit_ordered(df, matches)
        self.venue = {
            "final_sum": final_sum, "final_n": final_n, "runs_sum": runs_sum, "runs_n": runs_n,
            "ball_mean": ball_mean.ffill().bfill().to_numpy(),
            "global_final": float(matches["final"].mean()),
            "season_years": season.index.to_numpy(), "season_means": season.to_numpy(),
        }

    def _fit_ordered(self, df: pd.DataFrame, matches: pd.DataFrame) -> None:
        """For each training match, the venue statistics over earlier matches at its venue."""
        per_match = {}
        for mid, g in df.groupby("match_id"):
            runs, seen = np.zeros(TOTAL_BALLS + 1), np.zeros(TOTAL_BALLS + 1)
            idx = np.minimum(g["balls"].to_numpy(int), TOTAL_BALLS)
            runs[idx], seen[idx] = g["runs"].to_numpy(float), 1
            per_match[mid] = (runs, seen)
        ids, f_sum, f_n, r_sum, r_n = [], [], [], [], []
        for _, venue_matches in matches.groupby("venue"):
            tot_f = tot_n = 0.0
            tot_r, tot_rn = np.zeros(TOTAL_BALLS + 1), np.zeros(TOTAL_BALLS + 1)
            for mid, m in venue_matches.reset_index().sort_values(["date", "match_id"]).set_index(
                    "match_id").iterrows():
                ids.append(mid)
                f_sum.append(tot_f), f_n.append(tot_n), r_sum.append(tot_r.copy())
                r_n.append(tot_rn.copy())
                tot_f += m["final"]
                tot_n += 1
                tot_r += per_match[mid][0]
                tot_rn += per_match[mid][1]
        self._ordered = {"index": {mid: i for i, mid in enumerate(ids)},
                         "final_sum": np.array(f_sum), "final_n": np.array(f_n),
                         "runs_sum": np.array(r_sum), "runs_n": np.array(r_n)}

    # ---- features ----
    def _venue_features(self, d: pd.DataFrame, training: bool) -> dict[str, np.ndarray]:
        v = self.venue
        vi = d["venue"].map(self.vocabs["venue"]).fillna(0).astype(int).to_numpy()
        b = np.minimum(d["balls"].to_numpy(int), TOTAL_BALLS)  # a miscounted over can give 121
        runs = d["runs"].to_numpy(float)
        if training:  # only the matches at this venue played before the row's own match
            o = self._ordered
            m = d["match_id"].map(o["index"]).to_numpy(int)
            f_sum, f_n, r_sum, r_n = (o["final_sum"][m], o["final_n"][m],
                                      o["runs_sum"][m, b], o["runs_n"][m, b])
        else:
            f_sum, f_n = v["final_sum"][vi], v["final_n"][vi]
            r_sum, r_n = v["runs_sum"][vi, b], v["runs_n"][vi, b]
        mean_final = (f_sum + SHRINK * v["global_final"]) / (f_n + SHRINK)
        par = (r_sum + SHRINK * v["ball_mean"][b]) / (r_n + SHRINK)
        # previous season's mean total (never includes the current season)
        k = np.searchsorted(v["season_years"], d["year"].to_numpy(), side="left") - 1
        prev = np.where(k >= 0, v["season_means"][np.maximum(k, 0)], v["global_final"])
        return {"venue_mean_final": mean_final, "venue_par_runs": par,
                "runs_vs_par": runs - par, "prev_season_mean": prev}

    def _player_features(self, d: pd.DataFrame) -> dict[str, np.ndarray]:
        """Career rates before the match date. Missing id columns leave NaN (filled later)."""
        ps, n = self.player_stats, len(d)
        out = {c: np.full(n, np.nan) for c in GROUP_NUM_COLS["players"]}
        dates = d["date"].to_numpy() if "date" in d else np.full(n, LATEST)

        def col(name: str):
            return d[name].fillna("").to_numpy() if name in d else None

        striker, non_striker, bowler = col("striker_id"), col("nonstriker_id"), col("bowler_id")
        yet_to_bat, bowl_xi = col("yet_to_bat"), col("bowl_xi")
        sr_cache: dict = {}
        econ_cache: dict = {}
        group_cache: dict = {}

        def sr(pid, date):
            if (pid, date) not in sr_cache:
                sr_cache[pid, date] = ps.bat_sr(pid, date)
            return sr_cache[pid, date]

        def econ(pid, date):
            if (pid, date) not in econ_cache:
                econ_cache[pid, date] = ps.bowl_econ(pid, date)
            return econ_cache[pid, date]

        for i in range(n):
            date = dates[i]
            if striker is not None and striker[i]:
                out["striker_sr"][i] = sr(striker[i], date)
            if non_striker is not None and non_striker[i]:
                out["nonstriker_sr"][i] = sr(non_striker[i], date)
            if bowler is not None and bowler[i]:
                out["bowler_econ"][i] = econ(bowler[i], date)
            if yet_to_bat is not None:
                key = ("bat", yet_to_bat[i], date)
                if key not in group_cache:
                    ids = _ids(yet_to_bat[i])
                    group_cache[key] = (np.mean([sr(p, date) for p in ids]) if ids
                                        else ps.global_sr)
                out["yet_to_bat_sr"][i] = group_cache[key]
            if bowl_xi is not None and bowl_xi[i]:
                key = ("bowl", bowl_xi[i], date)
                if key not in group_cache:
                    ids = _ids(bowl_xi[i])
                    weights = np.array([ps.bowl_balls(p, date) for p in ids])
                    econs = np.array([econ(p, date) for p in ids])
                    group_cache[key] = (np.average(econs, weights=weights) if weights.sum() > 0
                                        else ps.global_econ)
                out["attack_econ"][i] = group_cache[key]
        return out

    def _numeric(self, df: pd.DataFrame, training: bool) -> np.ndarray:
        """Raw numeric matrix in `num_cols` order. Unavailable columns are NaN."""
        d = add_derived(df)
        computed: dict[str, np.ndarray] = {}
        if "venue_era" in self.groups:
            computed.update(self._venue_features(d, training))
        if "players" in self.groups:
            computed.update(self._player_features(d))
        columns = []
        for c in self.num_cols:
            if c in computed:
                columns.append(computed[c])
            elif c in d.columns:
                columns.append(d[c].to_numpy(float))
            else:
                columns.append(np.full(len(d), np.nan))
        return np.column_stack(columns)

    def transform(self, df: pd.DataFrame, training: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """Return (categorical int64 [n, 3], numeric float32 [n, len(num_cols)]).

        Pass training=True only for rows of the matches the encoder was fit on (it needs their
        match_id).
        """
        cats = np.stack(
            [df[c].map(self.vocabs[c]).fillna(0).astype(np.int64).to_numpy() for c in CAT_COLS],
            axis=1,
        )
        num = self._numeric(df, training)
        num = np.where(np.isnan(num), self.mean, num)  # unavailable inputs -> training mean
        return cats, ((num - self.mean) / self.std).astype(np.float32)

    # ---- persistence ----
    def state_dict(self) -> dict:
        return {"vocabs": self.vocabs, "mean": self.mean, "std": self.std,
                "groups": self.groups, "venue": self.venue, "player_stats": self.player_stats}

    @classmethod
    def from_state_dict(cls, state: dict) -> "FeatureEncoder":
        """Checkpoints from before feature groups existed have no `groups` key."""
        return cls(vocabs=state["vocabs"], mean=state["mean"], std=state["std"],
                   groups=state.get("groups", []), venue=state.get("venue"),
                   player_stats=state.get("player_stats"))
