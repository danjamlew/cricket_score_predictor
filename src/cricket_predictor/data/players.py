"""Career batting/bowling rates by player, available strictly before a given date."""
from bisect import bisect_left

import numpy as np
import pandas as pd

PRIOR_BALLS = 60.0  # pseudo-balls of league-average performance mixed into every player's rate


class PlayerStats:
    """Shrunk strike rate (runs per 100 balls) and economy (runs per over) per player.

    Built from the rows of a player-match table it is given (pass train matches only).
    A query for date D uses only matches dated strictly before D, so a match never sees
    itself, and unknown players or debutants fall back to the league-average prior.
    """

    def __init__(self, table: pd.DataFrame):
        self.global_sr = 100 * table["bat_runs"].sum() / max(table["bat_balls"].sum(), 1)
        self.global_econ = 6 * table["bowl_runs"].sum() / max(table["bowl_balls"].sum(), 1)
        self.name_to_id = dict(zip(table["name"], table["player"]))
        self._history: dict[str, tuple[list[str], np.ndarray]] = {}
        cols = ["bat_runs", "bat_balls", "bowl_runs", "bowl_balls"]
        for pid, g in table.sort_values("date").groupby("player"):
            self._history[pid] = (list(g["date"]), np.cumsum(g[cols].to_numpy(dtype=float), axis=0))

    def _before(self, pid: str, date: str) -> np.ndarray:
        """Cumulative [bat_runs, bat_balls, bowl_runs, bowl_balls] over matches before `date`."""
        dates, cum = self._history.get(pid, ([], None))
        n = bisect_left(dates, date)
        return np.zeros(4) if n == 0 else cum[n - 1]

    def bat_sr(self, pid: str, date: str) -> float:
        runs, balls = self._before(pid, date)[:2]
        return 100 * (runs + PRIOR_BALLS * self.global_sr / 100) / (balls + PRIOR_BALLS)

    def bowl_econ(self, pid: str, date: str) -> float:
        runs, balls = self._before(pid, date)[2:]
        return 6 * (runs + PRIOR_BALLS * self.global_econ / 6) / (balls + PRIOR_BALLS)

    def bowl_balls(self, pid: str, date: str) -> float:
        return self._before(pid, date)[3]
