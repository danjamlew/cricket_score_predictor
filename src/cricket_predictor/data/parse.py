"""Parse Cricsheet JSON into one row per legal ball of each first innings."""
import json
from pathlib import Path

import pandas as pd

from .normalise import canonical_team, canonical_venue, match_year

NOT_BOWLER_DISMISSALS = {"retired hurt", "retired out"}
FULL_INNINGS_BALLS = 120


def _is_legal(delivery: dict) -> bool:
    extras = delivery.get("extras", {})
    return "wides" not in extras and "noballs" not in extras


def _wickets_in(delivery: dict) -> int:
    return sum(
        1 for w in delivery.get("wickets", []) if w["kind"] not in NOT_BOWLER_DISMISSALS
    )


def usable_first_innings(data: dict) -> bool:
    """Full-length, non-super-over first innings from a match with no reduced target."""
    innings = data.get("innings", [])
    if len(innings) < 2 or innings[0].get("super_over"):
        return False
    target = innings[1].get("target")
    if target is None or target.get("overs", 20) < data["info"].get("overs", 20):
        return False  # no-result, or D/L-shortened match
    return True


def parse_match(path: Path) -> list[dict]:
    """Return state rows for a match's first innings (empty if unusable).

    Row k describes the state after k legal balls (k = 0..N), with the final total as target.
    """
    data = json.loads(Path(path).read_text())
    if not usable_first_innings(data):
        return []

    info = data["info"]
    innings = data["innings"][0]
    bat_team = canonical_team(innings["team"])
    bowl_team = canonical_team(next(t for t in info["teams"] if t != innings["team"]))
    base = {
        "match_id": Path(path).stem,
        "year": match_year(info),
        "venue": canonical_venue(info["venue"]),
        "bat_team": bat_team,
        "bowl_team": bowl_team,
    }

    # (balls, runs, wickets, runs_last12, runs_last30, wkts_last30)
    states = [(0, 0, 0, 0, 0, 0)]
    runs = wickets = balls = pending = pend_wkts = 0
    ball_runs, ball_wkts = [], []  # per legal ball (includes extras since the previous one)
    for over in innings["overs"]:
        for d in over["deliveries"]:
            r = d["runs"]["total"]
            w = _wickets_in(d)
            runs += r
            wickets += w
            pending += r
            pend_wkts += w
            if _is_legal(d):
                balls += 1
                ball_runs.append(pending)
                ball_wkts.append(pend_wkts)
                pending = pend_wkts = 0
                states.append(
                    (
                        balls,
                        runs,
                        wickets,
                        sum(ball_runs[-12:]),
                        sum(ball_runs[-30:]),
                        sum(ball_wkts[-30:]),
                    )
                )
    final = runs
    rows = []
    for b, r, w, l12, l30, w30 in states:
        rows.append({**base, "balls": b, "runs": r, "wickets": w,
                     "runs_last12": l12, "runs_last30": l30, "wkts_last30": w30,
                     "final": final})
    return rows


def build_table(raw_dir: str | Path) -> pd.DataFrame:
    rows = []
    for path in sorted(Path(raw_dir).glob("*.json")):
        rows.extend(parse_match(path))
    if not rows:
        raise FileNotFoundError(f"No usable matches found in {raw_dir}")
    return pd.DataFrame(rows)
