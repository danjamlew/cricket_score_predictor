"""Parse Cricsheet JSON into per-ball first-innings states and per-player match stats."""
import json
from collections import defaultdict
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
    Player columns use the Cricsheet registry id: the striker/non-striker are the batters of the
    next delivery (what the scoreboard shows), the bowler is the one who bowled the last ball.
    """
    data = json.loads(Path(path).read_text())
    if not usable_first_innings(data):
        return []

    info = data["info"]
    people = info["registry"]["people"]
    innings = data["innings"][0]
    deliveries = [d for over in innings["overs"] for d in over["deliveries"]]
    if not deliveries:
        return []
    bat_raw = innings["team"]
    bowl_raw = next(t for t in info["teams"] if t != bat_raw)
    bat_team, bowl_team = canonical_team(bat_raw), canonical_team(bowl_raw)
    bat_xi = {people[n] for n in info["players"][bat_raw]}
    bowl_xi = sorted(people[n] for n in info["players"][bowl_raw])
    base = {
        "match_id": Path(path).stem,
        "year": match_year(info),
        "venue": canonical_venue(info["venue"]),
        "bat_team": bat_team,
        "bowl_team": bowl_team,
    }
    match_cols = {
        "date": info["dates"][0],
        # 1 if the batting side won the toss (and so chose to bat), 0 if it was put in
        "toss_won_by_batting": int(canonical_team(info.get("toss", {}).get("winner", "")) == bat_team),
        "bowl_xi": "|".join(bowl_xi),
    }

    appeared: set[str] = set()  # batters who have been at the crease so far

    def snapshot(next_d: dict, last_bowler: str, balls: int, runs: int, wickets: int,
                 ball_runs, ball_wkts, ball_bnd, ball_dot, ball_ext, since_b, since_r) -> dict:
        striker, non_striker = people[next_d["batter"]], people[next_d["non_striker"]]
        yet_to_bat = bat_xi - appeared - {striker, non_striker}
        return {
            **base, "balls": balls, "runs": runs, "wickets": wickets,
            "runs_last12": sum(ball_runs[-12:]), "runs_last30": sum(ball_runs[-30:]),
            "wkts_last30": sum(ball_wkts[-30:]), "final": None,
            "balls_since_wkt": since_b, "runs_since_wkt": since_r,
            "boundaries_last12": sum(ball_bnd[-12:]), "boundaries_last30": sum(ball_bnd[-30:]),
            "dots_last12": sum(ball_dot[-12:]), "dots_last30": sum(ball_dot[-30:]),
            "extras_last30": sum(ball_ext[-30:]),
            **match_cols, "striker_id": striker, "nonstriker_id": non_striker,
            "bowler_id": last_bowler, "yet_to_bat": "|".join(sorted(yet_to_bat)),
        }

    runs = wickets = balls = since_b = since_r = 0
    pending = pend_wkts = pend_bnd = pend_ext = 0
    ball_runs, ball_wkts, ball_bnd, ball_dot, ball_ext = [], [], [], [], []  # per legal ball
    rows = [snapshot(deliveries[0], "", 0, 0, 0, ball_runs, ball_wkts, ball_bnd, ball_dot,
                     ball_ext, 0, 0)]
    for i, d in enumerate(deliveries):
        appeared.update((people[d["batter"]], people[d["non_striker"]]))
        r, w = d["runs"]["total"], _wickets_in(d)
        runs += r
        wickets += w
        pending += r
        pend_wkts += w
        pend_bnd += d["runs"]["batter"] in (4, 6)
        pend_ext += d["runs"]["extras"]
        if not _is_legal(d):
            continue
        balls += 1
        ball_runs.append(pending)
        ball_wkts.append(pend_wkts)
        ball_bnd.append(pend_bnd)
        ball_dot.append(int(r == 0))
        ball_ext.append(pend_ext)
        if pend_wkts:
            since_b = since_r = 0
        else:
            since_b += 1
            since_r += pending
        pending = pend_wkts = pend_bnd = pend_ext = 0
        next_d = deliveries[i + 1] if i + 1 < len(deliveries) else d
        rows.append(snapshot(next_d, people[d["bowler"]], balls, runs, wickets, ball_runs,
                             ball_wkts, ball_bnd, ball_dot, ball_ext, since_b, since_r))
    for row in rows:
        row["final"] = runs
    return rows


def parse_player_stats(path: Path) -> list[dict]:
    """Per-player batting and bowling totals for one match (all innings except super overs)."""
    data = json.loads(Path(path).read_text())
    info = data["info"]
    people = info["registry"]["people"]
    names = {pid: name for name, pid in people.items()}
    stats: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0])  # bat runs/balls, bowl runs/balls
    for innings in data.get("innings", []):
        if innings.get("super_over"):
            continue
        for over in innings["overs"]:
            for d in over["deliveries"]:
                extras = d.get("extras", {})
                bat = stats[people[d["batter"]]]
                bat[0] += d["runs"]["batter"]
                bat[1] += "wides" not in extras  # a wide is not a ball faced
                bowl = stats[people[d["bowler"]]]
                bowl[2] += d["runs"]["batter"] + extras.get("wides", 0) + extras.get("noballs", 0)
                bowl[3] += _is_legal(d)
    return [
        {"match_id": Path(path).stem, "date": info["dates"][0], "player": pid,
         "name": names[pid], "bat_runs": s[0], "bat_balls": s[1], "bowl_runs": s[2],
         "bowl_balls": s[3]}
        for pid, s in stats.items()
    ]


def build_table(raw_dir: str | Path) -> pd.DataFrame:
    rows = []
    for path in sorted(Path(raw_dir).glob("*.json")):
        rows.extend(parse_match(path))
    if not rows:
        raise FileNotFoundError(f"No usable matches found in {raw_dir}")
    return pd.DataFrame(rows)


def build_player_stats(raw_dir: str | Path) -> pd.DataFrame:
    rows = []
    for path in sorted(Path(raw_dir).glob("*.json")):
        rows.extend(parse_player_stats(path))
    if not rows:
        raise FileNotFoundError(f"No matches found in {raw_dir}")
    return pd.DataFrame(rows)
