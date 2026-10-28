import json

import numpy as np
import pandas as pd
import pytest

from cricket_predictor.data.features import (
    BASE_NUM_COLS, CAT_COLS, GROUP_NUM_COLS, FeatureEncoder, add_derived,
)
from cricket_predictor.data.normalise import canonical_team, canonical_venue
from cricket_predictor.data.parse import parse_match, parse_player_stats
from cricket_predictor.data.players import PlayerStats
from cricket_predictor.data.split import split_by_year, split_data

BAT_XI = list("ABCDEFGHIJK")
BOWL_XI = [f"X{i}" for i in range(1, 12)]


def _ball(runs=0, extras=None, wicket=None, batter="A", non_striker="C", bowler="X1"):
    d = {"batter": batter, "bowler": bowler, "non_striker": non_striker,
         "runs": {"batter": runs, "extras": sum((extras or {}).values()), "total": runs + sum((extras or {}).values())}}
    if extras:
        d["extras"] = extras
    if wicket:
        d["wickets"] = [{"kind": wicket, "player_out": batter}]
    return d


def _match(first, target_overs=20, super_over=False, n_innings=2, year="2019-04-01",
           toss_winner="Delhi Daredevils"):
    inn = [{"team": "Delhi Daredevils", "overs": [{"over": 0, "deliveries": first}]}]
    if super_over:
        inn[0]["super_over"] = True
    if n_innings > 1:
        inn.append({"team": "Mumbai Indians", "target": {"overs": target_overs, "runs": 10},
                    "overs": []})
    people = {n: f"id_{n}" for n in BAT_XI + BOWL_XI}
    return {"info": {"dates": [year], "venue": "Wankhede Stadium, Mumbai", "overs": 20,
                     "teams": ["Delhi Daredevils", "Mumbai Indians"],
                     "toss": {"winner": toss_winner, "decision": "bat"},
                     "players": {"Delhi Daredevils": BAT_XI, "Mumbai Indians": BOWL_XI},
                     "registry": {"people": people}}, "innings": inn}


def _write(tmp_path, data, name="1.json"):
    p = tmp_path / name
    p.write_text(json.dumps(data))
    return p


def test_legal_ball_counting_and_state(tmp_path):
    balls = [_ball(4), _ball(0, {"wides": 1}), _ball(1), _ball(0, wicket="bowled"),
             _ball(0, {"noballs": 1}, batter="B"), _ball(6, batter="B"),
             _ball(0, wicket="retired hurt", batter="B")]
    rows = parse_match(_write(tmp_path, _match(balls)))
    df = pd.DataFrame(rows)
    # 0 plus 5 legal balls (wide and no-ball excluded)
    assert list(df["balls"]) == [0, 1, 2, 3, 4, 5]
    assert list(df["runs"]) == [0, 4, 6, 6, 13, 13]  # wide credited to ball 2, no-ball to ball 4
    assert list(df["wickets"]) == [0, 0, 0, 1, 1, 1]   # retired hurt is not counted
    assert (df["final"] == 13).all()
    assert df["bat_team"].iloc[0] == "Delhi Capitals"
    assert df["venue"].iloc[0] == "Wankhede Stadium"


def test_new_state_columns(tmp_path):
    balls = [_ball(4), _ball(0, {"wides": 1}), _ball(1), _ball(0, wicket="bowled"),
             _ball(0, {"noballs": 1}, batter="B"), _ball(6, batter="B"),
             _ball(0, wicket="retired hurt", batter="B")]
    df = pd.DataFrame(parse_match(_write(tmp_path, _match(balls))))
    assert list(df["boundaries_last30"]) == [0, 1, 1, 1, 2, 2]
    assert list(df["dots_last30"]) == [0, 0, 0, 1, 1, 2]
    assert list(df["extras_last30"]) == [0, 0, 1, 1, 2, 2]  # the wide and the no-ball
    assert list(df["balls_since_wkt"]) == [0, 1, 2, 0, 1, 2]  # only the bowled wicket resets
    assert list(df["runs_since_wkt"]) == [0, 4, 6, 0, 7, 7]
    # striker/non-striker are the batters of the NEXT delivery; the bowler bowled the last ball
    assert list(df["striker_id"]) == ["id_A", "id_A", "id_A", "id_B", "id_B", "id_B"]
    assert df["nonstriker_id"].iloc[0] == "id_C"
    assert df["bowler_id"].iloc[0] == "" and df["bowler_id"].iloc[1] == "id_X1"
    # XI minus everyone who has been at the crease, including the next-ball batters
    assert len(df["yet_to_bat"].iloc[0].split("|")) == 9
    assert "id_B" in df["yet_to_bat"].iloc[2].split("|")
    assert "id_B" not in df["yet_to_bat"].iloc[3].split("|")
    assert df["bowl_xi"].iloc[0] == "|".join(sorted(f"id_{n}" for n in BOWL_XI))
    assert (df["date"] == "2019-04-01").all() and (df["toss_won_by_batting"] == 1).all()
    lost = pd.DataFrame(parse_match(_write(
        tmp_path, _match(balls, toss_winner="Mumbai Indians"), "2.json")))
    assert (lost["toss_won_by_batting"] == 0).all()


def test_player_match_stats(tmp_path):
    balls = [_ball(4), _ball(0, {"wides": 1}), _ball(1, bowler="X2"),
             _ball(0, {"byes": 1}), _ball(2, {"noballs": 1})]
    rows = {r["player"]: r for r in parse_player_stats(_write(tmp_path, _match(balls)))}
    a, x1, x2 = rows["id_A"], rows["id_X1"], rows["id_X2"]
    assert (a["bat_runs"], a["bat_balls"]) == (4 + 1 + 2, 4)       # the wide is not a ball faced
    assert (x1["bowl_runs"], x1["bowl_balls"]) == (4 + 1 + 0 + 2 + 1, 2)   # byes are not conceded
    assert (x2["bowl_runs"], x2["bowl_balls"]) == (1, 1)
    assert a["name"] == "A" and a["date"] == "2019-04-01"


def test_unusable_matches_are_dropped(tmp_path):
    assert parse_match(_write(tmp_path, _match([_ball(1)], super_over=True))) == []
    assert parse_match(_write(tmp_path, _match([_ball(1)], n_innings=1))) == []
    assert parse_match(_write(tmp_path, _match([_ball(1)], target_overs=6))) == []


def test_normalisation():
    assert canonical_team("Royal Challengers Bengaluru") == "Royal Challengers Bangalore"
    assert canonical_venue("MA Chidambaram Stadium, Chepauk, Chennai") == "MA Chidambaram Stadium"
    assert canonical_venue("Feroz Shah Kotla") == canonical_venue("Arun Jaitley Stadium, Delhi")


def _table():
    rows = []
    for mid, year in [("a", 2020), ("b", 2023), ("c", 2024), ("d", 2025)]:
        for b in range(0, 121, 20):
            rows.append({"match_id": mid, "year": year, "venue": "V" + mid, "bat_team": "X",
                         "bowl_team": "Y", "balls": b, "runs": b, "wickets": b // 40,
                         "runs_last12": 5, "runs_last30": 10, "wkts_last30": 0, "final": 150})
    return pd.DataFrame(rows)


def test_split_is_chronological_with_no_match_overlap():
    train, val, test = split_by_year(_table(), 2023, 2024)
    assert set(train["match_id"]) == {"a", "b"}
    assert set(val["match_id"]) == {"c"}
    assert set(test["match_id"]) == {"d"}


def test_stratified_split_covers_every_year_without_match_overlap():
    rows = []
    for year in (2020, 2021, 2022):
        for m in range(20):
            rows.append({**_table().iloc[0].to_dict(), "match_id": f"{year}-{m}", "year": year})
    df = pd.DataFrame(rows)
    cfg = {"mode": "stratified_year", "val_frac": 0.1, "test_frac": 0.1, "seed": 1}
    train, val, test = split_data(df, cfg)
    ids = [set(x["match_id"]) for x in (train, val, test)]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    assert sum(map(len, ids)) == 60
    for year in (2020, 2021, 2022):  # 20 matches per year -> 16 / 2 / 2
        assert [int((x["year"] == year).sum()) for x in (train, val, test)] == [16, 2, 2]
    again = split_data(df.sample(frac=1, random_state=3), cfg)  # row order must not matter
    assert [set(x["match_id"]) for x in again] == ids


def test_split_mode_defaults_to_chronological():
    train, val, test = split_data(_table(), {"train_until": 2023, "val_year": 2024})
    assert set(test["match_id"]) == {"d"}


def test_encoder_shapes_and_unseen_values():
    df = _table()
    enc = FeatureEncoder().fit(df[df["year"] <= 2023])
    cats, num = enc.transform(df)
    assert cats.shape == (len(df), len(CAT_COLS)) and num.shape == (len(df), len(BASE_NUM_COLS))
    assert (cats[df["venue"] == "Vd", 0] == 0).all()  # unseen venue -> index 0


def test_train_smoke(tmp_path):
    from cricket_predictor.evaluate import evaluate
    from cricket_predictor.train import load_config, train

    cfg = load_config("configs/default.yaml")
    cfg["paths"] = {"raw_dir": "", "processed_dir": str(tmp_path),
                    "checkpoint": str(tmp_path / "m.pt")}
    cfg["train"].update(epochs=2, batch_size=8)
    cfg["split"] = {"mode": "chronological", "train_until": 2023, "val_year": 2024}
    _table().to_csv(tmp_path / "first_innings_states.csv", index=False)
    train(cfg)
    result = evaluate(cfg, cfg["paths"]["checkpoint"])
    assert {"network", "run-rate x 20", "venue mean"} <= set(result["model"])
    assert result["MAE"].notna().all()


def test_experiment_tracking(tmp_path):
    from cricket_predictor.experiments import (
        comparison_table, config_differences, list_experiments, record, resolve_checkpoint,
    )
    from cricket_predictor.plot_compare import plot_compare
    from cricket_predictor.train import load_config, train

    cfg = load_config("configs/default.yaml")
    cfg["paths"] = {"raw_dir": "", "processed_dir": str(tmp_path),
                    "checkpoint": str(tmp_path / "m.pt"), "experiments_dir": str(tmp_path / "exp")}
    cfg["train"].update(epochs=1, batch_size=8)
    cfg["split"] = {"mode": "chronological", "train_until": 2023, "val_year": 2024}
    _table().to_csv(tmp_path / "first_innings_states.csv", index=False)

    train(cfg)
    record(cfg, "a", cfg["paths"]["checkpoint"], notes="first")
    cfg["model"]["dropout"] = 0.3
    train(cfg)
    record(cfg, "b", cfg["paths"]["checkpoint"], history=[(1.0, 0.5)], seconds=2.0)

    summaries = list_experiments(cfg)
    assert [s["name"] for s in summaries] == ["a", "b"]
    assert resolve_checkpoint(cfg, None).parent.name == "b"  # latest by default
    assert resolve_checkpoint(cfg, "a").parent.name == "a"
    with pytest.raises(FileNotFoundError):
        resolve_checkpoint(cfg, "missing")

    table = comparison_table(summaries)
    assert list(table["experiment"]) == ["a", "b", "(baseline) run-rate x 20",
                                         "(baseline) venue mean"]
    assert set(config_differences(summaries)) == {"model.dropout"}
    assert plot_compare(cfg, summaries, str(tmp_path / "c.png")).exists()


def _venue_frame():
    """Two training matches at one venue: finals 100 and 200, runs 40 / 100 after 60 balls.
    m1 is played before m2."""
    rows = []
    for mid, date, at60, final in [("m1", "2020-01-01", 40, 100), ("m2", "2020-02-01", 100, 200)]:
        for balls, runs in [(0, 0), (60, at60)]:
            rows.append({"match_id": mid, "date": date, "year": 2020, "venue": "V",
                         "bat_team": "X", "bowl_team": "Y", "balls": balls, "runs": runs,
                         "wickets": 0, "runs_last12": 0, "runs_last30": 0, "wkts_last30": 0,
                         "toss_won_by_batting": 1, "final": final})
    return pd.DataFrame(rows)


def test_venue_stats_for_training_rows_only_see_earlier_matches():
    df = _venue_frame()
    enc = FeatureEncoder().fit(df, ["venue_era"])
    d = add_derived(df)
    first = d[(d["match_id"] == "m1") & (d["balls"] == 60)].index[0]
    second = d[(d["match_id"] == "m2") & (d["balls"] == 60)].index[0]
    train = enc._venue_features(d, training=True)
    full = enc._venue_features(d, training=False)
    k = 10  # shrinkage; global mean final 150, global mean runs after 60 balls 70
    # the first match has no history: league average. Its own result never appears.
    assert train["venue_mean_final"][first] == pytest.approx(150)
    assert train["venue_par_runs"][first] == pytest.approx(70)
    # the second sees only the first (final 100, 40 runs after 60 balls)
    assert train["venue_mean_final"][second] == pytest.approx((100 + k * 150) / (1 + k))
    assert train["venue_par_runs"][second] == pytest.approx((40 + k * 70) / (1 + k))
    # val/test/inference use the full training tables
    assert full["venue_mean_final"][first] == pytest.approx((300 + k * 150) / (2 + k))
    assert full["venue_par_runs"][first] == pytest.approx((140 + k * 70) / (2 + k))
    assert full["runs_vs_par"][first] == pytest.approx(40 - full["venue_par_runs"][first])
    # an unseen venue falls back to the league averages
    unseen = d.assign(venue="Elsewhere")
    assert enc._venue_features(unseen, training=False)["venue_mean_final"] == pytest.approx(150)
    # a (source-data) innings with 121 legal balls must not index past the tables
    assert len(enc._venue_features(d.assign(balls=121), training=False)["venue_par_runs"]) == len(d)
    # runs after 0 balls are 0 by definition (an exact-zero mean is not "no data")
    ball0 = d[d["balls"] == 0].index[0]
    assert full["venue_par_runs"][ball0] == pytest.approx(0)
    # the previous season mean never includes the current season
    assert (full["prev_season_mean"] == 150).all()
    assert enc._venue_features(d.assign(year=2021), False)["prev_season_mean"][0] == 150


def test_ordered_stats_use_only_earlier_matches_by_date():
    """Regression: leave-one-out made the feature a decreasing function of the row's own final,
    so a network that knows the venue could read the target off it. Each training match must
    see exactly the matches played before it, whatever the order of the ids."""
    finals = [120, 140, 160, 180, 200, 220]
    rows = []
    for i, final in enumerate(finals):  # m0 is played last, m5 first
        rows.append({**_venue_frame().iloc[0].to_dict(), "match_id": f"m{i}",
                     "date": f"2020-01-{6 - i:02d}", "final": final, "runs": 0})
    enc = FeatureEncoder().fit(pd.DataFrame(rows), ["venue_era"])
    o = enc._ordered
    for i, final in enumerate(finals):
        k = o["index"][f"m{i}"]
        earlier = finals[i + 1:]
        assert o["final_n"][k] == len(earlier) and o["final_sum"][k] == sum(earlier)


def _player_table():
    return pd.DataFrame([
        {"match_id": "m1", "date": "2020-01-01", "player": "p", "name": "P", "bat_runs": 50,
         "bat_balls": 25, "bowl_runs": 0, "bowl_balls": 0},
        {"match_id": "m2", "date": "2021-01-01", "player": "p", "name": "P", "bat_runs": 0,
         "bat_balls": 100, "bowl_runs": 12, "bowl_balls": 12},
        {"match_id": "zz", "date": "2021-06-01", "player": "q", "name": "Q", "bat_runs": 99,
         "bat_balls": 9, "bowl_runs": 0, "bowl_balls": 0},
    ])


def test_player_stats_use_only_strictly_earlier_matches():
    ps = PlayerStats(_player_table())
    prior = ps.global_sr
    assert ps.bat_sr("p", "2020-01-01") == pytest.approx(prior)       # same date: not included
    assert ps.bat_sr("nobody", "2030-01-01") == pytest.approx(prior)  # unknown player -> prior
    after_first = 100 * (50 + 60 * prior / 100) / (25 + 60)
    assert ps.bat_sr("p", "2020-06-01") == pytest.approx(after_first)
    after_both = 100 * (50 + 60 * prior / 100) / (125 + 60)
    assert ps.bat_sr("p", "2021-06-01") == pytest.approx(after_both)
    assert ps.bowl_balls("p", "2020-06-01") == 0 and ps.bowl_balls("p", "2021-06-01") == 12


def test_player_features_fit_on_training_matches_only():
    df = _venue_frame().assign(date="2021-03-01", striker_id="p", nonstriker_id="q",
                               bowler_id="p", yet_to_bat="", bowl_xi="p")
    enc = FeatureEncoder().fit(df, ["players"], _player_table())
    assert "q" not in enc.player_stats._history  # match zz is not a training match
    assert "p" in enc.player_stats._history
    cats, num = enc.transform(df)
    assert num.shape == (len(df), len(BASE_NUM_COLS) + len(GROUP_NUM_COLS["players"]))
    # the id columns are absent at inference: those inputs become neutral (standardised 0)
    _, neutral = enc.transform(df.drop(columns=["striker_id", "bowler_id"]))
    cols = enc.num_cols
    assert (neutral[:, cols.index("striker_sr")] == 0).all()
    assert (neutral[:, cols.index("bowler_econ")] == 0).all()


def test_legacy_encoder_state_still_loads():
    df = _table()
    enc = FeatureEncoder().fit(df)
    legacy = {k: v for k, v in enc.state_dict().items() if k in ("vocabs", "mean", "std")}
    loaded = FeatureEncoder.from_state_dict(legacy)
    assert loaded.groups == [] and loaded.num_cols == BASE_NUM_COLS
    assert (loaded.transform(df)[1] == enc.transform(df)[1]).all()


def test_unknown_group_is_rejected():
    with pytest.raises(ValueError, match="Unknown feature group"):
        FeatureEncoder().fit(_table(), ["nope"])


def test_all_groups_train_evaluate_predict(tmp_path):
    """End to end on synthetic JSON: parse, fit every group, train, evaluate, predict."""
    from cricket_predictor.data.parse import build_player_stats, build_table
    from cricket_predictor.evaluate import evaluate
    from cricket_predictor.predict import predict_state
    from cricket_predictor.train import load_checkpoint, load_config, train

    raw = tmp_path / "raw"
    raw.mkdir()
    for i, year in enumerate(range(2019, 2027)):
        balls = [_ball(i % 5, batter=BAT_XI[(i + j) % 6], bowler=BOWL_XI[j % 4]) for j in range(30)]
        balls[7] = _ball(0, wicket="caught", batter=balls[7]["batter"], bowler="X2")
        _write(raw, _match(balls, year=f"{year}-04-01"), f"{i}.json")
    cfg = load_config("configs/default.yaml")
    cfg["paths"] = {"raw_dir": str(raw), "processed_dir": str(tmp_path),
                    "checkpoint": str(tmp_path / "m.pt")}
    cfg["split"] = {"mode": "chronological", "train_until": 2023, "val_year": 2024}
    cfg["features"] = {"groups": ["state_extras", "venue_era", "players"]}
    cfg["train"].update(epochs=2, batch_size=16)
    build_table(raw).to_csv(tmp_path / "first_innings_states.csv", index=False)
    build_player_stats(raw).to_csv(tmp_path / "player_match_stats.csv", index=False)

    train(cfg)
    result = evaluate(cfg, cfg["paths"]["checkpoint"])  # synthetic innings are short: later stages are empty
    assert result[result["stage"] == "all (balls 0-119)"]["MAE"].notna().all()
    model, encoder, ckpt_cfg = load_checkpoint(cfg["paths"]["checkpoint"])
    assert encoder.groups == cfg["features"]["groups"]
    args = dict(runs=50, wickets=1, balls=30, venue="Wankhede Stadium", bat_team="Delhi Capitals",
                bowl_team="Mumbai Indians")
    assert predict_state(model, encoder, ckpt_cfg, **args) > 0
    assert predict_state(model, encoder, ckpt_cfg, **args, striker="A", non_striker="B",
                         bowler="X1") > 0
    from cricket_predictor.feature_importance import permutation_importance
    imp = permutation_importance(cfg, cfg["paths"]["checkpoint"], repeats=2)
    assert set(imp["feature"]) == set(CAT_COLS) | set(encoder.num_cols)
    assert imp["importance"].is_monotonic_decreasing and imp["importance"].notna().all()
    assert set(imp["group"]) == {"base", "base (venue/teams)", "state_extras", "venue_era",
                                 "players"}
    with pytest.raises(ValueError, match="Unknown player"):
        predict_state(model, encoder, ckpt_cfg, **args, striker="Nobody")
