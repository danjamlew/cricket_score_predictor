import json

import pandas as pd
import pytest

from cricket_predictor.data.features import CAT_COLS, NUM_COLS, FeatureEncoder
from cricket_predictor.data.normalise import canonical_team, canonical_venue
from cricket_predictor.data.parse import parse_match
from cricket_predictor.data.split import split_by_year, split_data


def _ball(runs=0, extras=None, wicket=None):
    d = {"batter": "A", "bowler": "B", "non_striker": "C",
         "runs": {"batter": runs, "extras": sum((extras or {}).values()), "total": runs + sum((extras or {}).values())}}
    if extras:
        d["extras"] = extras
    if wicket:
        d["wickets"] = [{"kind": wicket, "player_out": "A"}]
    return d


def _match(first, target_overs=20, super_over=False, n_innings=2, year="2019-04-01"):
    inn = [{"team": "Delhi Daredevils", "overs": [{"over": 0, "deliveries": first}]}]
    if super_over:
        inn[0]["super_over"] = True
    if n_innings > 1:
        inn.append({"team": "Mumbai Indians", "target": {"overs": target_overs, "runs": 10},
                    "overs": []})
    return {"info": {"dates": [year], "venue": "Wankhede Stadium, Mumbai", "overs": 20,
                     "teams": ["Delhi Daredevils", "Mumbai Indians"]}, "innings": inn}


def _write(tmp_path, data, name="1.json"):
    p = tmp_path / name
    p.write_text(json.dumps(data))
    return p


def test_legal_ball_counting_and_state(tmp_path):
    balls = [_ball(4), _ball(0, {"wides": 1}), _ball(1), _ball(0, wicket="bowled"),
             _ball(0, {"noballs": 1}), _ball(6), _ball(0, wicket="retired hurt")]
    rows = parse_match(_write(tmp_path, _match(balls)))
    df = pd.DataFrame(rows)
    # 0 plus 5 legal balls (wide and no-ball excluded)
    assert list(df["balls"]) == [0, 1, 2, 3, 4, 5]
    assert list(df["runs"]) == [0, 4, 6, 6, 13, 13]  # wide credited to ball 2, no-ball to ball 4
    assert list(df["wickets"]) == [0, 0, 0, 1, 1, 1]   # retired hurt is not counted
    assert (df["final"] == 13).all()
    assert df["bat_team"].iloc[0] == "Delhi Capitals"
    assert df["venue"].iloc[0] == "Wankhede Stadium"


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
    assert cats.shape == (len(df), len(CAT_COLS)) and num.shape == (len(df), len(NUM_COLS))
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
