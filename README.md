# Cricket score predictor

PyTorch framework that predicts the final total of an IPL first innings from the match state at any ball, using the Cricsheet JSON files in `ipl_json/`.

Raw data is not in git. Download `ipl_json.zip` from [Cricsheet](https://cricsheet.org/downloads/) and unzip it to `ipl_json/` in the project root.

```bash
pip install -e ".[dev]"
python -m cricket_predictor prepare     # ipl_json -> data/processed/first_innings_states.csv
python -m cricket_predictor train       # records a new experiment (see below)
python -m cricket_predictor evaluate    # test split vs. baselines, by stage of the innings
python -m cricket_predictor predict --runs 85 --wickets 2 --balls 60 \
    --venue "Wankhede Stadium" --bat-team "Mumbai Indians" --bowl-team "Chennai Super Kings"
python -m cricket_predictor --experiment v2-all plot-importance   # rank input features
pytest
```

## Tracking experiments
Every `train` run is recorded under `experiments/<name>/` (checkpoint, config, test metrics, per-over error). The existing model is recorded as `v1`.

```bash
python3 -m cricket_predictor train --name v2 --notes "what changed"   # edit configs/default.yaml first
python3 -m cricket_predictor compare                                  # all experiments vs. v1 and baselines
python3 -m cricket_predictor compare v1 v2 --plot plots/compare.png
python3 -m cricket_predictor --experiment v1 plot-summary             # evaluate/plot/predict use the latest unless told
```
`compare` also lists which config values differ between runs and warns if the train/test splits differ (then the numbers are not comparable).

All settings live in `configs/default.yaml`.

## How it works
- **Data:** one row per legal ball (plus ball 0) of each first innings. Super overs, no-result and D/L-shortened matches are dropped. Team, venue and season names are normalised (`data/normalise.py`).
- **Base features (v1):** runs, wickets, balls bowled/left, run rate, runs in the last 12 and 30 balls, wickets in the last 30 balls, year, and embedded venue, batting team and bowling team.
- **Optional feature groups** (`features.groups` in the config, or `train --groups a,b`; `none` = base only):
  - `state_extras`: powerplay/death flags, wickets-in-hand x balls-left resources, balls and runs since the last wicket, boundaries/dots in the last 12 and 30 balls, extras in the last 30.
  - `venue_era`: Impact Player era flag, toss, venue average total and average score at this ball, runs vs that par, previous season's average total. Venue statistics come from training matches only, and a training match only sees matches at its venue played before it (leave-one-out would let the network read each match's own total off the feature).
  - `players`: career strike rate of the batters at the crease and of those yet to bat, economy of the last bowler and of the bowling XI. Rates use only training matches played strictly before the match date, shrunk toward the league average. `predict --striker "V Kohli" --non-striker ... --bowler ...` supplies players; anything not supplied is neutral.
  Re-run `prepare` after pulling changes that add columns. `train --seed N` repeats a run with another seed (differences of about 0.5 MAE between runs are within seed noise).
- **Model:** MLP with embeddings predicting remaining runs (softplus, so the total is never below the current score).
- **Split:** 80/10/10 by match, sampled randomly within each year (`split.mode: stratified_year`), so every season is represented in train, validation and test. Matches are never split. A chronological mode (`split.mode: chronological`) is also available. Each model is always evaluated on the split it was trained with, and `compare` only puts experiments with the same split in one table.
- **Baselines:** current run rate × 20 overs, and the venue's mean total.
