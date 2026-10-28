"""Command line: python -m cricket_predictor <command> (see --help)."""
import argparse
import copy
import time
from datetime import datetime

import pandas as pd

from .evaluate import evaluate
from .experiments import (
    comparison_table, config_differences, exp_path, list_experiments, record, resolve_checkpoint,
)
from .plot import plot_match
from .plot_compare import plot_compare
from .plot_summary import plot_summary
from .predict import predict_state
from .train import load_checkpoint, load_config, prepare, train


def _compare(cfg: dict, names: list[str], plot_out: str | None) -> None:
    summaries = list_experiments(cfg)
    if names:
        by_name = {s["name"]: s for s in summaries}
        missing = [n for n in names if n not in by_name]
        if missing:
            raise SystemExit(f"Unknown experiment(s): {', '.join(missing)}")
        summaries = [by_name[n] for n in names]
    if not summaries:
        raise SystemExit("No experiments recorded yet; run `train` or `register` first")

    table = comparison_table(summaries)
    print("Test-split MAE in runs (lower is better); stage columns are MAE by stage\n")
    print(table.to_string(index=False, na_rep="", float_format=lambda x: f"{x:.2f}"))
    for s in summaries:
        if s["notes"]:
            print(f"\n{s['name']}: {s['notes']}")

    splits = {str(s["config"]["split"]) for s in summaries}
    if len(splits) > 1:
        print("\nWARNING: these experiments use different train/test splits, so their "
              "numbers are not directly comparable.")
    diffs = config_differences(summaries)
    if diffs:
        print("\nConfig differences:")
        print(pd.DataFrame(diffs, index=[s["name"] for s in summaries]).T.to_string())
    if plot_out:
        print(f"\nsaved {plot_compare(cfg, summaries, plot_out)}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="cricket_predictor")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--experiment",
        help="experiment to use for evaluate/plot/plot-summary/predict (default: latest)",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare", help="parse ipl_json into a per-ball state table")

    t = sub.add_parser("train", help="train the network and record it as an experiment")
    t.add_argument("--name", help="experiment name (default: run-<timestamp>)")
    t.add_argument("--notes", default="", help="what is different about this run")

    r = sub.add_parser("register", help="record an existing checkpoint as an experiment")
    r.add_argument("name")
    r.add_argument("--checkpoint", help="default: paths.checkpoint from the config")
    r.add_argument("--notes", default="")

    c = sub.add_parser("compare", help="compare recorded experiments (all if none named)")
    c.add_argument("names", nargs="*")
    c.add_argument("--plot", metavar="PNG", help="also save a per-over comparison chart")

    sub.add_parser("evaluate", help="evaluate an experiment on the test split")
    pl = sub.add_parser("plot", help="plot the three methods' predictions through one match")
    pl.add_argument("--match-id", help="Cricsheet match id (default: first test-split match)")
    pl.add_argument("--out", default="plots/match.png")
    ps = sub.add_parser("plot-summary", help="plot every method's error over the whole test split")
    ps.add_argument("--out", default="plots/summary.png")
    p = sub.add_parser("predict", help="predict the final total from one match state")
    p.add_argument("--runs", type=int, required=True)
    p.add_argument("--wickets", type=int, required=True)
    p.add_argument("--balls", type=int, required=True, help="legal balls bowled so far")
    p.add_argument("--venue", required=True)
    p.add_argument("--bat-team", required=True)
    p.add_argument("--bowl-team", required=True)
    p.add_argument("--year", type=int, default=2026)
    p.add_argument("--runs-last12", type=int)
    p.add_argument("--runs-last30", type=int)
    p.add_argument("--wkts-last30", type=int)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.command == "prepare":
        df = prepare(cfg)
        print(f"{df['match_id'].nunique()} first innings, {len(df)} rows")
    elif args.command == "train":
        name = args.name or datetime.now().strftime("run-%Y%m%d-%H%M%S")
        dest = exp_path(cfg, name)
        if dest.exists():
            raise SystemExit(f"Experiment '{name}' already exists; choose another name")
        run_cfg = copy.deepcopy(cfg)
        run_cfg["paths"]["checkpoint"] = str(dest / "model.pt")
        start = time.time()
        result = train(run_cfg)
        record(cfg, name, run_cfg["paths"]["checkpoint"], args.notes, result["history"],
               time.time() - start)
        print(f"recorded experiment '{name}'; compare with: compare")
    elif args.command == "register":
        if exp_path(cfg, args.name).exists():
            raise SystemExit(f"Experiment '{args.name}' already exists; choose another name")
        record(cfg, args.name, args.checkpoint or cfg["paths"]["checkpoint"], args.notes)
        print(f"recorded experiment '{args.name}'")
    elif args.command == "compare":
        _compare(cfg, args.names, args.plot)
    else:
        checkpoint = resolve_checkpoint(cfg, args.experiment)
        print(f"using {checkpoint}")
        if args.command == "evaluate":
            table = evaluate(cfg, checkpoint)
            print(table.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
        elif args.command == "plot":
            print(f"saved {plot_match(cfg, checkpoint, args.match_id, args.out)}")
        elif args.command == "plot-summary":
            print(f"saved {plot_summary(cfg, checkpoint, args.out)}")
        else:
            model, encoder, ckpt_cfg = load_checkpoint(checkpoint)
            total = predict_state(
                model, encoder, ckpt_cfg, runs=args.runs, wickets=args.wickets,
                balls=args.balls, venue=args.venue, bat_team=args.bat_team,
                bowl_team=args.bowl_team, year=args.year, runs_last12=args.runs_last12,
                runs_last30=args.runs_last30, wkts_last30=args.wkts_last30,
            )
            print(f"Predicted final total: {total:.0f}")
