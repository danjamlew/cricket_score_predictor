"""Experiment tracking: one folder per run under `experiments/`.

experiments/<name>/
    model.pt       checkpoint (weights, encoder, config)
    summary.json   name, timestamp, notes, config, split, training stats, test metrics
    by_over.csv    per-over MAE and bias for the network and the baselines
"""
import json
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch

from .evaluate import evaluate
from .plot_summary import error_by_over
from .train import load_checkpoint


def experiments_dir(cfg: dict) -> Path:
    return Path(cfg["paths"].get("experiments_dir", "experiments"))


def exp_path(cfg: dict, name: str) -> Path:
    return experiments_dir(cfg) / name


def list_experiments(cfg: dict) -> list[dict]:
    """Summaries of all recorded experiments, oldest first."""
    root = experiments_dir(cfg)
    summaries = [json.loads(p.read_text()) for p in root.glob("*/summary.json")]
    return sorted(summaries, key=lambda s: s["created"])


def resolve_checkpoint(cfg: dict, name: str | None) -> Path:
    """Checkpoint of the named experiment; default is the latest one, else paths.checkpoint."""
    if name:
        path = exp_path(cfg, name) / "model.pt"
        if not path.exists():
            known = ", ".join(s["name"] for s in list_experiments(cfg)) or "none"
            raise FileNotFoundError(f"No experiment '{name}' (recorded: {known})")
        return path
    recorded = list_experiments(cfg)
    if recorded:
        return exp_path(cfg, recorded[-1]["name"]) / "model.pt"
    return Path(cfg["paths"]["checkpoint"])


def record(
    cfg: dict, name: str, checkpoint: str | Path, notes: str = "",
    history: list | None = None, seconds: float | None = None,
) -> Path:
    """Evaluate a checkpoint on its own test split and store everything under experiments/<name>."""
    dest = exp_path(cfg, name)
    dest.mkdir(parents=True, exist_ok=True)
    model_path = dest / "model.pt"
    if Path(checkpoint).resolve() != model_path.resolve():
        shutil.copy2(checkpoint, model_path)

    ckpt = torch.load(model_path, weights_only=False)
    model, _, run_cfg = load_checkpoint(model_path)
    metrics = evaluate(run_cfg, model_path)
    by_over, n_test_matches = error_by_over(run_cfg, model_path)
    by_over.to_csv(dest / "by_over.csv", index=False)

    summary = {
        "name": name,
        "created": datetime.now().isoformat(timespec="seconds"),
        "notes": notes,
        "config": {k: run_cfg[k] for k in ("seed", "split", "features", "model", "train")
                   if k in run_cfg},
        "best_val_loss": ckpt.get("val_loss"),
        "epochs_run": len(history) if history else None,
        "train_seconds": round(seconds, 1) if seconds is not None else None,
        "n_parameters": sum(p.numel() for p in model.parameters()),
        "n_test_matches": n_test_matches,
        "metrics": metrics.to_dict("records"),
    }
    (dest / "summary.json").write_text(json.dumps(summary, indent=2))
    return dest


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        out.update(_flatten(v, key + ".") if isinstance(v, dict) else {key: v})
    return out


def comparison_table(summaries: list[dict]) -> pd.DataFrame:
    """One row per experiment (network) followed by the two baselines from the first run."""
    stages = ["ball 0", "overs 1-5", "overs 6-10", "overs 11-15", "overs 16-19"]

    def row(label: str, summary: dict, model: str) -> dict:
        m = pd.DataFrame(summary["metrics"])
        m = m[m["model"] == model].set_index("stage")
        out = {"experiment": label, "MAE": m.loc["all (balls 0-119)", "MAE"],
               "RMSE": m.loc["all (balls 0-119)", "RMSE"]}
        out.update({s: m.loc[s, "MAE"] for s in stages})
        return out

    rows = [{**row(s["name"], s, "network"), "val loss": s["best_val_loss"]} for s in summaries]
    for baseline in ("run-rate x 20", "venue mean"):
        rows.append(row(f"(baseline) {baseline}", summaries[0], baseline))
    table = pd.DataFrame(rows)
    table["val loss"] = table["val loss"].map(lambda v: "" if pd.isna(v) else f"{v:.5f}")
    first_mae = table.loc[0, "MAE"]
    table.insert(2, f"MAE vs {summaries[0]['name']}", table["MAE"] - first_mae)
    return table


def config_differences(summaries: list[dict]) -> dict[str, list]:
    """Config keys whose values are not identical across the experiments."""
    flat = [_flatten(s["config"]) for s in summaries]
    keys = sorted(set().union(*flat))
    return {k: [f.get(k) for f in flat] for k in keys if len({str(f.get(k)) for f in flat}) > 1}
