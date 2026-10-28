"""Training loop, checkpointing and config/table helpers shared by the CLI."""
import copy
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn

from .data.dataset import make_loader
from .data.features import NUM_COLS, FeatureEncoder
from .data.parse import build_table
from .data.split import split_data
from .models.mlp import ScorePredictor


def load_config(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def table_path(cfg: dict) -> Path:
    return Path(cfg["paths"]["processed_dir"]) / "first_innings_states.csv"


def prepare(cfg: dict) -> pd.DataFrame:
    df = build_table(cfg["paths"]["raw_dir"])
    out = table_path(cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return df


def load_table(cfg: dict) -> pd.DataFrame:
    path = table_path(cfg)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `prepare` first")
    return pd.read_csv(path, dtype={"match_id": str})


def build_model(cfg: dict, encoder: FeatureEncoder) -> ScorePredictor:
    m = cfg["model"]
    return ScorePredictor(
        encoder.cardinalities, len(NUM_COLS), m["embed_dim"], tuple(m["hidden_dims"]), m["dropout"]
    )


def _run_epoch(model, loader, loss_fn, optimiser=None) -> float:
    training = optimiser is not None
    model.train(training)
    total, n = 0.0, 0
    with torch.set_grad_enabled(training):
        for cats, num, target in loader:
            loss = loss_fn(model(cats, num), target)
            if training:
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
            total += loss.item() * len(target)
            n += len(target)
    return total / n


def train(cfg: dict) -> dict:
    set_seed(cfg["seed"])
    t = cfg["train"]
    df = load_table(cfg)
    train_df, val_df, _ = split_data(df, cfg["split"])
    encoder = FeatureEncoder().fit(train_df)

    scale = t["target_scale"]
    train_loader = make_loader(train_df, encoder, t["batch_size"], scale, shuffle=True)
    val_loader = make_loader(val_df, encoder, t["batch_size"], scale, shuffle=False)

    model = build_model(cfg, encoder)
    optimiser = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, factor=0.5, patience=3)
    loss_fn = nn.HuberLoss(delta=t["huber_delta"])

    best_val, best_state, stale, history = float("inf"), None, 0, []
    for epoch in range(1, t["epochs"] + 1):
        tr = _run_epoch(model, train_loader, loss_fn, optimiser)
        va = _run_epoch(model, val_loader, loss_fn)
        scheduler.step(va)
        history.append((tr, va))
        print(f"epoch {epoch:3d}  train {tr:.5f}  val {va:.5f}")
        if va < best_val:
            best_val, best_state, stale = va, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= t["patience"]:
                print("early stopping")
                break

    model.load_state_dict(best_state)
    ckpt_path = Path(cfg["paths"]["checkpoint"])
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": best_state, "config": cfg, "encoder": encoder.state_dict(), "val_loss": best_val},
        ckpt_path,
    )
    print(f"saved {ckpt_path} (best val loss {best_val:.5f})")
    return {"best_val": best_val, "history": history}


def load_checkpoint(path: str | Path) -> tuple[ScorePredictor, FeatureEncoder, dict]:
    ckpt = torch.load(path, weights_only=False)
    encoder = FeatureEncoder.from_state_dict(ckpt["encoder"])
    model = build_model(ckpt["config"], encoder)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, encoder, ckpt["config"]
