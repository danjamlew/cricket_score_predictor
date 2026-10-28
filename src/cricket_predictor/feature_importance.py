"""Permutation feature importance on the test split, and a ranking plot."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from .data.features import BASE_NUM_COLS, CAT_COLS, GROUP_NUM_COLS  # noqa: E402
from .data.split import split_data  # noqa: E402
from .plot import GRID, INK, INK_MUTED, SURFACE  # noqa: E402
from .train import load_checkpoint, load_table  # noqa: E402

# Categorical slots 1-5 of the reference palette, in fixed order, one per feature group.
GROUP_COLOURS = {
    "base": "#2a78d6", "base (venue/teams)": "#eb6834", "state_extras": "#1baf7a",
    "venue_era": "#eda100", "players": "#e87ba4",
}


def feature_group(name: str) -> str:
    if name in CAT_COLS:
        return "base (venue/teams)"
    if name in BASE_NUM_COLS:
        return "base"
    return next(g for g, cols in GROUP_NUM_COLS.items() if name in cols)


@torch.no_grad()
def permutation_importance(
    cfg: dict, checkpoint: str | Path, repeats: int = 5, seed: int = 0
) -> pd.DataFrame:
    """Increase in test MAE (runs) when one feature is shuffled across rows.

    Columns: feature, group, importance (mean over repeats), std. The three categorical
    inputs are each shuffled as one feature.
    """
    model, encoder, ckpt_cfg = load_checkpoint(checkpoint)
    df = load_table(cfg)
    _, _, test_df = split_data(df, ckpt_cfg["split"])
    test_df = test_df[test_df["balls"] < 120]  # same range as `evaluate`
    scale = ckpt_cfg["train"]["target_scale"]
    cats_np, num_np = encoder.transform(test_df)
    cats, num = torch.from_numpy(cats_np), torch.from_numpy(num_np)
    runs, final = test_df["runs"].to_numpy(float), test_df["final"].to_numpy(float)

    def mae(c: torch.Tensor, n: torch.Tensor) -> float:
        return float(np.abs(runs + model(c, n).numpy() * scale - final).mean())

    base, rng, rows = mae(cats, num), np.random.default_rng(seed), []
    features = [(c, "cat", i) for i, c in enumerate(CAT_COLS)] + [
        (c, "num", i) for i, c in enumerate(encoder.num_cols)]
    for name, kind, j in features:
        deltas = []
        for _ in range(repeats):
            perm = torch.from_numpy(rng.permutation(len(test_df)))
            c, n = cats.clone(), num.clone()
            if kind == "cat":
                c[:, j] = cats[perm, j]
            else:
                n[:, j] = num[perm, j]
            deltas.append(mae(c, n) - base)
        rows.append({"feature": name, "group": feature_group(name),
                     "importance": np.mean(deltas), "std": np.std(deltas)})
    return pd.DataFrame(rows).sort_values("importance", ascending=False, ignore_index=True)


def plot_importance(cfg: dict, checkpoint: str | Path, out: str, repeats: int = 5) -> Path:
    table = permutation_importance(cfg, checkpoint, repeats)
    table = table.iloc[::-1]  # barh draws bottom-up: most important at the top
    fig, ax = plt.subplots(figsize=(9, 0.32 * len(table) + 1.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.barh(table["feature"], table["importance"], xerr=table["std"], height=0.62,
            color=[GROUP_COLOURS[g] for g in table["group"]],
            error_kw={"ecolor": INK_MUTED, "elinewidth": 1, "capsize": 0})
    for y, (v, s) in enumerate(zip(table["importance"], table["std"])):
        ax.text(max(v, 0) + s + 0.05, y, f"{v:.2f}", va="center", fontsize=8, color=INK_MUTED)
    ax.axvline(0, color=INK, linewidth=1)
    ax.set_xlabel("Increase in test MAE when the feature is shuffled (runs)", color=INK_MUTED)
    ax.set_title("Which inputs does the model rely on? (permutation importance)",
                 loc="left", color=INK, fontsize=12)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    present = [g for g in GROUP_COLOURS if g in set(table["group"])]
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=GROUP_COLOURS[g]) for g in present], present,
              frameon=False, labelcolor=INK, fontsize=9, loc="lower right", title="feature group",
              title_fontsize=9)
    fig.text(0.01, 0.01, "Bars are means over repeated shuffles (whiskers: one std). Correlated "
             "features share credit, so each one understates its group.", fontsize=7.5,
             color=INK_MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out_path
