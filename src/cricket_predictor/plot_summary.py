"""Summary plot of every method over the whole test split."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .data.split import split_data  # noqa: E402
from .evaluate import method_predictions  # noqa: E402
from .plot import COLOURS, GRID, INK, INK_MUTED, SURFACE  # noqa: E402
from .train import load_checkpoint, load_table  # noqa: E402


def error_by_over(cfg: dict, checkpoint: str) -> tuple[pd.DataFrame, int]:
    """Per-over MAE and mean signed error (prediction - actual) for each method.

    Returns (frame with columns over, model, mae, bias; number of test matches).
    """
    model, encoder, ckpt_cfg = load_checkpoint(checkpoint)
    df = load_table(cfg)
    train_df, _, test_df = split_data(df, ckpt_cfg["split"])  # the split the model was trained with
    if test_df.empty:
        raise ValueError("Test split is empty; adjust the split settings")
    test_df = test_df[test_df["balls"] < 120]  # same range as `evaluate`
    over = test_df["balls"] // 6  # overs completed; ball 0 falls in over 0

    rows = []
    for name, pred in method_predictions(model, encoder, ckpt_cfg, train_df, test_df).items():
        err = pred - test_df["final"]
        grouped = pd.DataFrame({"over": over, "err": err}).groupby("over")["err"]
        rows.append(pd.DataFrame({
            "over": grouped.size().index, "model": name,
            "mae": grouped.apply(lambda e: e.abs().mean()).to_numpy(),
            "bias": grouped.mean().to_numpy(),
        }))
    return pd.concat(rows, ignore_index=True), test_df["match_id"].nunique()


def _style(ax, ylabel: str, title: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_xlim(0, 19)
    ax.set_xticks(range(0, 20, 2))
    ax.set_xlabel("Overs completed", color=INK_MUTED)
    ax.set_ylabel(ylabel, color=INK_MUTED)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.tick_params(colors=INK_MUTED, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def plot_summary(cfg: dict, checkpoint: str, out: str) -> Path:
    table, n_matches = error_by_over(cfg, checkpoint)
    fig, (ax_mae, ax_bias) = plt.subplots(1, 2, figsize=(12, 5), facecolor=SURFACE)

    for name, colour in COLOURS.items():
        sub = table[table["model"] == name]
        ax_mae.plot(sub["over"], sub["mae"], color=colour, linewidth=2, label=name)
        ax_bias.plot(sub["over"], sub["bias"], color=colour, linewidth=2, label=name)
    ax_bias.axhline(0, color=INK, linewidth=1)

    _style(ax_mae, "Mean absolute error (runs)", "Average miss, lower is better")
    _style(ax_bias, "Mean error: prediction minus actual (runs)",
           "Bias: above 0 guesses too high, below 0 too low")
    ax_mae.legend(frameon=False, labelcolor=INK, fontsize=9, loc="upper right")
    fig.suptitle(f"Predicting the final total across {n_matches} test-split innings",
                 x=0.01, ha="left", color=INK, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out_path
