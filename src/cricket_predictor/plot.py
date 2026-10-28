"""Plot each method's running prediction of the final total through one innings."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .data.split import split_by_year  # noqa: E402
from .evaluate import method_predictions  # noqa: E402
from .train import load_checkpoint, load_table  # noqa: E402

# Categorical slots 1-3 of the reference palette, in fixed order, plus neutral ink for the truth.
SURFACE, INK, INK_MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2de"
COLOURS = {"network": "#2a78d6", "run-rate x 20": "#eb6834", "venue mean": "#1baf7a"}


def plot_match(cfg: dict, checkpoint: str, match_id: str | None, out: str) -> Path:
    model, encoder, ckpt_cfg = load_checkpoint(checkpoint)
    df = load_table(cfg)
    train_df, _, test_df = split_by_year(df, cfg["split"]["train_until"], cfg["split"]["val_year"])

    if match_id is None:
        match_id = str(test_df["match_id"].iloc[0])
    match = df[df["match_id"] == match_id]
    if match.empty:
        raise ValueError(f"Match {match_id} not found (only usable first innings are included)")
    in_train = match_id in set(train_df["match_id"])

    preds = method_predictions(model, encoder, ckpt_cfg, train_df, match)
    row = match.iloc[0]
    final = int(row["final"])
    x = match["balls"] / 6  # overs completed

    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.axhline(final, color=INK, linestyle="--", linewidth=1.5, label=f"Actual total ({final})")
    for name, pred in preds.items():
        ax.plot(x, pred.loc[match.index], color=COLOURS[name], linewidth=2, label=name)

    # Direct labels at the right edge, nudged apart so they never collide.
    ends = sorted(((preds[n].iloc[-1], n) for n in preds), key=lambda t: t[0])
    last_y = None
    for y, name in ends:
        ty = y if last_y is None or y - last_y >= 8 else last_y + 8
        ax.annotate(name, (x.iloc[-1], y), xytext=(x.iloc[-1] + 0.3, ty), color=INK,
                    fontsize=9, va="center", annotation_clip=False)
        last_y = ty

    ax.set_xlim(0, 20)
    ax.set_xticks(range(0, 21, 2))
    ax.set_xlabel("Overs completed", color=INK_MUTED)
    ax.set_ylabel("Predicted final total (runs)", color=INK_MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.tick_params(colors=INK_MUTED, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)

    note = "  (in the training data, so the network has seen this match)" if in_train else ""
    ax.set_title(
        f"{row['bat_team']} v {row['bowl_team']}, {row['venue']} {row['year']}{note}",
        loc="left", color=INK, fontsize=12,
    )
    ax.legend(frameon=False, labelcolor=INK, loc="lower right", fontsize=9)
    fig.tight_layout(rect=(0, 0, 0.88, 1))

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out_path
