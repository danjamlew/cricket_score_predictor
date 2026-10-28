"""Plot the per-over error of several recorded experiments on one chart."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .experiments import exp_path  # noqa: E402
from .plot import INK, INK_MUTED, SURFACE  # noqa: E402
from .plot_summary import _style  # noqa: E402

# Categorical slots 1-8 in fixed order: an experiment keeps its colour (by creation order).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BASELINE_STYLES = {"run-rate x 20": (INK_MUTED, "--"), "venue mean": (INK_MUTED, ":")}


def plot_compare(cfg: dict, summaries: list[dict], out: str) -> Path:
    if len(summaries) > len(SERIES):
        raise ValueError(f"At most {len(SERIES)} experiments can be plotted; pick a subset")
    tables = {s["name"]: pd.read_csv(exp_path(cfg, s["name"]) / "by_over.csv") for s in summaries}

    fig, (ax_mae, ax_bias) = plt.subplots(1, 2, figsize=(12, 5), facecolor=SURFACE)
    first = tables[summaries[0]["name"]]
    for name, (colour, style) in BASELINE_STYLES.items():
        sub = first[first["model"] == name]
        ax_mae.plot(sub["over"], sub["mae"], color=colour, linestyle=style, linewidth=1.2,
                    label=f"{name} (baseline)")
        ax_bias.plot(sub["over"], sub["bias"], color=colour, linestyle=style, linewidth=1.2)

    for colour, s in zip(SERIES, summaries):
        sub = tables[s["name"]]
        sub = sub[sub["model"] == "network"]
        ax_mae.plot(sub["over"], sub["mae"], color=colour, linewidth=2, label=s["name"])
        ax_bias.plot(sub["over"], sub["bias"], color=colour, linewidth=2)
    ax_bias.axhline(0, color=INK, linewidth=1)

    _style(ax_mae, "Mean absolute error (runs)", "Average miss, lower is better")
    _style(ax_bias, "Mean error: prediction minus actual (runs)",
           "Bias: above 0 guesses too high, below 0 too low")
    ax_mae.legend(frameon=False, labelcolor=INK, fontsize=9, loc="upper right")
    fig.suptitle("Experiment comparison on the test split", x=0.01, ha="left", color=INK,
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out_path
