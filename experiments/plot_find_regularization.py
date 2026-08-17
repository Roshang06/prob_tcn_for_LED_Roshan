import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

POWER_COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00"]
PANEL_LABELS = ["(a)", "(b)", "(c)", "(d)"]


def _load(stats_df):
    if isinstance(stats_df, (str, Path)):
        return pd.read_csv(stats_df)
    return stats_df


def plot_delta_vs_anneal(stats_df, out_path):
    stats_df = _load(stats_df)
    offsets = sorted(stats_df.dc_ma.unique())
    powers = sorted(stats_df.power.unique())
    color = {p: POWER_COLORS[i % len(POWER_COLORS)] for i, p in enumerate(powers)}

    n_rows = int(np.ceil(len(offsets) / 2))
    fig = Figure(figsize=(9, 4 * n_rows))
    axes = fig.subplots(n_rows, 2, squeeze=False).flatten()

    for panel_index, dc_ma in enumerate(offsets):
        ax = axes[panel_index]
        panel = stats_df[stats_df.dc_ma == dc_ma]
        for power in powers:
            trace = panel[panel.power == power].sort_values("anneal")
            if trace.empty:
                continue
            x = trace.anneal.to_numpy()
            ax.fill_between(x, trace.ci_lo.to_numpy(), trace.ci_hi.to_numpy(),
                            color=color[power], alpha=0.2, linewidth=0)
            ax.plot(x, trace.median_delta.to_numpy(), marker="o", color=color[power],
                    label=f"$\\lambda$={power:g}")
        ax.axhline(0, color="black", lw=0.8, linestyle="--")
        ax.text(0.05, 0.95, PANEL_LABELS[panel_index], transform=ax.transAxes, va="top", ha="left")
        ax.set_title(f"{dc_ma} mA")
        ax.set_xlabel("Noise Anneal Start")
        ax.set_ylabel("Median $\\Delta$ EVM% (ctrl $-$ reg)")
        ax.grid(True, alpha=0.3)

    for ax in axes[len(offsets):]:
        ax.set_visible(False)
    axes[0].legend(fontsize=8, frameon=False, loc="best")
    fig.suptitle("Median pairwise EVM improvement over deterministic control")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


def write_summary_table(stats_df, out_path):
    stats_df = _load(stats_df)
    table = stats_df.rename(columns={
        "dc_ma": "DC Offset (mA)",
        "power": "lambda_1 (power penalty)",
        "anneal": "Noise Anneal Start",
        "median_delta": "delta",
        "p_value": "p-value",
    })
    table = table[["DC Offset (mA)", "lambda_1 (power penalty)", "Noise Anneal Start",
                   "delta", "ci_lo", "ci_hi", "p-value"]]
    table = table.sort_values(["DC Offset (mA)", "lambda_1 (power penalty)", "Noise Anneal Start"])
    table.to_csv(out_path, index=False)
    return table


if __name__ == "__main__":
    stats_path = Path(sys.argv[1])
    out_dir = stats_path.parent
    plot_delta_vs_anneal(stats_path, out_dir / "delta_vs_anneal.png")
    write_summary_table(stats_path, out_dir / "summary_table.csv")
    print(f"wrote {out_dir / 'delta_vs_anneal.png'} and {out_dir / 'summary_table.csv'}")