import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.colors import TwoSlopeNorm

POWER_COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00"]
PANEL_LABELS = ["(a)", "(b)", "(c)", "(d)"]


def _significance_stars(p_value):
    if p_value < 1e-3:
        return "***"
    if p_value < 1e-2:
        return "**"
    if p_value < 5e-2:
        return "*"
    return ""


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
                    label=f"$\\lambda_1$={power:g}")
        ax.axhline(0, color="black", lw=0.8, linestyle="--")
        ax.text(0.05, 0.95, PANEL_LABELS[panel_index], transform=ax.transAxes, va="top", ha="left")
        ax.set_title(f"{dc_ma} mA")
        ax.set_xlabel("Noise Anneal Start")
        ax.set_ylabel("Median $\\Delta$ EVM% (ctrl $-$ reg)")
        ax.grid(True, alpha=0.3)

    for ax in axes[len(offsets):]:
        ax.set_visible(False)
    axes[0].legend(fontsize=8, frameon=False, loc="best")
    fig.suptitle("Median Pairwise EVM Improvement over Deterministic Control")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


def plot_delta_pvalue_heatmap(stats_df, out_path):
    stats_df = _load(stats_df)
    offsets = sorted(stats_df.dc_ma.unique())
    powers = sorted(stats_df.power.unique())
    anneals = sorted(stats_df.anneal.unique())

    vmax = float(np.abs(stats_df.median_delta).max())
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    n_rows = int(np.ceil(len(offsets) / 2))
    fig = Figure(figsize=(9, 4.8 * n_rows))
    axes = fig.subplots(n_rows, 2, squeeze=False).flatten()

    for panel_index, dc_ma in enumerate(offsets):
        ax = axes[panel_index]
        panel = stats_df[stats_df.dc_ma == dc_ma]
        delta = np.full((len(powers), len(anneals)), np.nan)
        pval = np.full_like(delta, np.nan)
        for _, row in panel.iterrows():
            i = powers.index(row.power)
            j = anneals.index(row.anneal)
            delta[i, j] = row.median_delta
            pval[i, j] = row.p_value

        ax.imshow(delta, cmap="RdYlGn", norm=norm, aspect="auto")

        for i in range(len(powers)):
            for j in range(len(anneals)):
                ax.text(j, i - 0.18, f"{delta[i, j]:+.2f}", ha="center", va="center",
                        fontsize=11, fontweight="bold")
                ax.text(j, i + 0.22, f"p={pval[i, j]:.1e}{_significance_stars(pval[i, j])}",
                        ha="center", va="center", fontsize=8)

        ax.set_xticks(range(len(anneals)))
        ax.set_xticklabels([f"{a:g}" for a in anneals])
        ax.set_yticks(range(len(powers)))
        ax.set_yticklabels([f"{p:g}" for p in powers])
        ax.set_xlabel("Noise Anneal Start")
        ax.set_title(f"{PANEL_LABELS[panel_index]} {dc_ma} mA")
        if panel_index % 2 == 0:
            ax.set_ylabel("Drive Mean Power Penalty $\\lambda_1$")

    for ax in axes[len(offsets):]:
        ax.set_visible(False)

    fig.suptitle("Median $\\Delta$ EVM% (ctrl $-$ reg)\n"
                 "Stars: * p<0.05  ** p<0.01  *** p<0.001", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
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


# to run, write python plot_find_regularization.py /path/to/<run_name>_find_regularization_<timestamp>/stats.csv
if __name__ == "__main__":
    stats_path = Path(sys.argv[1])
    out_dir = stats_path.parent
    plot_delta_vs_anneal(stats_path, out_dir / "delta_vs_anneal.png")
    plot_delta_pvalue_heatmap(stats_path, out_dir / "delta_pvalue_heatmap.png")
    write_summary_table(stats_path, out_dir / "summary_table.csv")
    print(f"wrote {out_dir / 'delta_vs_anneal.png'}, {out_dir / 'delta_pvalue_heatmap.png'} "
          f"and {out_dir / 'summary_table.csv'}")