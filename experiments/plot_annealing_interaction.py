'''Annealing interaction figure for a finished joint sweep, one panel per clip-penalty rho.

Standalone on purpose: it reads only the run tables an experiment already wrote, so a figure
can be restyled and regenerated without touching hardware or retraining anything.

    python experiments/plot_annealing_interaction.py <experiment_dir> [-o out.png]
'''
import argparse
import json
from pathlib import Path

import numpy as np
from matplotlib import colormaps
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Style. Everything the figure looks like is set here so it can be tuned without reading the
# plotting code. The min marker carries full opacity because it is the pareto bound; the max
# and median sit behind it.
RHO_PANEL_ORDER = (0.25, 0.5, 0.75)
WEIGHT_CMAP = "plasma"
CONTROL_COLOR = "#000000"
CONTROL_LABEL = "ctrl: best deterministic channel with no annealing"
CONTROL_HATCH = "//"

# weights share an anneal column, so each is nudged sideways to keep its triple readable
WEIGHT_DODGE_PERCENT = 3.5
BOX_WIDTH = 5

MIN_ALPHA = 1.0
MEDIAN_ALPHA = 0.55
MAX_ALPHA = 0.30
SPAN_LINE_ALPHA = 0.30
PARETO_LINE_ALPHA = 1.0

MIN_MARKER = "o"
MEDIAN_MARKER = "s"
MAX_MARKER = "^"
MARKER_SIZE = 7
PARETO_LINE_WIDTH = 1.8
SPAN_LINE_WIDTH = 1.2

CONTROL_X_GAP_PERCENT = 18.0  # how far right of the 100% column the control column sits
FIGURE_SIZE = (13, 4.6)

STAR_MARKER = "*"
STAR_SIZE = 20
STAR_COLOR = "#d62728"


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def resolve_run_directories(experiment_dir):
    '''Find the (ed_train, ed_val) pair to read. Accepts the parent run directory or the
    ed_val directory itself. Joint-sweep directories win over the older single-axis sweeps
    when a run contains several.'''
    experiment_dir = Path(experiment_dir)

    if experiment_dir.name.startswith("ed_val"):
        parent = experiment_dir.parent
        suffix = experiment_dir.name[len("ed_val"):].rsplit("_", 2)[0]
        candidates = sorted(parent.glob(f"ed_train{suffix}*"))
        if not candidates:
            raise SystemExit(f"no ed_train directory beside {experiment_dir}")
        return candidates[0], experiment_dir

    def pick(prefix):
        found = sorted(experiment_dir.glob(f"{prefix}*"))
        if not found:
            raise SystemExit(f"no {prefix}* directory under {experiment_dir}")
        preferred = [path for path in found if "joint" in path.name]
        return (preferred or found)[0]

    return pick("ed_train"), pick("ed_val")


def load_sweep_table(experiment_dir):
    '''One row per validated E/D: the training settings joined onto its hardware EVM.'''
    train_dir, validation_dir = resolve_run_directories(experiment_dir)
    trained = {row["run_id"]: row for row in read_jsonl(train_dir / "runs.jsonl")}

    table = []
    for row in read_jsonl(validation_dir / "runs.jsonl"):
        settings = trained.get(row["model"])
        if settings is None:
            continue

        table.append({
            "anneal_start": float(settings.get("noise_anneal_start", 1.0)),
            "rho": float(settings.get("clip_penalty_rho") or 0.0),
            "weight": float(settings.get("clip_penalty_weight") or 0.0),
            "seed": settings.get("seed"),
            "channel_form": row.get("channel_form", "unknown"),
            "is_prob": str(row.get("channel_form", "")).startswith("prob"),
            "evm_pct": float(row["evm_pct"]),
        })

    if not table:
        raise SystemExit(f"no validated runs found under {experiment_dir}")
    print(f"read {len(table)} validated runs from {validation_dir.name}")
    return table


def summarize_over_runs(rows):
    '''(min, median, max) of the validation EVM over every run trained at one grid cell.'''
    values = [row["evm_pct"] for row in rows]
    return float(np.min(values)), float(np.median(values)), float(np.max(values))


def dodge_offsets(count):
    '''Symmetric sideways nudges so several weights can share one anneal column.'''
    return (np.arange(count) - (count - 1) / 2.0) * WEIGHT_DODGE_PERCENT


def draw_column(ax, x_position, summary, color):
    minimum, median, maximum = summary
    ax.plot([x_position, x_position], [minimum, maximum], color=color,
            lw=SPAN_LINE_WIDTH, alpha=SPAN_LINE_ALPHA, zorder=1)
    ax.plot(x_position, maximum, marker=MAX_MARKER, color=color, ms=MARKER_SIZE,
            alpha=MAX_ALPHA, linestyle="none", zorder=2)
    ax.plot(x_position, median, marker=MEDIAN_MARKER, color=color, ms=MARKER_SIZE,
            alpha=MEDIAN_ALPHA, linestyle="none", zorder=3)
    ax.plot(x_position, minimum, marker=MIN_MARKER, color=color, ms=MARKER_SIZE,
            alpha=MIN_ALPHA, linestyle="none", zorder=4)

def values_over_runs(cell):
    '''
    Returns the EVM values for the cell for all runs
    '''
    return [row['evm_pct'] for row in cell]

def draw_box(ax, x_pos, values, color, width=BOX_WIDTH, is_control=False):
    '''Fill always encodes the clip weight. Control boxes keep that fill but take a black
    hatched outline, so the deterministic-channel column stays identifiable without giving up
    the weight colour.'''
    line_color = CONTROL_COLOR if is_control else color
    box_plot = ax.boxplot(
        [values],
        positions=[x_pos],
        widths=width,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color=line_color, lw=1.4),
        boxprops=dict(facecolor=color, edgecolor=line_color, alpha=0.3,
                      hatch=CONTROL_HATCH if is_control else None),
        whiskerprops=dict(color=line_color, linestyle="dashed"),
        capprops=dict(color=line_color),
        manage_ticks=False
    )
    return box_plot


def plot_annealing_interaction(table, out_path):
    prob_rows = [row for row in table if row["is_prob"] and row["weight"] > 0.0]
    control_rows = [row for row in table if not row["is_prob"] and row["weight"] > 0.0]

    rhos = [rho for rho in RHO_PANEL_ORDER if any(row["rho"] == rho for row in prob_rows)]
    rhos += sorted({row["rho"] for row in prob_rows} - set(RHO_PANEL_ORDER))
    weights = sorted({row["weight"] for row in prob_rows})

    cmap = colormaps[WEIGHT_CMAP]
    ramp = np.linspace(0.12, 0.78, max(len(weights), 1))
    color_for_weight = {weight: cmap(position) for weight, position in zip(weights, ramp)}
    offset_for_weight = dict(zip(weights, dodge_offsets(len(weights))))

    control_x = 100.0 + CONTROL_X_GAP_PERCENT

    fig = Figure(figsize=FIGURE_SIZE)
    axes = np.atleast_1d(fig.subplots(1, len(rhos), sharey=True))

    best_median = np.inf
    best_info = None   # (x, y, ax, rho, weight)

    for rho, ax in zip(rhos, axes):
        rho_rows = [row for row in prob_rows if row["rho"] == rho]
        anneal_values = sorted({row["anneal_start"] for row in rho_rows})

        for weight in weights:
            color = color_for_weight[weight]
            offset = offset_for_weight[weight]

            median_x, median_y = [], []
            for anneal in anneal_values:
                cell = [row for row in rho_rows
                        if row["anneal_start"] == anneal and row["weight"] == weight]
                if not cell:
                    continue

                evm_values = values_over_runs(cell)
                x_position = anneal * 100.0 + offset
                draw_box(ax, x_position, evm_values, color)
                median_x.append(x_position)
                median = np.median(evm_values)
                if median < best_median:
                    best_median = median
                    best_info = (x_position, median, ax, rho, weight)
                median_y.append(median)

            ax.plot(median_x, median_y, color=color, lw=PARETO_LINE_WIDTH,
                    alpha=PARETO_LINE_ALPHA, zorder=3)

            control_cell = [row for row in control_rows
                            if row["rho"] == rho and row["weight"] == weight]
            if control_cell:
                draw_box(ax, control_x + offset, values_over_runs(control_cell),
                            color, is_control=True)


    


        if control_rows:
            ax.axvline(100.0 + CONTROL_X_GAP_PERCENT / 2, color="#bbbbbb", lw=0.8, ls=":")

        ticks = list(np.array(anneal_values) * 100.0)
        labels = [f"{value * 100:g}" for value in anneal_values]
        if control_rows:
            ticks.append(control_x)
            labels.append("ctrl")
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)

        ax.set_title(f"Annealing Sweep for Rho = {rho:g}")
        ax.set_xlabel("Noise Anneal Start (%)")
        ax.grid(True, alpha=0.3)

    if best_info is not None:
        bx, by, bax, brho, bweight = best_info
        bax.plot(bx, by, marker=STAR_MARKER, color=STAR_COLOR,
                 ms=STAR_SIZE, linestyle="none", zorder=6,
                 markeredgecolor="white", markeredgewidth=0.6)
        bax.annotate(
            rf"Best Median $\rho={brho:g}$, clip_weight = {bweight:g}",
            xy=(bx, by),
            xytext=(6, 8), textcoords="offset points",
            fontsize=8, color=STAR_COLOR, zorder=7,
            ha="left", va="bottom",
        )


    axes[0].set_ylabel("Validation EVM (%)")

    weight_handles = [Line2D([], [], marker=MIN_MARKER, color=color_for_weight[weight],
                             ms=MARKER_SIZE, lw=PARETO_LINE_WIDTH, alpha=MIN_ALPHA,
                             label=f"clip weight = {weight:g}") for weight in weights]
    weight_handles.append(Patch(facecolor="#ffffff", edgecolor=CONTROL_COLOR,
                                hatch=CONTROL_HATCH, label=f"({CONTROL_LABEL})"))
    shape_handles = [
        Patch(facecolor="#444444", edgecolor="#444444", alpha=0.35,
              label="IQR (25–75%) over runs"),
        Line2D([], [], color="#444444", lw=1.4, label="median over runs"),
        Line2D([], [], color="#444444", label="whiskers (min–max)", linestyle="dashed"),
    ]
    axes[0].legend(handles=weight_handles, fontsize=8, frameon=False, loc="upper left")
    axes[-1].legend(handles=shape_handles, fontsize=8, frameon=False, loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment_dir", help="controlled-CDF run directory, or an ed_val_* directory")
    parser.add_argument("-o", "--out", default=None, help="output image path")
    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir)
    out_path = Path(args.out) if args.out else experiment_dir / "annealing_interaction.png"
    plot_annealing_interaction(load_sweep_table(experiment_dir), out_path)


if __name__ == "__main__":
    main()
