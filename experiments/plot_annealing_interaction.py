'''

Annealing interaction figure for a finished joint sweep, one panel per clip-penalty rho.

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
from scipy.stats import spearmanr

# Style. Everything the figure looks like is set here so it can be tuned without reading the
# plotting code. The min marker carries full opacity because it is the pareto bound; the max
# and median sit behind it.
WEIGHT_CMAP = "plasma"
CONTROL_COLOR = "#000000"
CONTROL_LABEL = "ctrl: best deterministic channel with no annealing"
CONTROL_HATCH = "//"

WEIGHT_DODGE_PERCENT = 3
BOX_WIDTH = 3.0
PLOT_CTRL_BASELINE = True

# Tukey whiskers: the last point within 1.5 IQR of the quartiles, with anything beyond drawn
# as a flier. Full range (0, 100) lets one bad seed set the y limits for every panel.
WHISKER_IQR_MULTIPLE = 1.5
SHOW_OUTLIERS = False
OUTLIER_MARKER = "x"
OUTLIER_SIZE = 4

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
PANEL_WIDTH = 4.4

STAR_MARKER = "*"
STAR_SIZE = 20
STAR_COLOR = "#d62728"

SCATTER_FIGURE_SIZE = (7.5, 5.2)
SCATTER_MARKERS = ("o", "s", "^", "D", "v", "P")
SCATTER_MARKER_SIZE = 6
SCATTER_ALPHA = 0.75


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
            "drive_mean_power_weight": float(settings.get("drive_mean_power_weight")
                                             or settings.get("drive_l2_weight") or 0.0),
            "kurtosis_weight": float(settings.get("kurtosis_weight")
                                     or settings.get("drive_kurtosis_weight") or 0.0),
            "drive_rms": settings.get("drive_rms"),
            "drive_kurtosis": settings.get("drive_kurtosis"),
            "drive_papr": settings.get("drive_papr"),
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
    '''Fill always encodes the penalty weight. Control boxes keep that fill but take a black
    hatched outline, so the deterministic-channel column stays identifiable without giving up
    the weight colour.
    '''
    line_color = CONTROL_COLOR if is_control else color
    box_plot = ax.boxplot(
        [values],
        positions=[x_pos],
        widths=width,
        patch_artist=True,
        whis=WHISKER_IQR_MULTIPLE,
        showfliers=SHOW_OUTLIERS,
        medianprops=dict(color=line_color, lw=1.4),
        boxprops=dict(facecolor=color, edgecolor=line_color, alpha=0.3,
                      hatch=CONTROL_HATCH if is_control else None),
        whiskerprops=dict(color=line_color, linestyle="dashed"),
        capprops=dict(color=line_color),
        flierprops=dict(marker=OUTLIER_MARKER, markersize=OUTLIER_SIZE,
                        markeredgecolor=line_color, markerfacecolor="none"),
        manage_ticks=False
    )
    return box_plot


def split_control_rows(table):
    '''(swept prob-channel rows, deterministic-channel control rows). The control rows are
    dropped when PLOT_CTRL_BASELINE is off, so a run that collected a baseline can still be
    plotted without one.'''
    sweep_rows = [row for row in table if row["is_prob"]]
    control_rows = [row for row in table if not row["is_prob"]] if PLOT_CTRL_BASELINE else []
    return sweep_rows, control_rows


def plot_annealing_interaction(table, out_path, trace_field, trace_label, title,
                               include_zero_trace=True):
    '''Boxplot of validation EVM against noise anneal start, one trace per trace_field value,
    with the deterministic-channel runs drawn as a control column past a dotted separator.'''
    def in_scope(row):
        return include_zero_trace or row[trace_field] > 0.0

    sweep_rows, collected_control_rows = split_control_rows(table)
    prob_rows = [row for row in sweep_rows if in_scope(row)]
    control_rows = [row for row in collected_control_rows if in_scope(row)]
    trace_values = sorted({row[trace_field] for row in prob_rows})
    anneal_values = sorted({row["anneal_start"] for row in prob_rows})

    cmap = colormaps[WEIGHT_CMAP]
    ramp = np.linspace(0.12, 0.78, max(len(trace_values), 1))
    color_for_trace = {value: cmap(position) for value, position in zip(trace_values, ramp)}
    offset_for_trace = dict(zip(trace_values, dodge_offsets(len(trace_values))))

    control_x = 100.0 + CONTROL_X_GAP_PERCENT

    fig = Figure(figsize=(PANEL_WIDTH * 2, FIGURE_SIZE[1]))
    ax = fig.subplots()

    best_median = np.inf
    best_info = None

    for trace_value in trace_values:
        color = color_for_trace[trace_value]
        offset = offset_for_trace[trace_value]

        median_x, median_y = [], []
        for anneal in anneal_values:
            cell = [row for row in prob_rows
                    if row["anneal_start"] == anneal and row[trace_field] == trace_value]
            if not cell:
                continue

            evm_values = values_over_runs(cell)
            x_position = anneal * 100.0 + offset
            draw_box(ax, x_position, evm_values, color)
            median_x.append(x_position)
            median = np.median(evm_values)
            if median < best_median:
                best_median = median
                best_info = (x_position, median,
                             f"Best Median {trace_label} = {trace_value:g}")
            median_y.append(median)

        ax.plot(median_x, median_y, color=color, lw=PARETO_LINE_WIDTH,
                alpha=PARETO_LINE_ALPHA, zorder=3)

        control_cell = [row for row in control_rows if row[trace_field] == trace_value]
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

    ax.set_title(title)
    ax.set_xlabel("Noise Anneal Start (%)")
    ax.set_ylabel("Validation EVM (%)")
    ax.grid(True, alpha=0.3)

    if best_info is not None:
        best_x, best_y, best_description = best_info
        ax.plot(best_x, best_y, marker=STAR_MARKER, color=STAR_COLOR,
                ms=STAR_SIZE, linestyle="none", zorder=6,
                markeredgecolor="white", markeredgewidth=0.6)
        ax.annotate(best_description, xy=(best_x, best_y), xytext=(6, 8),
                    textcoords="offset points", fontsize=8, color=STAR_COLOR,
                    zorder=7, ha="left", va="bottom")

    handles = [Line2D([], [], marker=MIN_MARKER, color=color_for_trace[value],
                      ms=MARKER_SIZE, lw=PARETO_LINE_WIDTH, alpha=MIN_ALPHA,
                      label=f"{trace_label} = {value:g}") for value in trace_values]
    if control_rows:
        handles.append(Patch(facecolor="#ffffff", edgecolor=CONTROL_COLOR,
                             hatch=CONTROL_HATCH, label=f"({CONTROL_LABEL})"))
    handles += [
        Patch(facecolor="#444444", edgecolor="#444444", alpha=0.35,
              label="IQR (25-75%) over runs"),
        Line2D([], [], color="#444444", lw=1.4, label="median over runs"),
        Line2D([], [], color="#444444", linestyle="dashed",
               label=f"whiskers ({WHISKER_IQR_MULTIPLE:g} x IQR)"),
    ]
    if SHOW_OUTLIERS:
        handles.append(Line2D([], [], color="#444444", linestyle="none", marker=OUTLIER_MARKER,
                              markersize=OUTLIER_SIZE, markerfacecolor="none", label="outlier run"))
    ax.legend(handles=handles, fontsize=8, frameon=False, loc="upper left")

    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"wrote {out_path}")


# achieved drive/channel properties worth scattering against EVM, as (field, label, log x)
SCATTER_METRICS = (
    ("drive_rms", "Encoder Drive RMS", False),
    ("drive_kurtosis", "Encoder Drive Kurtosis", False),
    ("drive_papr", "Encoder Drive PAPR", False),
)

DC_OFFSET = '120 mA'

PENALTY_LABELS = (
    ("drive_mean_power_weight", f"Drive Mean Power at {DC_OFFSET}"),
    ("kurtosis_weight", "drive kurtosis penalty"),
)


def penalty_family(row):
    for field, label in PENALTY_LABELS:
        if row.get(field, 0.0) > 0.0:
            return label
    return "no penalty"


def plot_metric_vs_evm(table, out_path, x_field, x_label, log_x=True):
    '''An achieved drive or channel property against hardware EVM, one point per validated
    E/D, coloured by which penalty produced it. Runs with the penalty off are measured the
    same way, so the unpenalized reference sits on the same axis.'''
    rows = [row for row in table if row["is_prob"] and row.get(x_field) is not None]
    if not rows:
        raise SystemExit(f"no runs recorded {x_field}")

    families = sorted({penalty_family(row) for row in rows})
    cmap = colormaps[WEIGHT_CMAP]
    ramp = np.linspace(0.12, 0.85, max(len(families), 1))
    color_for_family = dict(zip(families, [cmap(position) for position in ramp]))
    marker_for_family = dict(zip(families, SCATTER_MARKERS))

    fig = Figure(figsize=SCATTER_FIGURE_SIZE)
    ax = fig.subplots()

    for family in families:
        cell = [row for row in rows if penalty_family(row) == family]
        ax.plot([row[x_field] for row in cell], [row["evm_pct"] for row in cell],
                linestyle="none", marker=marker_for_family[family],
                ms=SCATTER_MARKER_SIZE, color=color_for_family[family],
                alpha=SCATTER_ALPHA, markeredgecolor="white", markeredgewidth=0.4,
                label=family)

    correlation, p_value = spearmanr([row[x_field] for row in rows],
                                     [row["evm_pct"] for row in rows])

    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Validation EVM (%)")
    ax.set_title(f"{x_label} Vs. Hardware EVM over {len(rows)} Runs\n"
                 f"Spearman r = {correlation:.3f}, p = {p_value:.3g}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, frameon=False, loc="best")

    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"wrote {out_path}")


def penalty_sweep_families(table):
    '''Which penalties this sweep varied, as {field: label}. Read off the table rather
    than the config so a finished sweep can be re-rendered without it.

    A penalty held at one value across every run is a standing constraint rather than a swept
    axis, so it gets no figure of its own. That is the usual case for the kurtosis target once
    it is pinned, while mean power is swept.'''
    return {field: label for field, label in PENALTY_LABELS
            if len({row[field] for row in table}) > 1}


def pinned_combined_mean_power_weight(table):
    '''The mean-power weight kurtosis was stacked on top of, or None if the sweep had no such family.'''
    if len({row["kurtosis_weight"] for row in table}) <= 1:
        return None

    weights = {row["drive_mean_power_weight"] for row in table
               if row["kurtosis_weight"] > 0.0 and row["drive_mean_power_weight"] > 0.0}
    if not weights:
        return None
    if len(weights) > 1:
        raise ValueError("expected one pinned mean-power weight beneath the kurtosis sweep, found "
                         f"{sorted(weights)}")
    return weights.pop()


def plot_one_family(table, out_path, field, title, trace_label=None):
    '''One penalty family's interaction figure with an exact title, for publication. The other
    families are held at 0 exactly as in the automatic figures, so this is the same plot with
    the caption text under the caller's control rather than generated.'''
    families = penalty_sweep_families(table)
    if field not in families:
        raise ValueError(f"{field} was not swept in this run; varied fields are "
                         f"{sorted(families) or 'none'}")

    others = [other for other in families if other != field]
    plot_annealing_interaction(
        [row for row in table if all(row[other] == 0.0 for other in others)],
        out_path,
        trace_field=field, trace_label=trace_label or families[field],
        title=title)


def plot_penalty_sweep_figures(table, out_dir):
    '''Every figure the penalty sweep produces: one interaction plot per penalty family, the
    combined mean-power-plus-kurtosis plot, and the achieved-drive scatters.'''
    # the control turns off penalties the swept arm pins, so it would otherwise read as an
    # extra swept family and filter the whole sweep out of its own figure
    sweep_rows, control_rows = split_control_rows(table)

    families = penalty_sweep_families(sweep_rows)
    if not families:
        raise ValueError("this run table varied no penalty weight, so there is nothing to sweep over")

    # one figure per family, holding the others at 0 so the no-penalty runs appear in every
    # figure as the shared reference
    for field, label in families.items():
        others = [other for other in families if other != field]
        plot_annealing_interaction(
            [row for row in sweep_rows if all(row[other] == 0.0 for other in others)] + control_rows,
            Path(out_dir) / f"{field}_interaction.png",
            trace_field=field, trace_label=label,
            title=f"Penalty Sweep: {label}")

    # kurtosis on top of a pinned mean power, where amplitude is held fixed and drive shape is the only
    # thing varying across the traces
    combined_mean_power_weight = pinned_combined_mean_power_weight(sweep_rows)
    if combined_mean_power_weight is not None:
        plot_annealing_interaction(
            [row for row in sweep_rows
             if row["drive_mean_power_weight"] == combined_mean_power_weight] + control_rows,
            Path(out_dir) / "combined_mean_power_kurtosis_interaction.png",
            trace_field="kurtosis_weight",
            trace_label=f"kurtosis weight at mean power {combined_mean_power_weight:g}",
            title=f"Kurtosis on top of drive mean power {combined_mean_power_weight:g}")

    scatter_rows = sweep_rows + control_rows
    for field, label, log_x in SCATTER_METRICS:
        if any(row.get(field) is not None for row in scatter_rows):
            plot_metric_vs_evm(scatter_rows, Path(out_dir) / f"{field}_vs_evm.png",
                               field, label, log_x=log_x)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment_dir", help="controlled-CDF run directory, or an ed_val_* directory")
    parser.add_argument("-o", "--out", default=None, help="output image path")
    parser.add_argument("--field", default=None,
                        help="render only this penalty family, e.g. drive_mean_power_weight")
    parser.add_argument("--title", default=None,
                        help="exact figure title, replacing the generated one. requires --field. "
                             "matplotlib mathtext works, e.g. 'EVM vs annealing at 120 mA'")
    parser.add_argument("--trace-label", default=None,
                        help="legend text for each trace, replacing the generated one")
    args = parser.parse_args()

    if args.title and not args.field:
        parser.error("--title needs --field, since a run produces one figure per penalty family")

    experiment_dir = Path(args.experiment_dir)
    table = load_sweep_table(experiment_dir)

    if args.field:
        out_path = Path(args.out) if args.out else experiment_dir / f"{args.field}_interaction.png"
        plot_one_family(table, out_path, args.field,
                        title=args.title or f"Penalty Sweep: {args.field}",
                        trace_label=args.trace_label)
        return

    out_path = Path(args.out) if args.out else experiment_dir / "annealing_interaction.png"
    plot_penalty_sweep_figures(table, out_path.parent)


if __name__ == "__main__":
    main()
