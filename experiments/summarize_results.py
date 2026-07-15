"""
Aggregate multi-DC-offset results and generate paper figures.

Edit RUN_CONFIGS and PLOT_PATH at the top, then:
    cd <repo_root>/experiments
    python summarize_results.py

Each RUN_CONFIG entry must point to:
  - channel_exp_dir  : channel-model grid-search output folder
  - ed_exp_dir       : encoder-decoder grid-search output folder (E/D checkpoints;
                       needed to replay validation bursts for the predicted-vs-actual
                       EVM transfer plot)
  - ed_val_exp_dir   : ed_validation output folder (must pair with the
                       encoder-decoder grid search that used the above channel models)
  - dataset_path     : the zarr dataset the channel models were trained on
                       (must match format: new burst format or legacy symbol-only)
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import zarr
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.grid_search.adapters import MODEL_REGISTRY
from modules.utils import calculate_per_burst_rrmse_pct_loss

# USER CONFIG
RUN_CONFIGS = [
    {
        "label": "50 mA",
        "dc_ma": 50,
        "channel_exp_dir": "data/experiments/train_and_validate/dawn_fold_channel_models_20260714_1613",
        "ed_exp_dir":      "data/experiments/train_and_validate/dawn_fold_encoder_decoder_20260714_2242",
        "ed_val_exp_dir":  "data/experiments/train_and_validate/dawn_fold_ed_validation_20260714_2326",
        "dataset_path":    "data/sweeps/tall_river_dc0.05A_fmin300000_fmax7.6e+06_20260713_1711.zarr",
    },
    # add more DC offsets here, e.g.:
    # {
    #     "label": "80mA",
    #     "dc_ma": 80,
    #     "channel_exp_dir": "data/experiments/train_and_validate/channel_models_YYYYMMDD_HHMM",
    #     "ed_val_exp_dir":  "data/experiments/train_and_validate/ed_validation_YYYYMMDD_HHMM",
    #     "dataset_path":    "data/sweeps/<filename>.zarr",
    # },
]

PLOT_PATH       = Path(__file__).resolve().parent.parent / "data/plots"
DEVICE          = "cpu"
N_POWER_BINS    = 10       # bins for plot 1
MAX_QQ_SAMPLES  = 10_000  # downsample for Q-Q speed
# Style (matches make_figures.ipynb)
_FONT = 7
_SMALL = 5
_mm = 1 / 25.4
_fw  = 88  * _mm   # single-column width
_fh  = 88  * _mm
_fw2 = 180 * _mm   # double-column width

plt.rcParams.update({
    "font.size": _FONT,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
    "axes.labelsize": _FONT,
    "axes.titlesize": _FONT,
    "xtick.labelsize": _FONT,
    "ytick.labelsize": _FONT,
    "legend.fontsize": _FONT,
    "figure.dpi": 300,
    "lines.linewidth": 1.0,
    "lines.markersize": 4,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "axes.linewidth": 0.8,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.edgecolor": "black",
})
sns.set_style("whitegrid")
plt.rcParams.update({
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.spines.left": True,
    "axes.spines.bottom": True,
    "axes.edgecolor": "black",
    "axes.linewidth": 0.8,
})

_CMAP = plt.get_cmap("viridis")   # sequential scales only (frequency/power gradients)

# Shared style dictionary: (model_family, distribution) -> plot kwargs.
# Used identically across all figures so the same model type always looks the same.
# Okabe-Ito palette (colorblind-safe), assigned semantically: near-black = GMP
# baseline, cool hues = nonprob (blue TCN, green LRU), warm hues = prob
# (orange/vermillion TCN, purple/sky LRU). Adjacent-pair CVD separation validated
# (worst ΔE 19 under protan/deutan/tritan simulation vs floor 8 / target 12).
MODEL_STYLES: dict[tuple[str, str], dict] = {
    ("gmp", "none"):       {"color": "#333333", "marker": "o", "linestyle": "-"},
    ("tcn", "none"):       {"color": "#0072B2", "marker": "^", "linestyle": "--"},
    ("tcn", "gaussian"):   {"color": "#E69F00", "marker": "s", "linestyle": "-."},
    ("tcn", "students_t"): {"color": "#D55E00", "marker": "D", "linestyle": ":"},
    ("lru", "none"):       {"color": "#009E73", "marker": "p", "linestyle": "--"},
    ("lru", "gaussian"):   {"color": "#CC79A7", "marker": "X", "linestyle": "-."},
    ("lru", "students_t"): {"color": "#56B4E9", "marker": "v", "linestyle": ":"},
}
_FALLBACK_STYLES = [
    {"color": "#F0E442", "marker": "*", "linestyle": "-"},
    {"color": "#000000", "marker": "+", "linestyle": "--"},
]
_extra_style_cache: dict[tuple, dict] = {}


def _get_style(model: str, dist: str) -> dict:
    """Return the shared plot style for a (model, distribution) key."""
    key = (model, dist or "none")
    if key in MODEL_STYLES:
        return MODEL_STYLES[key]
    if key not in _extra_style_cache:
        style_index = len(_extra_style_cache) % len(_FALLBACK_STYLES)
        _extra_style_cache[key] = _FALLBACK_STYLES[style_index]
    return _extra_style_cache[key]


REPO_ROOT = Path(__file__).resolve().parent.parent


def _resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


# DATA HELPERS
def _read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_channel_runs(channel_exp_dir: str) -> list[dict]:
    """Load runs.jsonl from a channel-model grid search and annotate with is_prob."""
    rows = _read_jsonl(_resolve(channel_exp_dir) / "runs.jsonl")
    for row in rows:
        distribution = row.get("distribution") or "none"
        row["is_tcn"] = row.get("model") == "tcn"
        row["is_prob"] = row["is_tcn"] and distribution in ("gaussian", "students_t")
    return rows


def load_adapter(run: dict, channel_exp_dir: str, device: str = DEVICE):
    """Rebuild a trained channel-model adapter from its run directory."""
    run_dir = _resolve(channel_exp_dir) / "runs" / run["run_id"]
    with open(run_dir / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    return MODEL_REGISTRY[run["model"]].load(cfg["params"], run_dir / "model.pt", device)


def load_dataset(dataset_path: str, device: str = DEVICE):
    """
    Return (X_sent, Y_recv, ks_indices, symbol_length) from a zarr dataset.
    Supports both the new burst format and the legacy symbol-only format.
    X and Y are symbol-length arrays (preamble and CP stripped when present).
    """
    root = zarr.open_group(_resolve(dataset_path), mode="r")
    attrs = dict(root.attrs)
    active_carrier_indices = np.array(attrs["active_carrier_indices"])
    cyclic_prefix_length = int(attrs["cyclic_prefix_length"])

    if "sent_burst" in root:
        preamble_length = int(attrs.get("preamble_length", 0))
        sent_burst = torch.tensor(root["sent_burst"][:], dtype=torch.float32, device=device)
        received_burst = torch.tensor(root["received_burst"][:], dtype=torch.float32, device=device)
        symbol_offset = preamble_length + cyclic_prefix_length
        X, Y = sent_burst[:, symbol_offset:], received_burst[:, symbol_offset:]
    else:
        X = torch.tensor(root["sent_baseband"][:], dtype=torch.float32, device=device)
        Y = torch.tensor(root["received_baseband"][:], dtype=torch.float32, device=device)

    return X, Y, active_carrier_indices, X.shape[1]


def _predict_mean(adapter, X: torch.Tensor) -> torch.Tensor:
    """Get the mean prediction from any adapter type (handles prob/nonprob TCN and GMP)."""
    out = adapter.predict(X)
    return out[1] if isinstance(out, tuple) else out


def _best_by(runs: list[dict], key: str, *fallback_keys: str) -> dict | None:
    """Return the run with the minimum value for key, trying fallback_keys in
    order (legacy metric names from older experiments) when key is absent."""
    def score(r):
        for k in (key, *fallback_keys):
            if r.get(k) is not None:
                return float(r[k])
        return float("inf")
    return min(runs, key=score, default=None)


def per_trial_evm(ed_val_exp_dir: str, ks_indices: np.ndarray) -> dict[str, np.ndarray]:
    """
    Compute per-trial EVM% from the time-domain signals stored in validation.zarr.
    Returns {run_id: array of shape [num_trials]}.

    sent_time and received_time are symbol-only (no CP, no preamble), so we
    rfft directly and index by ks_indices.
    """
    root = zarr.open_group(_resolve(ed_val_exp_dir) / "validation.zarr", mode="r")
    evm_by_run = {}
    for run_id in root.keys():
        group = root[run_id]
        sent = group["sent_time"][:]     # (num_trials, symbol_length)
        received = group["received_time"][:]

        trial_evms = []
        for trial in range(sent.shape[0]):
            sent_spectrum = np.fft.rfft(sent[trial], norm="ortho")[ks_indices]
            received_spectrum = np.fft.rfft(received[trial], norm="ortho")[ks_indices]
            signal_power = np.mean(np.abs(sent_spectrum) ** 2)
            residual_power = np.mean(np.abs(sent_spectrum - received_spectrum) ** 2)
            trial_evms.append(float(np.sqrt(residual_power / (signal_power + 1e-12)) * 100))
        evm_by_run[run_id] = np.array(trial_evms)
    return evm_by_run


def _binned_rrmse(
    X: torch.Tensor,
    Y: torch.Tensor,
    adapter,
    n_bins: int = N_POWER_BINS,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-bin per-burst RRMSE (%) binned by sent-power (mean squared amplitude)."""
    power = X.square().mean(dim=-1)
    bin_edges = torch.linspace(power.min(), power.max(), n_bins + 1)
    bin_index = torch.bucketize(power, bin_edges)
    bin_centers = (0.5 * (bin_edges[:-1] + bin_edges[1:])).cpu().numpy()

    predicted = _predict_mean(adapter, X)
    receptive_field = int(getattr(adapter.model, "receptive_field", 0))
    target, predicted = Y[..., receptive_field:], predicted[..., receptive_field:]

    rrmse_per_bin = []
    for bin_number in range(n_bins):
        in_bin = bin_index == bin_number
        if in_bin.any():
            rrmse_per_bin.append(float(calculate_per_burst_rrmse_pct_loss(target[in_bin], predicted[in_bin])))
        else:
            rrmse_per_bin.append(float("nan"))
    return bin_centers, np.array(rrmse_per_bin)


# PLOT 1 & 2 shared helpers
def _model_type_label(model: str, dist: str) -> str:
    dist_name = {"none": "nonprob", "gaussian": "Gaussian", "students_t": "Student's-t"}.get(dist, dist)
    return f"{model.upper()} {dist_name}"


def _model_type_sort_key(mt: tuple[str, str]) -> tuple:
    return ({"gmp": 0, "tcn": 1}.get(mt[0], 99),
            {"none": 0, "gaussian": 1, "students_t": 2}.get(mt[1], 99))


def _discover_model_types(run_configs: list[dict]) -> list[tuple[str, str]]:
    seen = []
    for cfg in run_configs:
        for r in load_channel_runs(cfg["channel_exp_dir"]):
            mt = (r["model"], r.get("distribution") or "none")
            if mt not in seen:
                seen.append(mt)
    return sorted(seen, key=_model_type_sort_key)


# PLOT 1: Val RRMSE vs Sent Power - 2×2 subplot grid, one panel per DC bias
# Model type -> style from MODEL_STYLES (color/marker/linestyle), consistent across panels.
# Shared x/y axes so panels are directly comparable.

_PANEL_LABELS = ["(a)", "(b)", "(c)", "(d)"]

def plot_val_rrmse_vs_power(run_configs: list[dict]) -> None:
    model_types = _discover_model_types(run_configs)
    n = len(run_configs)

    fig, axes = plt.subplots(2, 2, figsize=(_fw2, 2 * _fh),
                             sharex=True, sharey=True,
                             constrained_layout=True)
    axes_flat = axes.flatten()

    for panel_index, cfg in enumerate(run_configs):
        ax = axes_flat[panel_index]
        X, Y, _, _ = load_dataset(cfg["dataset_path"])
        runs = load_channel_runs(cfg["channel_exp_dir"])

        for model, dist in model_types:
            style = _get_style(model, dist)
            candidates = [r for r in runs
                          if r["model"] == model and (r.get("distribution") or "none") == dist]
            if not candidates:
                continue
            best    = _best_by(candidates, "val_per_burst_rrmse_pct", "val_rrmse_pct")
            adapter = load_adapter(best, cfg["channel_exp_dir"])
            centers, rrmse = _binned_rrmse(X, Y, adapter)
            mask = ~np.isnan(rrmse)
            if not mask.any():
                continue
            ax.plot(
                centers[mask], rrmse[mask],
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                markersize=2.5,
                label=_model_type_label(model, dist),
            )

        ax.set_title(cfg["label"], fontsize=_FONT)
        ax.grid(True)
        ax.text(0.05, 0.95, _PANEL_LABELS[panel_index], transform=ax.transAxes,
                fontsize=_FONT, va="top", ha="left")
        ax.tick_params(labelbottom=True)
        if panel_index % 2 == 0:
            ax.set_ylabel("Val RRMSE (%)")
        ax.set_xlabel("Sent Power (Mean Squared Amplitude)")

    axes_flat[0].legend(fontsize=_FONT, handlelength=3, labelspacing=0.5)

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    plt.savefig(PLOT_PATH / "val_rrmse_vs_power.svg", format="svg", bbox_inches="tight")
    plt.savefig(PLOT_PATH / "val_rrmse_vs_power.png", bbox_inches="tight")
    plt.show()


# PLOT 2: Pareto - channel model params vs experimental EVM%
# 2×2 subplot grid, one panel per DC bias (mirrors plot 1).
# One trace per (model, distribution) type using MODEL_STYLES.
# Lines connect points ordered by param count; error bars = 2× SE (≈ 95% CI).

def plot_pareto_evm(run_configs: list[dict]) -> None:
    n = len(run_configs)

    fig, axes = plt.subplots(2, 2, figsize=(_fw2, 2 * _fh),
                             sharex=True, sharey=True,
                             constrained_layout=True)
    axes_flat = axes.flatten()

    for panel_index, cfg in enumerate(run_configs):
        ax = axes_flat[panel_index]
        _, _, active_carrier_indices, _ = load_dataset(cfg["dataset_path"])
        channel_by_run_id = {row["run_id"]: row for row in load_channel_runs(cfg["channel_exp_dir"])}
        validation_rows = _read_jsonl(_resolve(cfg["ed_val_exp_dir"]) / "runs.jsonl")
        trial_evms = per_trial_evm(cfg["ed_val_exp_dir"], active_carrier_indices)

        # per channel model, keep the lowest-mean-EVM E/D validated against it
        best_ed_per_channel: dict[str, dict] = {}
        for validation_row in validation_rows:
            channel_run_id = validation_row.get("channel_run_id")
            if channel_run_id is None:
                continue
            trials = trial_evms.get(validation_row["run_id"], np.array([validation_row.get("evm_pct", float("nan"))]))
            mean_evm = float(np.nanmean(trials))
            if channel_run_id not in best_ed_per_channel or mean_evm < best_ed_per_channel[channel_run_id]["mean_evm"]:
                best_ed_per_channel[channel_run_id] = {"mean_evm": mean_evm, "trials": trials,
                                                       "channel_run": channel_by_run_id.get(channel_run_id)}

        traces: dict[tuple, list] = {}
        for channel_run_id, entry in best_ed_per_channel.items():
            channel_run = entry["channel_run"]
            if channel_run is None:
                continue
            trials = entry["trials"]
            mean_evm = float(np.nanmean(trials))
            standard_error = (float(np.nanstd(trials, ddof=1)) / np.sqrt(len(trials))
                              if len(trials) > 1 else 0.0)
            key = (channel_run.get("model", "unknown"), channel_run.get("distribution") or "none")
            traces.setdefault(key, []).append({
                "num_params": channel_run["num_params"],
                "mean_evm": mean_evm,
                "standard_error": standard_error,
            })

        for key in sorted(traces.keys(), key=_model_type_sort_key):
            model, dist = key
            style = _get_style(model, dist)
            points = sorted(traces[key], key=lambda point: point["num_params"])
            param_counts = np.array([point["num_params"] for point in points], dtype=float)
            mean_evms = np.array([point["mean_evm"] for point in points])
            standard_errors = np.array([point["standard_error"] for point in points])

            # faint raw points; solid line tracks the pareto front (best EVM so far
            # with increasing parameter count)
            ax.errorbar(param_counts, mean_evms, yerr=2 * standard_errors,
                        color=style["color"], marker=style["marker"],
                        markersize=3, capsize=2, capthick=0.5,
                        elinewidth=0.5, linestyle="none", zorder=2, alpha=0.3)
            pareto_evms = np.minimum.accumulate(mean_evms)
            ax.plot(param_counts, pareto_evms, color=style["color"], linestyle="-",
                    linewidth=1.2, alpha=0.85, zorder=3,
                    drawstyle="steps-post", label=_model_type_label(model, dist))
            on_front = mean_evms == pareto_evms
            ax.plot(param_counts[on_front], mean_evms[on_front], linestyle="none",
                    marker=style["marker"], markersize=3,
                    color=style["color"], alpha=0.85, zorder=4)

        ax.set_title(cfg["label"], fontsize=_FONT)
        ax.grid(True)
        ax.text(0.05, 0.95, _PANEL_LABELS[panel_index], transform=ax.transAxes,
                fontsize=_FONT, va="top", ha="left")
        ax.tick_params(labelbottom=True)
        if panel_index % 2 == 0:
            ax.set_ylabel("Experimental EVM (%)")
        ax.set_xlabel("Channel Model Parameter Count")

    axes_flat[0].legend(fontsize=_SMALL, handlelength=2, markerscale=0.8,
                        labelspacing=0.3, borderpad=0.4)

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    plt.savefig(PLOT_PATH / "pareto_evm.svg", format="svg", bbox_inches="tight")
    plt.savefig(PLOT_PATH / "pareto_evm.png", bbox_inches="tight")
    plt.show()


# PLOT 2b: predicted vs actual actual EVM per validated E/D run
# Predicted = the E/D's own training environment evaluated on the exact validation
# bursts: encoder -> frozen channel model (noisy forward for prob models, so their
# prediction includes the noise they model) -> decoder. A calibrated environment
# puts runs on the y=x diagonal; deterministic environments predict optimistically.

_ED_ARCH_KEYS = ("nlayers", "dilation_base", "kernel_size", "hidden_channels")


def _form_style(channel_form: str) -> dict:
    form_to_key = {
        "gmp": ("gmp", "none"),
        "nonprob TCN": ("tcn", "none"),
        "prob TCN": ("tcn", "gaussian"),
        "nonprob LRU": ("lru", "none"),
        "prob LRU": ("lru", "gaussian"),
    }
    return _get_style(*form_to_key.get(channel_form, (channel_form, "none")))


def _replay_predicted_trial_evms(cfg) -> dict[str, np.ndarray]:
    """Per validated run: predicted per-trial EVM% from replaying its validation
    sent bursts through encoder -> frozen channel model -> decoder."""
    from modules.models import TCN

    root = zarr.open_group(_resolve(cfg["dataset_path"]), mode="r")
    active_carrier_indices = np.array(root.attrs["active_carrier_indices"])
    cyclic_prefix_length = int(root.attrs["cyclic_prefix_length"])
    clip_threshold = 3.0

    validation = zarr.open_group(_resolve(cfg["ed_val_exp_dir"]) / "validation.zarr", mode="r")
    validation_rows = _read_jsonl(_resolve(cfg["ed_val_exp_dir"]) / "runs.jsonl")
    encoder_decoder_rows = {row["run_id"]: row for row in _read_jsonl(_resolve(cfg["ed_exp_dir"]) / "runs.jsonl")}
    channel_by_run_id = {row["run_id"]: row for row in load_channel_runs(cfg["channel_exp_dir"])}

    channel_cache: dict[str, object] = {}
    predicted_evm = {}
    for row in validation_rows:
        run_id, ed_run_id, channel_run_id = row["run_id"], row["model"], row.get("channel_run_id")
        if run_id not in validation or ed_run_id not in encoder_decoder_rows or channel_run_id not in channel_by_run_id:
            continue

        architecture = {key: encoder_decoder_rows[ed_run_id][key] for key in _ED_ARCH_KEYS}
        encoder = TCN(**architecture).to(DEVICE)
        decoder = TCN(**architecture).to(DEVICE)
        checkpoint = torch.load(_resolve(cfg["ed_exp_dir"]) / "runs" / ed_run_id / "model.pt",
                                map_location=DEVICE, weights_only=True)
        encoder.load_state_dict(checkpoint["encoder"])
        decoder.load_state_dict(checkpoint["decoder"])
        encoder.eval()
        decoder.eval()

        if channel_run_id not in channel_cache:
            channel_cache[channel_run_id] = load_adapter(channel_by_run_id[channel_run_id], cfg["channel_exp_dir"])
        channel = channel_cache[channel_run_id]

        symbol = torch.tensor(validation[run_id]["sent_time"][:].astype(np.float32), device=DEVICE)
        symbol_length = symbol.shape[1]
        burst = torch.hstack([symbol[:, -cyclic_prefix_length:], symbol])  # [CP | symbol]
        with torch.no_grad():
            encoded = encoder(burst).clamp(-clip_threshold, clip_threshold)
            channel_output = channel.predict(encoded)
            if isinstance(channel_output, tuple):   # prob: noisy forward
                channel_output = channel_output[0]
            decoded = decoder(channel_output)[:, -symbol_length:].cpu().numpy()

        sent_spectrum = np.fft.fft(symbol.cpu().numpy().astype(np.float64), norm="ortho", axis=1)[:, active_carrier_indices]
        decoded_spectrum = np.fft.fft(decoded.astype(np.float64), norm="ortho", axis=1)[:, active_carrier_indices]
        signal_power = np.mean(np.abs(sent_spectrum) ** 2, axis=1)
        predicted_evm[run_id] = np.sqrt(np.mean(np.abs(decoded_spectrum - sent_spectrum) ** 2, axis=1) / signal_power) * 100
    return predicted_evm


def plot_predicted_vs_actual_evm(run_configs: list[dict]) -> None:
    n = len(run_configs)
    fig, axes = plt.subplots(2, 2, figsize=(_fw2, 2 * _fh),
                             sharex=True, sharey=True, constrained_layout=True)
    axes_flat = axes.flatten()

    for panel_index, cfg in enumerate(run_configs):
        ax = axes_flat[panel_index]
        if "ed_exp_dir" not in cfg:
            print(f"[predicted_vs_actual] {cfg['label']}: no ed_exp_dir; skipping")
            continue
        _, _, active_carrier_indices, _ = load_dataset(cfg["dataset_path"])
        predicted = _replay_predicted_trial_evms(cfg)
        actual = per_trial_evm(cfg["ed_val_exp_dir"], active_carrier_indices)
        validation_rows = _read_jsonl(_resolve(cfg["ed_val_exp_dir"]) / "runs.jsonl")

        seen_forms = set()
        axis_limits = [np.inf, -np.inf]
        for row in validation_rows:
            run_id, form = row["run_id"], row["channel_form"]
            if run_id not in predicted or run_id not in actual:
                continue
            predicted_evm, actual_evm = float(np.nanmean(predicted[run_id])), float(np.nanmean(actual[run_id]))
            style = _form_style(form)
            ax.scatter([predicted_evm], [actual_evm], s=14, color=style["color"], marker=style["marker"],
                       label=form if form not in seen_forms else None, zorder=3)
            seen_forms.add(form)
            axis_limits = [min(axis_limits[0], predicted_evm, actual_evm),
                           max(axis_limits[1], predicted_evm, actual_evm)]

        pad = 0.08 * (axis_limits[1] - axis_limits[0] + 1e-9)
        lo, hi = axis_limits[0] - pad, axis_limits[1] + pad
        ax.plot([lo, hi], [lo, hi], color="grey", ls="--", lw=0.8, zorder=1)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_title(cfg["label"], fontsize=_FONT)
        ax.grid(True)
        ax.text(0.05, 0.95, _PANEL_LABELS[panel_index], transform=ax.transAxes,
                fontsize=_FONT, va="top", ha="left")
        ax.tick_params(labelbottom=True)
        if panel_index % 2 == 0:
            ax.set_ylabel("Actual EVM (%)")
        ax.set_xlabel("Predicted EVM (%)")

    axes_flat[0].legend(fontsize=_SMALL, handlelength=1.2, markerscale=0.9,
                        labelspacing=0.3, borderpad=0.4)
    for ax in axes_flat[n:]:
        ax.set_visible(False)

    plt.savefig(PLOT_PATH / "predicted_vs_actual_evm.svg", format="svg", bbox_inches="tight")
    plt.savefig(PLOT_PATH / "predicted_vs_actual_evm.png", bbox_inches="tight")
    plt.show()


# PLOT 2c: ECDF of actual EVM per channel form
# P(hw EVM <= x) across validated E/D runs: "more likely to perform well" as a
# literal CDF statement. A form whose curve sits up-and-left stochastically
# dominates, even if another form owns the single best run.

def plot_hw_evm_ecdf(run_configs: list[dict]) -> None:
    n = len(run_configs)
    fig, axes = plt.subplots(2, 2, figsize=(_fw2, 2 * _fh),
                             sharex=True, sharey=True, constrained_layout=True)
    axes_flat = axes.flatten()

    for panel_index, cfg in enumerate(run_configs):
        ax = axes_flat[panel_index]
        val_rows = _read_jsonl(_resolve(cfg["ed_val_exp_dir"]) / "runs.jsonl")
        by_form: dict[str, list[float]] = {}
        for row in val_rows:
            by_form.setdefault(row["channel_form"], []).append(float(row["evm_pct"]))

        legend_handles = []
        for form in sorted(by_form):
            style = _form_style(form)
            evms = np.sort(by_form[form])
            quantiles = np.arange(1, len(evms) + 1) / len(evms)
            ax.step(np.concatenate([[evms[0]], evms]),
                    np.concatenate([[0.0], quantiles]),
                    where="post", color=style["color"], lw=1.2)
            ax.plot(evms, quantiles, linestyle="none", marker=style["marker"],
                    color=style["color"], ms=3)
            legend_handles.append(plt.Line2D(
                [], [], color=style["color"], lw=1.2, marker=style["marker"],
                markersize=3, label=f"{form} (n={len(evms)})"))

        ax.set_title(cfg["label"], fontsize=_FONT)
        ax.set_ylim(0, 1.02)
        ax.grid(True)
        ax.text(0.05, 0.95, _PANEL_LABELS[panel_index], transform=ax.transAxes,
                fontsize=_FONT, va="top", ha="left")
        ax.tick_params(labelbottom=True)
        if panel_index % 2 == 0:
            ax.set_ylabel("P(EVM ≤ x)")
        ax.set_xlabel("EVM (%)")
        if panel_index == 0:
            first_panel_handles = legend_handles

    axes_flat[0].legend(handles=first_panel_handles, fontsize=_SMALL,
                        handlelength=1.6, labelspacing=0.3,
                        borderpad=0.4, loc="lower right")
    for ax in axes_flat[n:]:
        ax.set_visible(False)

    plt.savefig(PLOT_PATH / "hw_evm_ecdf.svg", format="svg", bbox_inches="tight")
    plt.savefig(PLOT_PATH / "hw_evm_ecdf.png", bbox_inches="tight")
    plt.show()


# PLOT 3: Q-Q plots for best prob TCN at each DC bias
def plot_qq(run_configs: list[dict]) -> None:
    n = len(run_configs)
    ncols = min(n, 2)
    nrows = (n + ncols - 1) // ncols
    cmap  = _CMAP

    fig, axs_grid = plt.subplots(
        nrows, ncols,
        figsize=(_fw2 if ncols > 1 else _fw, nrows * _fh),
        constrained_layout=True,
    )
    fig.suptitle("Q-Q Plots of Standardized Residuals for Prob TCN Channel Models",
                 fontsize=_FONT)
    axes = np.array(axs_grid).flatten()

    for panel_index, cfg in enumerate(run_configs):
        ax = axes[panel_index]
        runs = load_channel_runs(cfg["channel_exp_dir"])
        prob_runs = [r for r in runs if r["is_prob"]]
        if not prob_runs:
            ax.text(0.5, 0.5, "No prob TCN found", transform=ax.transAxes,
                    ha="center", va="center")
            ax.set_visible(False)
            continue

        best = _best_by(prob_runs, "val_per_burst_rrmse_pct")
        adapter = load_adapter(best, cfg["channel_exp_dir"])
        model = adapter.model
        model.eval()

        X, Y, _, _ = load_dataset(cfg["dataset_path"])
        device = next(model.parameters()).device
        with torch.no_grad():
            noisy, y_mean, y_std, nu = model(X.to(device))

        residuals = ((Y.to(device) - y_mean) / y_std)[:, model.receptive_field:].detach().cpu().numpy().flatten()
        nu_flat = nu[:, model.receptive_field:].detach().cpu().numpy().flatten()

        if len(residuals) > MAX_QQ_SAMPLES:
            sample_indices = np.random.choice(len(residuals), MAX_QQ_SAMPLES, replace=False)
            residuals, nu_flat = residuals[sample_indices], nu_flat[sample_indices]

        is_gaussian = (best.get("distribution", "gaussian") == "gaussian")

        if not is_gaussian:
            (theoretical_quantiles, sample_quantiles), _ = stats.probplot(residuals, dist="norm")
            ax.plot(theoretical_quantiles, sample_quantiles, ".", color=cmap(0.9), markersize=1,
                    label="Assuming Gaussian", rasterized=True)

        if is_gaussian:
            (theoretical_quantiles, sample_quantiles), _ = stats.probplot(residuals, dist="norm")
            ax.plot(theoretical_quantiles, sample_quantiles, ".", color=cmap(0.5), markersize=1,
                    label="Gaussian Model", rasterized=True)
        else:
            normalized_cdf = stats.t.cdf(residuals, df=nu_flat)
            normal_residuals = stats.norm.ppf(normalized_cdf)
            (theoretical_quantiles, sample_quantiles), _ = stats.probplot(normal_residuals, dist="norm")
            ax.plot(theoretical_quantiles, sample_quantiles, ".", color=cmap(0.5), markersize=1,
                    label="t to Normal Transform", rasterized=True)

        ax.plot([-4.5, 4.5], [-4.5, 4.5], "--", color=cmap(0.1), label="y=x Standard Normal")
        ax.set_xlim(-4.5, 4.5)
        ax.set_ylim(-4.5, 4.5)
        dist_label = "Gaussian" if is_gaussian else "Student's-t"
        ax.set_title(f"{dist_label} TCN - {cfg['label']}", fontsize=_SMALL)
        ax.set_xlabel("Theoretical Quantiles", fontsize=_SMALL)
        ax.set_ylabel("Sample Quantiles" if panel_index % ncols == 0 else "", fontsize=_SMALL)
        ax.grid(True)
        ax.set_box_aspect(1)
        ax.legend(fontsize=_SMALL, handlelength=0.3, labelspacing=0.2)

    for ax in axes[n:]:
        ax.set_visible(False)

    plt.savefig(PLOT_PATH / "qq_prob_tcn.svg", format="svg", bbox_inches="tight")
    plt.savefig(PLOT_PATH / "qq_prob_tcn.png", bbox_inches="tight")
    plt.show()


# PLOT 4: TCN predicted response to a Gaussian wave packet (lowest DC bias)
def plot_packet_response(run_configs: list[dict]) -> None:
    cfg = min(run_configs, key=lambda c: c["dc_ma"])

    runs = load_channel_runs(cfg["channel_exp_dir"])
    prob_runs = [r for r in runs if r["is_prob"]]
    if not prob_runs:
        print(f"[plot_packet_response] No prob TCN found for {cfg['label']}; skipping.")
        return
    best = _best_by(prob_runs, "val_per_burst_rrmse_pct", "val_rrmse_pct")

    adapter = load_adapter(best, cfg["channel_exp_dir"])
    model = adapter.model
    model.eval()

    # infer baseband sampling rate from dataset
    root = zarr.open_group(_resolve(cfg["dataset_path"]), mode="r")
    attrs = dict(root.attrs)
    active_carrier_indices = np.array(attrs["active_carrier_indices"])
    f_min_hz = float(attrs.get("f_min_hz", 300e3))
    subcarrier_spacing_hz = f_min_hz / active_carrier_indices[0]   # e.g. 300e3 / 30 = 10 kHz
    if "sent_burst" in root:
        preamble_length = int(attrs.get("preamble_length", 0))
        cyclic_prefix_length = int(attrs["cyclic_prefix_length"])
        symbol_length = root["sent_burst"].shape[1] - preamble_length - cyclic_prefix_length
    else:
        symbol_length = root["sent_baseband"].shape[1]
    sample_rate_hz = symbol_length * subcarrier_spacing_hz

    # In-distribution test input: a Gaussian-shaped magnitude spectrum over the
    # active OFDM band (f_min..f_max), zero phase, with the +/-3 sigma points landing
    # on the band edges. Its IFFT is a smooth Gaussian wave packet that occupies the
    # same band the channel model was trained on, so it stays fully in-distribution.
    receptive_field = int(model.receptive_field)
    carriers = np.arange(symbol_length // 2 + 1)
    first_carrier, last_carrier = int(active_carrier_indices.min()), int(active_carrier_indices.max())
    center_carrier = 0.5 * (first_carrier + last_carrier)
    carrier_sigma = (last_carrier - first_carrier) / 6.0   # 3 sigma reaches each band edge
    in_band = (carriers >= first_carrier) & (carriers <= last_carrier)
    packet_spectrum = np.zeros(symbol_length // 2 + 1, dtype=complex)
    packet_spectrum[in_band] = np.exp(-0.5 * ((carriers[in_band] - center_carrier) / carrier_sigma) ** 2)
    packet = np.fft.irfft(packet_spectrum, n=symbol_length)
    packet = np.roll(packet, symbol_length // 2)   # centre the packet within each period
    packet *= 3.0 / np.abs(packet).max()           # scale peak to the +/-3 training range

    repetitions = max(1, int(np.ceil(receptive_field / symbol_length))) + 2   # enough periods for full RF context
    num_samples = repetitions * symbol_length
    packet_input = torch.tensor(np.tile(packet, repetitions), dtype=torch.float32).unsqueeze(0)

    device = next(model.parameters()).device
    with torch.no_grad():
        _, mean, std, nu = model(packet_input.to(device))

    # zoom to a window around one packet (past the receptive field), packet centre at t=0
    ns_per_sample = 1e9 / sample_rate_hz
    context_periods = max(1, int(np.ceil(receptive_field / symbol_length)))
    center = context_periods * symbol_length + symbol_length // 2
    envelope_width_samples = symbol_length / (2 * np.pi * carrier_sigma)
    half_window = int(np.ceil(6 * envelope_width_samples))   # +/-6 sigma around the packet
    window_start, window_end = center - half_window, center + half_window
    time_ns = (np.arange(num_samples) - center) * ns_per_sample

    predicted_mean = mean.squeeze(0).cpu().numpy()
    predicted_std = std.squeeze(0).cpu().numpy()
    predicted_nu = nu.squeeze(0).cpu().numpy()
    input_signal = packet_input.squeeze(0).numpy()
    is_gaussian = best.get("distribution", "gaussian") == "gaussian"

    cmap = _CMAP
    fig, ax1 = plt.subplots(figsize=(_fw, _fh))

    if not is_gaussian:
        lo, hi = stats.t.interval(0.997, predicted_nu[window_start:window_end],
                                  loc=predicted_mean[window_start:window_end],
                                  scale=predicted_std[window_start:window_end])
        ax1.fill_between(time_ns[window_start:window_end], lo, hi, color=cmap(0.9), alpha=0.8,
                         label="99.7% CI Student's-t")

    ax1.fill_between(time_ns[window_start:window_end],
                     predicted_mean[window_start:window_end] - 3 * predicted_std[window_start:window_end],
                     predicted_mean[window_start:window_end] + 3 * predicted_std[window_start:window_end],
                     color=cmap(0.5), alpha=0.8, label="±3 Std Dev")
    ax1.plot(time_ns[window_start:window_end], predicted_mean[window_start:window_end],
             color=cmap(0.1), label="Predicted Mean")

    ax1.set_xlabel("Time (ns)")
    ax1.set_ylabel("Predicted Received Amplitude", color=cmap(0.1))
    ax1.tick_params(axis="y", labelcolor=cmap(0.1))
    ax1.grid(False)
    ax1.set_zorder(1)
    ax1.patch.set_visible(False)

    ax2 = ax1.twinx()
    ax2.set_zorder(0)
    ax2.grid(True, alpha=0.3)
    ax2.plot(time_ns[window_start:window_end], input_signal[window_start:window_end], "--",
             color="black", alpha=0.8, label="Input Signal")
    ax2.set_ylabel("Input Signal Amplitude", color="black")
    ax2.tick_params(axis="y", labelcolor="black")

    left_handles, left_labels = ax1.get_legend_handles_labels()
    right_handles, right_labels = ax2.get_legend_handles_labels()
    ax1.legend(left_handles + right_handles, left_labels + right_labels, loc="upper right",
               fontsize=_SMALL, frameon=True, framealpha=1)
    ax1.set_title(f"TCN Response - Gaussian Wave Packet - {cfg['label']}",
                  fontsize=_FONT)

    plt.tight_layout()
    plt.savefig(PLOT_PATH / "packet_response.svg", format="svg", bbox_inches="tight")
    plt.savefig(PLOT_PATH / "packet_response.png", bbox_inches="tight")
    plt.show()


# PLOT 4b: best prob TCN's predicted noise std vs input power (one line per DC bias)

def plot_predicted_std_vs_power(run_configs: list[dict],
                                power_levels=None,
                                num_bursts: int = 256) -> None:
    """For each DC bias, take the best prob TCN channel model and plot its mean
    predicted noise std against input drive power. Test inputs are band-limited
    Gaussian bursts (random spectrum on the active OFDM carriers), matching the
    training waveform statistics, rescaled to each target power. The dataset's
    actual training power range is shaded so extrapolation is visible."""
    if power_levels is None:
        power_levels = np.linspace(0.05, 1.5, 15)

    fig, ax = plt.subplots(figsize=(_fw, _fh))
    markers = ["o", "^", "s", "D"]
    colors = [_CMAP(x) for x in np.linspace(0.1, 0.8, max(len(run_configs), 2))]

    for config_index, cfg in enumerate(run_configs):
        runs = load_channel_runs(cfg["channel_exp_dir"])
        prob_runs = [r for r in runs if r["is_prob"]]
        if not prob_runs:
            print(f"[plot_predicted_std_vs_power] No prob TCN found for {cfg['label']}; skipping.")
            continue
        best = _best_by(prob_runs, "val_per_burst_rrmse_pct", "val_rrmse_pct")

        adapter = load_adapter(best, cfg["channel_exp_dir"])
        model = adapter.model
        model.eval()
        device = next(model.parameters()).device
        receptive_field = int(model.receptive_field)

        X, _, active_carrier_indices, symbol_length = load_dataset(cfg["dataset_path"])
        training_power = X.square().mean(dim=-1)
        training_min = float(training_power.min())
        training_max = float(training_power.max())

        # band-limited Gaussian bursts: random complex spectrum on the active carriers
        rng = np.random.default_rng(0)
        half_spectrum = np.zeros((num_bursts, symbol_length // 2 + 1), dtype=complex)
        half_spectrum[:, active_carrier_indices] = (
            rng.standard_normal((num_bursts, len(active_carrier_indices)))
            + 1j * rng.standard_normal((num_bursts, len(active_carrier_indices))))
        bursts = np.fft.irfft(half_spectrum, n=symbol_length, norm="ortho")
        bursts /= np.sqrt((bursts ** 2).mean(axis=-1, keepdims=True))

        mean_std_per_power = []
        for target_power in power_levels:
            scaled = torch.tensor(bursts * np.sqrt(target_power), dtype=torch.float32, device=device)
            with torch.no_grad():
                _, _, predicted_std, _ = model(scaled)
            mean_std_per_power.append(predicted_std[:, receptive_field:].mean().item())

        ax.plot(power_levels, mean_std_per_power,
                color=colors[config_index], marker=markers[config_index % len(markers)],
                markersize=2.5, label=f"{cfg['label']} ({best['run_id']})")
        if config_index == 0:
            ax.axvspan(training_min, training_max, color="grey", alpha=0.12,
                       label="training power range")

    ax.set_xlabel("Input Power (Mean Squared Amplitude)")
    ax.set_ylabel("Mean Predicted Noise Std")
    ax.set_title("Best Prob TCN: Learned Noise Std vs Drive Power", fontsize=_FONT)
    ax.grid(True)
    ax.legend(fontsize=_SMALL, handlelength=2, labelspacing=0.4)

    plt.tight_layout()
    plt.savefig(PLOT_PATH / "predicted_std_vs_power.svg", format="svg", bbox_inches="tight")
    plt.savefig(PLOT_PATH / "predicted_std_vs_power.png", bbox_inches="tight")
    plt.show()


# PLOT 5/6: EVM% (and SNR dB) vs frequency from the fixed preamble
def evm_percent_to_snr_db(evm_percent):
    evm_fraction = np.asarray(evm_percent) / 100.0
    return -20.0 * np.log10(evm_fraction + 1e-12)


def load_fixed_preamble_and_received_preambles(dataset_path, device=DEVICE):
    root = zarr.open_group(_resolve(dataset_path), mode="r")
    attrs = dict(root.attrs)
    active_carrier_indices = np.array(attrs["active_carrier_indices"])
    preamble_length = int(attrs["preamble_length"])
    cyclic_prefix_length = int(attrs["cyclic_prefix_length"])
    subcarrier_spacing_hz = float(attrs["f_min_hz"]) / active_carrier_indices[0]
    symbol_length = root["sent_burst"].shape[1] - preamble_length - cyclic_prefix_length
    baseband_sampling_rate_hz = symbol_length * subcarrier_spacing_hz
    band_min_hz = float(attrs["f_min_hz"])
    band_max_hz = float(attrs.get("f_max_hz", active_carrier_indices[-1] * subcarrier_spacing_hz))
    sent_preamble = torch.tensor(root["sent_burst"][0, :preamble_length], dtype=torch.float32, device=device)
    received_preambles = np.array(root["received_burst"][:, :preamble_length])
    return (sent_preamble, received_preambles, preamble_length,
            baseband_sampling_rate_hz, band_min_hz, band_max_hz)


def preamble_in_band_frequencies(preamble_length, baseband_sampling_rate_hz, band_min_hz, band_max_hz):
    preamble_frequencies_hz = np.fft.rfftfreq(preamble_length, d=1.0 / baseband_sampling_rate_hz)
    in_band_mask = (preamble_frequencies_hz >= band_min_hz) & (preamble_frequencies_hz <= band_max_hz)
    return preamble_frequencies_hz[in_band_mask], in_band_mask


def measured_preamble_evm_vs_frequency(received_preambles, in_band_mask):
    received_spectrum = np.fft.rfft(received_preambles, norm="ortho", axis=-1)[:, in_band_mask]
    mean_signal_spectrum = received_spectrum.mean(axis=0)
    residual_noise_power = np.mean(np.abs(received_spectrum - mean_signal_spectrum[None, :]) ** 2, axis=0)
    return np.sqrt(residual_noise_power) / (np.abs(mean_signal_spectrum) + 1e-12) * 100.0


def predicted_preamble_evm_vs_frequency(adapter, sent_preamble, in_band_mask, is_gaussian,
                                        num_stochastic_samples=5000):
    model = adapter.model
    model.eval()
    device = next(model.parameters()).device
    preamble_batch = sent_preamble.unsqueeze(0).to(device)
    with torch.no_grad():
        _, predicted_mean, predicted_std, predicted_nu = model(preamble_batch)
        if is_gaussian:
            predicted_distribution = torch.distributions.Normal(predicted_mean, predicted_std)
        else:
            predicted_distribution = torch.distributions.StudentT(predicted_nu, predicted_mean, predicted_std)
        stochastic_samples = predicted_distribution.sample((num_stochastic_samples,)).squeeze(1).cpu().numpy()
    mean_response = predicted_mean.squeeze(0).cpu().numpy()
    mean_signal_spectrum = np.fft.rfft(mean_response, norm="ortho")[in_band_mask]
    sample_spectra = np.fft.rfft(stochastic_samples, norm="ortho", axis=-1)[:, in_band_mask]
    residual_noise_power = np.mean(np.abs(sample_spectra - mean_signal_spectrum[None, :]) ** 2, axis=0)
    return np.sqrt(residual_noise_power) / (np.abs(mean_signal_spectrum) + 1e-12) * 100.0


def best_probabilistic_channel_runs(channel_exp_dir):
    channel_runs = load_channel_runs(channel_exp_dir)
    probabilistic_runs = [run for run in channel_runs
                          if (run.get("distribution") or "none") in ("gaussian", "students_t")]
    selected_runs = []
    for model_name in dict.fromkeys(run["model"] for run in probabilistic_runs):
        architecture_runs = [run for run in probabilistic_runs if run["model"] == model_name]
        best_run = _best_by(architecture_runs, "val_nll", "val_per_burst_rrmse_pct")
        selected_runs.append((model_name, best_run.get("distribution") or "none", best_run))
    selected_runs.sort(key=lambda selected: _model_type_sort_key((selected[0], selected[1])))
    return selected_runs


def plot_preamble_evm_vs_frequency(run_configs, as_snr_db=False):
    number_of_configs = len(run_configs)
    fig, axes = plt.subplots(2, 2, figsize=(_fw2, 2 * _fh), sharex=True, sharey=True,
                             constrained_layout=True)
    axes_flat = axes.flatten()

    for panel_index, cfg in enumerate(run_configs):
        ax = axes_flat[panel_index]
        (sent_preamble, received_preambles, preamble_length, baseband_sampling_rate_hz,
         band_min_hz, band_max_hz) = load_fixed_preamble_and_received_preambles(cfg["dataset_path"])
        in_band_frequencies_hz, in_band_mask = preamble_in_band_frequencies(
            preamble_length, baseband_sampling_rate_hz, band_min_hz, band_max_hz)
        frequencies_mhz = in_band_frequencies_hz / 1e6

        measured_evm = measured_preamble_evm_vs_frequency(received_preambles, in_band_mask)
        measured_curve = evm_percent_to_snr_db(measured_evm) if as_snr_db else measured_evm
        ax.plot(frequencies_mhz, measured_curve, color="black", marker=".", markersize=2.5,
                linestyle="-", label="Measured Preamble")

        for model_name, distribution_name, channel_run in best_probabilistic_channel_runs(cfg["channel_exp_dir"]):
            adapter = load_adapter(channel_run, cfg["channel_exp_dir"])
            predicted_evm = predicted_preamble_evm_vs_frequency(
                adapter, sent_preamble, in_band_mask, distribution_name == "gaussian")
            predicted_curve = evm_percent_to_snr_db(predicted_evm) if as_snr_db else predicted_evm
            style = _get_style(model_name, distribution_name)
            ax.plot(frequencies_mhz, predicted_curve, color=style["color"], marker=style["marker"],
                    linestyle=style["linestyle"], markersize=2.5,
                    label=f"Predicted {_model_type_label(model_name, distribution_name)}")

        ax.set_title(cfg["label"], fontsize=_FONT)
        ax.grid(True)
        ax.text(0.05, 0.95, _PANEL_LABELS[panel_index], transform=ax.transAxes,
                fontsize=_FONT, va="top", ha="left")
        ax.tick_params(labelbottom=True)
        if panel_index % 2 == 0:
            ax.set_ylabel("SNR (dB)" if as_snr_db else "EVM (%)")
        ax.set_xlabel("Frequency (MHz)")

    axes_flat[0].legend(fontsize=_SMALL, handlelength=3, labelspacing=0.5)
    for ax in axes_flat[number_of_configs:]:
        ax.set_visible(False)

    file_stem = "preamble_snr_vs_frequency" if as_snr_db else "preamble_evm_vs_frequency"
    plt.savefig(PLOT_PATH / f"{file_stem}.svg", format="svg", bbox_inches="tight")
    plt.savefig(PLOT_PATH / f"{file_stem}.png", bbox_inches="tight")
    plt.show()


def plot_preamble_residual_evm_vs_frequency(run_configs):
    number_of_configs = len(run_configs)
    fig, axes = plt.subplots(2, 2, figsize=(_fw2, 2 * _fh), sharex=True, sharey=True,
                             constrained_layout=True)
    axes_flat = axes.flatten()

    for panel_index, cfg in enumerate(run_configs):
        ax = axes_flat[panel_index]
        (sent_preamble, received_preambles, preamble_length, baseband_sampling_rate_hz,
         band_min_hz, band_max_hz) = load_fixed_preamble_and_received_preambles(cfg["dataset_path"])
        in_band_frequencies_hz, in_band_mask = preamble_in_band_frequencies(
            preamble_length, baseband_sampling_rate_hz, band_min_hz, band_max_hz)
        frequencies_mhz = in_band_frequencies_hz / 1e6
        measured_evm = measured_preamble_evm_vs_frequency(received_preambles, in_band_mask)

        ax.axhline(0.0, color="black", linewidth=0.8)
        for model_name, distribution_name, channel_run in best_probabilistic_channel_runs(cfg["channel_exp_dir"]):
            adapter = load_adapter(channel_run, cfg["channel_exp_dir"])
            predicted_evm = predicted_preamble_evm_vs_frequency(
                adapter, sent_preamble, in_band_mask, distribution_name == "gaussian")
            residual_evm = measured_evm - predicted_evm
            style = _get_style(model_name, distribution_name)
            ax.plot(frequencies_mhz, residual_evm, color=style["color"], marker=style["marker"],
                    linestyle=style["linestyle"], markersize=2.5,
                    label=f"{_model_type_label(model_name, distribution_name)} (mean {np.mean(residual_evm):.1f}%)")

        ax.set_title(cfg["label"], fontsize=_FONT)
        ax.grid(True)
        ax.text(0.05, 0.95, _PANEL_LABELS[panel_index], transform=ax.transAxes,
                fontsize=_FONT, va="top", ha="left")
        ax.tick_params(labelbottom=True)
        if panel_index % 2 == 0:
            ax.set_ylabel("Residual EVM (%)")
        ax.set_xlabel("Frequency (MHz)")

    axes_flat[0].legend(fontsize=_SMALL, handlelength=3, labelspacing=0.5)
    for ax in axes_flat[number_of_configs:]:
        ax.set_visible(False)

    plt.savefig(PLOT_PATH / "preamble_residual_evm_vs_frequency.svg", format="svg", bbox_inches="tight")
    plt.savefig(PLOT_PATH / "preamble_residual_evm_vs_frequency.png", bbox_inches="tight")
    plt.show()


# PLOT 7: EVM% vs frequency from empirical E/D validation waveforms
def subcarrier_frequencies_mhz(dataset_path, active_carrier_indices):
    attrs = dict(zarr.open_group(_resolve(dataset_path), mode="r").attrs)
    subcarrier_spacing_hz = float(attrs["f_min_hz"]) / active_carrier_indices[0]
    return active_carrier_indices * subcarrier_spacing_hz / 1e6


def validation_run_evm_vs_frequency(ed_val_exp_dir, validation_run_id, active_carrier_indices):
    validation_group = zarr.open_group(_resolve(ed_val_exp_dir) / "validation.zarr", mode="r")[validation_run_id]
    sent_spectrum = np.fft.rfft(validation_group["sent_time"][:], norm="ortho", axis=-1)[:, active_carrier_indices]
    received_spectrum = np.fft.rfft(validation_group["received_time"][:], norm="ortho", axis=-1)[:, active_carrier_indices]
    residual_power = np.mean(np.abs(sent_spectrum - received_spectrum) ** 2, axis=0)
    signal_power = np.mean(np.abs(sent_spectrum) ** 2, axis=0)
    return np.sqrt(residual_power / (signal_power + 1e-12)) * 100.0


def best_validation_run_for_channel(ed_val_exp_dir, channel_run_id, active_carrier_indices):
    validation_runs = _read_jsonl(_resolve(ed_val_exp_dir) / "runs.jsonl")
    matching_validation_runs = [run for run in validation_runs if run.get("channel_run_id") == channel_run_id]
    if not matching_validation_runs:
        return None
    return min(matching_validation_runs,
               key=lambda run: float(np.nanmean(validation_run_evm_vs_frequency(
                   ed_val_exp_dir, run["run_id"], active_carrier_indices))))["run_id"]


def plot_validation_evm_vs_frequency(run_configs):
    number_of_configs = len(run_configs)
    fig, axes = plt.subplots(2, 2, figsize=(_fw2, 2 * _fh), sharex=True, sharey=True,
                             constrained_layout=True)
    axes_flat = axes.flatten()

    for panel_index, cfg in enumerate(run_configs):
        ax = axes_flat[panel_index]
        _, _, active_carrier_indices, _ = load_dataset(cfg["dataset_path"])
        frequencies_mhz = subcarrier_frequencies_mhz(cfg["dataset_path"], active_carrier_indices)

        for model_name, distribution_name, channel_run in best_probabilistic_channel_runs(cfg["channel_exp_dir"]):
            validation_run_id = best_validation_run_for_channel(
                cfg["ed_val_exp_dir"], channel_run["run_id"], active_carrier_indices)
            if validation_run_id is None:
                continue
            evm_curve = validation_run_evm_vs_frequency(cfg["ed_val_exp_dir"], validation_run_id, active_carrier_indices)
            style = _get_style(model_name, distribution_name)
            ax.plot(frequencies_mhz, evm_curve, color=style["color"], marker=style["marker"],
                    linestyle=style["linestyle"], markersize=2.5,
                    label=f"{_model_type_label(model_name, distribution_name)} E/D")

        ax.set_title(cfg["label"], fontsize=_FONT)
        ax.grid(True)
        ax.text(0.05, 0.95, _PANEL_LABELS[panel_index], transform=ax.transAxes,
                fontsize=_FONT, va="top", ha="left")
        ax.tick_params(labelbottom=True)
        if panel_index % 2 == 0:
            ax.set_ylabel("Validation EVM (%)")
        ax.set_xlabel("Frequency (MHz)")

    axes_flat[0].legend(fontsize=_SMALL, handlelength=3, labelspacing=0.5)
    for ax in axes_flat[number_of_configs:]:
        ax.set_visible(False)

    plt.savefig(PLOT_PATH / "validation_evm_vs_frequency.svg", format="svg", bbox_inches="tight")
    plt.savefig(PLOT_PATH / "validation_evm_vs_frequency.png", bbox_inches="tight")
    plt.show()


# GMP ERR: term ranking for the best GMP model at each DC bias
def print_best_gmp_err(run_configs: list[dict]) -> None:
    """Grab the best-performing GMP channel model (lowest val RRMSE) at each DC bias
    and print its error-reduction-ratio (ERR) term ranking + linear/nonlinear split."""
    for cfg in run_configs:
        gmp_runs = [r for r in load_channel_runs(cfg["channel_exp_dir"]) if r["model"] == "gmp"]
        if not gmp_runs:
            print(f"\n[{cfg['label']}] no GMP model found; skipping ERR.")
            continue
        best = _best_by(gmp_runs, "val_per_burst_rrmse_pct", "val_rrmse_pct",
                        "per_burst_rrmse_pct", "rrmse_pct")
        score = next(best[k] for k in ("val_per_burst_rrmse_pct", "val_rrmse_pct",
                                       "per_burst_rrmse_pct", "rrmse_pct") if best.get(k) is not None)
        print("\n" + "=" * 50)
        print(f"GMP ERR - {cfg['label']}  (run {best['run_id']}, val RRMSE {score:.2f}%)")
        try:
            adapter = load_adapter(best, cfg["channel_exp_dir"])
            X, Y, _, _ = load_dataset(cfg["dataset_path"])
            adapter.model.calculate_err(X, Y, plot=True)
        except Exception as e:
            print(f"  ERR calculation failed: {e}")


# MAIN
if __name__ == "__main__":
    PLOT_PATH.mkdir(parents=True, exist_ok=True)
    run_configs = sorted(RUN_CONFIGS, key=lambda c: c["dc_ma"])

    print("Plot 1 - Val RRMSE vs Sent Power ...")
    plot_val_rrmse_vs_power(run_configs)

    print("Plot 2 - Pareto: channel params vs experimental EVM% ...")
    plot_pareto_evm(run_configs)

    print("Plot 2b - Predicted vs actual actual EVM ...")
    plot_predicted_vs_actual_evm(run_configs)

    print("Plot 2c - ECDF of actual EVM per channel form ...")
    plot_hw_evm_ecdf(run_configs)

    print("Plot 3 - Q-Q plots for best prob TCN ...")
    plot_qq(run_configs)

    print("Plot 4 - Gaussian wave packet TCN response ...")
    plot_packet_response(run_configs)

    print("Plot 4b - Prob TCN learned noise std vs drive power ...")
    plot_predicted_std_vs_power(run_configs)

    print("Plot 5 - Preamble EVM% vs frequency ...")
    plot_preamble_evm_vs_frequency(run_configs)

    print("Plot 6 - Preamble SNR (dB) vs frequency ...")
    plot_preamble_evm_vs_frequency(run_configs, as_snr_db=True)

    print("Plot 6b - Preamble residual EVM% (measured - predicted) vs frequency ...")
    plot_preamble_residual_evm_vs_frequency(run_configs)

    print("Plot 7 - E/D validation EVM% vs frequency ...")
    plot_validation_evm_vs_frequency(run_configs)

    # print("\nBest GMP ERR term ranking per DC bias ...")
    # print_best_gmp_err(run_configs)

    print(f"\nDone. Figures saved to {PLOT_PATH}/")
