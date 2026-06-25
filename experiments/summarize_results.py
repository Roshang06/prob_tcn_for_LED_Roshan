"""
summarize_results.py  —  aggregate multi-DC-offset results and generate paper figures.

Edit RUN_CONFIGS and PLOT_PATH at the top, then:
    cd <repo_root>/experiments
    python summarize_results.py

Each RUN_CONFIG entry must point to:
  - channel_exp_dir  : channel-model grid-search output folder
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
from modules.utils import calculate_rrmse_pct_loss

# ─── USER CONFIG ────────────────────────────────────────────────────────────────
RUN_CONFIGS = [
    {
        "label": "121mA",
        "dc_ma": 121,
        "channel_exp_dir": "data/experiments/train_and_validate/fleet_flare_channel_models_20260623_2332",
        "ed_val_exp_dir":  "data/experiments/train_and_validate/fleet_flare_ed_validation_20260623_2349",
        "dataset_path":    "data/sweeps/dc0.122A_fmin300000_fmax1.299e+07_20260623_1752.zarr",
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
SQUARE_WAVE_HZ  = 1e6     # frequency of the square wave in plot 4
# ────────────────────────────────────────────────────────────────────────────────


# ─── STYLE (matches make_figures.ipynb) ─────────────────────────────────────────
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

_CMAP = plt.get_cmap("viridis")

# Shared style dictionary: (model_family, distribution) → plot kwargs.
# Used identically across all figures so the same model type always looks the same.
MODEL_STYLES: dict[tuple[str, str], dict] = {
    ("gmp", "none"):       {"color": _CMAP(0.05), "marker": "o", "linestyle": "-"},
    ("tcn", "none"):       {"color": _CMAP(0.38), "marker": "^", "linestyle": "--"},
    ("tcn", "gaussian"):   {"color": _CMAP(0.65), "marker": "s", "linestyle": "-."},
    ("tcn", "students_t"): {"color": _CMAP(0.90), "marker": "D", "linestyle": ":"},
}
_FALLBACK_STYLES = [
    {"color": _CMAP(0.5), "marker": "p", "linestyle": "-"},
    {"color": _CMAP(0.2), "marker": "X", "linestyle": "--"},
]
_extra_style_cache: dict[tuple, dict] = {}


def _get_style(model: str, dist: str) -> dict:
    """Return the shared plot style for a (model, distribution) key."""
    key = (model, dist or "none")
    if key in MODEL_STYLES:
        return MODEL_STYLES[key]
    if key not in _extra_style_cache:
        idx = len(_extra_style_cache) % len(_FALLBACK_STYLES)
        _extra_style_cache[key] = _FALLBACK_STYLES[idx]
    return _extra_style_cache[key]


REPO_ROOT = Path(__file__).resolve().parent.parent


def _resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


# ─── DATA HELPERS ───────────────────────────────────────────────────────────────

def _read_jsonl(path: Path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def load_channel_runs(channel_exp_dir: str) -> list[dict]:
    """Load runs.jsonl from a channel-model grid search and annotate with is_prob."""
    rows = _read_jsonl(_resolve(channel_exp_dir) / "runs.jsonl")
    for r in rows:
        dist = r.get("distribution") or "none"
        r["is_tcn"]  = r.get("model") == "tcn"
        r["is_prob"] = r["is_tcn"] and dist in ("gaussian", "students_t")
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
    ks_indices = np.array(attrs["active_carrier_indices"])
    cp_length  = int(attrs["cyclic_prefix_length"])

    if "sent_burst" in root:
        preamble_length = int(attrs.get("preamble_length", 0))
        burst = torch.tensor(root["sent_burst"][:], dtype=torch.float32, device=device)
        recv  = torch.tensor(root["received_burst"][:], dtype=torch.float32, device=device)
        offset = preamble_length + cp_length
        X, Y = burst[:, offset:], recv[:, offset:]
    else:
        X = torch.tensor(root["sent_baseband"][:], dtype=torch.float32, device=device)
        Y = torch.tensor(root["received_baseband"][:], dtype=torch.float32, device=device)

    return X, Y, ks_indices, X.shape[1]


def _predict_mean(adapter, X: torch.Tensor) -> torch.Tensor:
    """Get the mean prediction from any adapter type (handles prob/nonprob TCN and GMP)."""
    out = adapter.predict(X)
    return out[1] if isinstance(out, tuple) else out


def _best_by(runs: list[dict], key: str, fallback_key: str | None = None) -> dict | None:
    """Return the run with the minimum value for key (or fallback_key)."""
    def score(r):
        v = r.get(key)
        if v is None and fallback_key:
            v = r.get(fallback_key)
        return float("inf") if v is None else float(v)
    return min(runs, key=score, default=None)


def per_trial_evm(ed_val_exp_dir: str, ks_indices: np.ndarray) -> dict[str, np.ndarray]:
    """
    Compute per-trial EVM% from the time-domain signals stored in validation.zarr.
    Returns {run_id: array of shape [num_trials]}.

    sent_time and received_time are symbol-only (no CP, no preamble), so we
    rfft directly and index by ks_indices.
    """
    root = zarr.open_group(_resolve(ed_val_exp_dir) / "validation.zarr", mode="r")
    result = {}
    for key in root.keys():
        g = root[key]
        sent = g["sent_time"][:]     # (num_trials, N_sym)
        recv = g["received_time"][:]
        evms = []
        for i in range(sent.shape[0]):
            sf = np.fft.rfft(sent[i], norm="ortho")[ks_indices]
            rf = np.fft.rfft(recv[i],  norm="ortho")[ks_indices]
            sig_pwr = np.mean(np.abs(sf) ** 2)
            mse     = np.mean(np.abs(sf - rf) ** 2)
            evms.append(float(np.sqrt(mse / (sig_pwr + 1e-12)) * 100))
        result[key] = np.array(evms)
    return result


def _binned_rrmse(
    X: torch.Tensor,
    Y: torch.Tensor,
    adapter,
    n_bins: int = N_POWER_BINS,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-bin RRMSE (%) binned by sent-power (mean squared amplitude)."""
    pwr = X.square().mean(dim=-1)
    bins = torch.linspace(pwr.min(), pwr.max(), n_bins + 1)
    bin_ids = torch.bucketize(pwr, bins)
    centers = (0.5 * (bins[:-1] + bins[1:])).cpu().numpy()

    Y_hat = _predict_mean(adapter, X)
    vals = []
    for i in range(n_bins):
        mask = bin_ids == i
        if mask.any():
            vals.append(float(calculate_rrmse_pct_loss(Y[mask], Y_hat[mask])))
        else:
            vals.append(float("nan"))
    return centers, np.array(vals)


# ─── PLOT 1 & 2 shared helpers ──────────────────────────────────────────────────

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


# ─── PLOT 1: Val RRMSE vs Sent Power – 2×2 subplot grid, one panel per DC bias ─
# Model type → style from MODEL_STYLES (color/marker/linestyle), consistent across panels.
# Shared x/y axes so panels are directly comparable.

_PANEL_LABELS = ["(a)", "(b)", "(c)", "(d)"]

def plot_val_rrmse_vs_power(run_configs: list[dict]) -> None:
    model_types = _discover_model_types(run_configs)
    n = len(run_configs)

    fig, axes = plt.subplots(2, 2, figsize=(_fw2, 2 * _fh),
                             sharex=True, sharey=True,
                             constrained_layout=True)
    axes_flat = axes.flatten()

    for idx, cfg in enumerate(run_configs):
        ax = axes_flat[idx]
        X, Y, _, _ = load_dataset(cfg["dataset_path"])
        runs = load_channel_runs(cfg["channel_exp_dir"])

        for model, dist in model_types:
            style = _get_style(model, dist)
            candidates = [r for r in runs
                          if r["model"] == model and (r.get("distribution") or "none") == dist]
            if not candidates:
                continue
            best    = _best_by(candidates, "val_rrmse_pct")
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
        ax.text(0.05, 0.95, _PANEL_LABELS[idx], transform=ax.transAxes,
                fontsize=_FONT, va="top", ha="left")
        if idx % 2 == 0:
            ax.set_ylabel("Val RRMSE (%)")
        if idx >= 2:
            ax.set_xlabel("Sent Power (Mean Squared Amplitude)")

    axes_flat[0].legend(fontsize=_FONT, handlelength=3, labelspacing=0.5)

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    plt.savefig(PLOT_PATH / "val_rrmse_vs_power.svg", format="svg", bbox_inches="tight")
    plt.savefig(PLOT_PATH / "val_rrmse_vs_power.png", bbox_inches="tight")
    plt.show()


# ─── PLOT 2: Pareto – channel model params vs experimental EVM% ────────────────
# 2×2 subplot grid, one panel per DC bias (mirrors plot 1).
# One trace per (model, distribution) type using MODEL_STYLES.
# Lines connect points ordered by param count; error bars = 2× SE (≈ 95% CI).

def plot_pareto_evm(run_configs: list[dict]) -> None:
    n = len(run_configs)

    fig, axes = plt.subplots(2, 2, figsize=(_fw2, 2 * _fh),
                             sharex=True, sharey=True,
                             constrained_layout=True)
    axes_flat = axes.flatten()

    for idx, cfg in enumerate(run_configs):
        ax = axes_flat[idx]
        _, _, ks, _ = load_dataset(cfg["dataset_path"])
        ch_map   = {r["run_id"]: r for r in load_channel_runs(cfg["channel_exp_dir"])}
        val_runs = _read_jsonl(_resolve(cfg["ed_val_exp_dir"]) / "runs.jsonl")
        t_evms   = per_trial_evm(cfg["ed_val_exp_dir"], ks)

        best_ed: dict[str, dict] = {}
        for vr in val_runs:
            ch_id = vr.get("channel_run_id")
            if ch_id is None:
                continue
            trials   = t_evms.get(vr["run_id"], np.array([vr.get("evm_pct", float("nan"))]))
            mean_evm = float(np.nanmean(trials))
            if ch_id not in best_ed or mean_evm < best_ed[ch_id]["mean_evm"]:
                best_ed[ch_id] = {"mean_evm": mean_evm, "trials": trials,
                                  "ch_run": ch_map.get(ch_id)}

        trace_data: dict[tuple, list] = {}
        for ch_id, entry in best_ed.items():
            ch_run = entry["ch_run"]
            if ch_run is None:
                continue
            trials   = entry["trials"]
            mean_evm = float(np.nanmean(trials))
            se       = (float(np.nanstd(trials, ddof=1)) / np.sqrt(len(trials))
                        if len(trials) > 1 else 0.0)
            key = (ch_run.get("model", "unknown"), ch_run.get("distribution") or "none")
            trace_data.setdefault(key, []).append({
                "num_params": ch_run["num_params"],
                "mean_evm":  mean_evm,
                "se":        se,
            })

        for key in sorted(trace_data.keys(), key=_model_type_sort_key):
            model, dist = key
            style  = _get_style(model, dist)
            points = sorted(trace_data[key], key=lambda p: p["num_params"])
            xs  = np.array([p["num_params"] for p in points], dtype=float)
            ys  = np.array([p["mean_evm"]   for p in points])
            ses = np.array([p["se"]         for p in points])
            ax.plot(xs, ys, color=style["color"], linestyle=style["linestyle"],
                    linewidth=1.0, zorder=1)
            ax.errorbar(xs, ys, yerr=2 * ses,
                        color=style["color"], marker=style["marker"],
                        markersize=3, capsize=2, capthick=0.5,
                        elinewidth=0.5, linestyle="none", zorder=2,
                        label=_model_type_label(model, dist))

        ax.set_title(cfg["label"], fontsize=_FONT)
        ax.grid(True)
        ax.text(0.05, 0.95, _PANEL_LABELS[idx], transform=ax.transAxes,
                fontsize=_FONT, va="top", ha="left")
        if idx % 2 == 0:
            ax.set_ylabel("Experimental EVM (%)")
        if idx >= 2:
            ax.set_xlabel("Channel Model Parameter Count")

    axes_flat[0].legend(fontsize=_SMALL, handlelength=2, markerscale=0.8,
                        labelspacing=0.3, borderpad=0.4)

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    plt.savefig(PLOT_PATH / "pareto_evm.svg", format="svg", bbox_inches="tight")
    plt.savefig(PLOT_PATH / "pareto_evm.png", bbox_inches="tight")
    plt.show()


# ─── PLOT 3: Q-Q plots for best prob TCN at each DC bias ───────────────────────

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
    fig.suptitle("Q-Q Plots of Standardized Residuals – Prob TCN Channel Models",
                 fontsize=_FONT)
    axes = np.array(axs_grid).flatten()

    for idx, cfg in enumerate(run_configs):
        ax = axes[idx]
        runs = load_channel_runs(cfg["channel_exp_dir"])
        prob_runs = [r for r in runs if r["is_prob"]]
        if not prob_runs:
            ax.text(0.5, 0.5, "No prob TCN found", transform=ax.transAxes,
                    ha="center", va="center")
            ax.set_visible(False)
            continue

        best = _best_by(prob_runs, "val_rrmse_pct")
        adapter = load_adapter(best, cfg["channel_exp_dir"])
        model = adapter.model
        model.eval()

        X, Y, _, _ = load_dataset(cfg["dataset_path"])
        dev = next(model.parameters()).device
        with torch.no_grad():
            noisy, y_mean, y_std, nu = model(X.to(dev))

        r = ((Y.to(dev) - y_mean) / y_std)[:, model.receptive_field:].detach().cpu().numpy().flatten()
        nu_flat = nu[:, model.receptive_field:].detach().cpu().numpy().flatten()

        if len(r) > MAX_QQ_SAMPLES:
            sel = np.random.choice(len(r), MAX_QQ_SAMPLES, replace=False)
            r, nu_flat = r[sel], nu_flat[sel]

        is_gaussian = (best.get("distribution", "gaussian") == "gaussian")

        if not is_gaussian:
            (osm, osr), _ = stats.probplot(r, dist="norm")
            ax.plot(osm, osr, ".", color=cmap(0.9), markersize=1,
                    label="Assuming Gaussian", rasterized=True)

        if is_gaussian:
            (osm, osr), _ = stats.probplot(r, dist="norm")
            ax.plot(osm, osr, ".", color=cmap(0.5), markersize=1,
                    label="Gaussian Model", rasterized=True)
        else:
            cdf_vals  = stats.t.cdf(r, df=nu_flat)
            r_norm    = stats.norm.ppf(cdf_vals)
            (osm, osr), _ = stats.probplot(r_norm, dist="norm")
            ax.plot(osm, osr, ".", color=cmap(0.5), markersize=1,
                    label="t→Normal Transform", rasterized=True)

        ax.plot([-4.5, 4.5], [-4.5, 4.5], "--", color=cmap(0.1), label="y=x Standard Normal")
        ax.set_xlim(-4.5, 4.5)
        ax.set_ylim(-4.5, 4.5)
        dist_label = "Gaussian" if is_gaussian else "Student's-t"
        ax.set_title(f"{dist_label} TCN – {cfg['label']}", fontsize=_SMALL)
        ax.set_xlabel("Theoretical Quantiles", fontsize=_SMALL)
        ax.set_ylabel("Sample Quantiles" if idx % ncols == 0 else "", fontsize=_SMALL)
        ax.grid(True)
        ax.set_box_aspect(1)
        ax.legend(fontsize=_SMALL, handlelength=0.3, labelspacing=0.2)

    for ax in axes[n:]:
        ax.set_visible(False)

    plt.savefig(PLOT_PATH / "qq_prob_tcn.svg", format="svg", bbox_inches="tight")
    plt.savefig(PLOT_PATH / "qq_prob_tcn.png", bbox_inches="tight")
    plt.show()


# ─── PLOT 4: TCN predicted response to square wave (lowest DC bias) ─────────────

def plot_square_wave(run_configs: list[dict]) -> None:
    cfg = min(run_configs, key=lambda c: c["dc_ma"])

    runs = load_channel_runs(cfg["channel_exp_dir"])
    prob_runs = [r for r in runs if r["is_prob"]]
    if not prob_runs:
        print(f"[plot_square_wave] No prob TCN found for {cfg['label']}; skipping.")
        return
    best = _best_by(prob_runs, "val_rrmse_pct")

    adapter = load_adapter(best, cfg["channel_exp_dir"])
    model = adapter.model
    model.eval()

    # infer baseband sampling rate from dataset
    root  = zarr.open_group(_resolve(cfg["dataset_path"]), mode="r")
    attrs = dict(root.attrs)
    ks    = np.array(attrs["active_carrier_indices"])
    f_min = float(attrs.get("f_min_hz", 300e3))
    spacing_hz = f_min / ks[0]   # e.g. 300e3 / 30 = 10 kHz
    if "sent_burst" in root:
        preamble = int(attrs.get("preamble_length", 0))
        cp       = int(attrs["cyclic_prefix_length"])
        sym_len  = root["sent_burst"].shape[1] - preamble - cp
    else:
        sym_len = root["sent_baseband"].shape[1]
    fs = sym_len * spacing_hz   # baseband sampling rate in Hz

    n_samples = 8000
    t = torch.arange(n_samples, dtype=torch.float32)
    x_sq = torch.sign(torch.sin(2 * torch.pi * SQUARE_WAVE_HZ / fs * t))
    x_sq = ((x_sq + 1.0) / 2.0) * 6.0 - 3.0   # {-1,1} → {-3, +3}
    x_sq = x_sq.unsqueeze(0)   # [1, n_samples]

    dev = next(model.parameters()).device
    with torch.no_grad():
        _, mean, std, nu = model(x_sq.to(dev))

    ns_per_sample = 1e9 / fs
    t_ns = t.numpy() * ns_per_sample
    rf   = model.receptive_field
    n_window = max(int(3000 / ns_per_sample), 10)
    sl, sr = rf, min(rf + n_window, n_samples)

    mean_np = mean.squeeze(0).cpu().numpy()
    std_np  = std.squeeze(0).cpu().numpy()
    nu_np   = nu.squeeze(0).cpu().numpy()
    x_np    = x_sq.squeeze(0).numpy()
    is_gaussian = best.get("distribution", "gaussian") == "gaussian"

    cmap = _CMAP
    fig, ax1 = plt.subplots(figsize=(_fw, _fh))

    if not is_gaussian:
        lo, hi = stats.t.interval(0.997, nu_np[sl:sr], loc=mean_np[sl:sr], scale=std_np[sl:sr])
        ax1.fill_between(t_ns[sl:sr], lo, hi, color=cmap(0.9), alpha=0.8,
                         label="99.7% CI Student's-t")

    ax1.fill_between(t_ns[sl:sr],
                     mean_np[sl:sr] - 3 * std_np[sl:sr],
                     mean_np[sl:sr] + 3 * std_np[sl:sr],
                     color=cmap(0.5), alpha=0.8, label="±3 Std Dev")
    ax1.plot(t_ns[sl:sr], mean_np[sl:sr], color=cmap(0.1), label="Predicted Mean")

    ax1.set_xlabel("Time (ns)")
    ax1.set_ylabel("Predicted Received Amplitude", color=cmap(0.1))
    ax1.tick_params(axis="y", labelcolor=cmap(0.1))
    ax1.grid(False)
    ax1.set_zorder(1)
    ax1.patch.set_visible(False)

    ax2 = ax1.twinx()
    ax2.set_zorder(0)
    ax2.grid(True, alpha=0.3)
    ax2.plot(t_ns[sl:sr], x_np[sl:sr], "--", color="black", alpha=0.8, label="Input Signal")
    ax2.set_ylabel("Normalized Input Signal Amplitude", color="black")
    ax2.tick_params(axis="y", labelcolor="black")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=_SMALL,
               frameon=True, framealpha=1)
    freq_khz = int(SQUARE_WAVE_HZ / 1e3)
    ax1.set_title(f"TCN Response – {freq_khz} kHz Square Wave – {cfg['label']}",
                  fontsize=_FONT)

    plt.tight_layout()
    plt.savefig(PLOT_PATH / "square_wave_response.svg", format="svg", bbox_inches="tight")
    plt.savefig(PLOT_PATH / "square_wave_response.png", bbox_inches="tight")
    plt.show()


# ─── MAIN ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    PLOT_PATH.mkdir(parents=True, exist_ok=True)
    run_configs = sorted(RUN_CONFIGS, key=lambda c: c["dc_ma"])

    print("Plot 1 – Val RRMSE vs Sent Power ...")
    plot_val_rrmse_vs_power(run_configs)

    print("Plot 2 – Pareto: channel params vs experimental EVM% ...")
    plot_pareto_evm(run_configs)

    print("Plot 3 – Q-Q plots for best prob TCN ...")
    plot_qq(run_configs)

    print("Plot 4 – Square wave TCN response ...")
    plot_square_wave(run_configs)

    print(f"\nDone. Figures saved to {PLOT_PATH}/")
