'''
Controlled per-category CDF experiment, run after a finished train_and_validate.

take the single best channel model of each form, train k
encoder/decoders that differ only by seed on each, and validate them all on the live
channel. Every form then contributes exactly k runs from an equally-good channel, so
the per-form EVM ECDFs are a fair comparison.
'''
import json
import random
import yaml
from datetime import datetime
from pathlib import Path

import numpy as np
import zarr
from matplotlib import colormaps
from matplotlib.colors import ListedColormap, LogNorm, Normalize
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from pyflux.core.experiment import ExperimentalContext
from modules.experimental_blocks import *
from modules.constellation_diagram import get_constellation
from modules.grid_search import (
    EncoderDecoderGridSearch, EncoderDecoderValidation,
    select_channel_models, select_encoder_decoders,
)
from modules.grid_search.base import generate_run_name
from modules.utils import OFDMConfig

HERE = Path(__file__).resolve().parent

CONFIG_FILE = HERE / "train_and_validate.yml"
EXP_DIR = HERE.parent / "data/experiments/train_and_validate"
LOG_DIR = HERE.parent / "data/logs"

# The annealing sweep, not the model form, is the comparison here, so colour encodes the
# annealing value (a sequential ramp, since annealing is ordered) rather than following the
# global per-form palette. Marker still distinguishes the form; nonprob is a black baseline.
FORM_MARKER = {
    "nonprob TCN": "o",
    "prob TCN": "o",
    "nonprob LRU": "s",
    "prob LRU": "s",
    "gmp": "D",
}
NONPROB_COLOR = "#000000"
ANNEAL_CMAP = "plasma"
CLIP_PENALTY_CMAP = "plasma"

# marker shape encodes rho in the clip figure, assigned in ascending rho order, so traces
# sharing a rho stay recognizable even where the colour ramp separates them
CLIP_RHO_MARKERS = ("o", "s", "^", "D", "v", "P")

BASELINE_CLIP_WEIGHT = 0.0
BASELINE_ANNEAL = 1.0


def _read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _fixed_ed_grid_config():
    '''ENCODER_DECODER config with every sweep flattened to its first value, so the only
    variation left is whatever axis the caller adds back.'''
    with open(CONFIG_FILE, encoding="utf-8") as f:
        full = yaml.safe_load(f)
    grid_config = {k.lower(): v for k, v in full["ENCODER_DECODER"].items()}

    def first(value):
        return value[0] if isinstance(value, list) else value

    grid_config["params"] = {key: first(value) for key, value in grid_config["params"].items()}
    return grid_config


def build_seed_swept_grid(seeds, anneal_starts):
    '''Fixed training architecture plus a seed axis and a noise_anneal_start axis, so seed
    and annealing are the only sources of variation. The clip penalty is forced off rather
    than inherited from the config, so this sweep measures annealing alone and does not shift
    underneath the result when the ENCODER_DECODER penalty settings are edited.'''
    grid_config = _fixed_ed_grid_config()
    grid_config["params"]["seed"] = list(seeds)
    grid_config["params"]["noise_anneal_start"] = list(anneal_starts)
    grid_config["params"]["clip_penalty_weight"] = BASELINE_CLIP_WEIGHT
    return grid_config


def build_clip_swept_grid(seeds, anneal_start, rhos, clip_weights):
    '''Fixed training architecture and a single annealing value (the empirical best from the
    annealing sweep), swept over the cartesian product of clip-penalty rho and weight plus a
    seed axis. A weight-0 arm rides along as the no-penalty control; its rho duplicates are
    dropped in keep_clip_sweep_points, since rho does nothing once the weight is 0.'''
    grid_config = _fixed_ed_grid_config()
    params = grid_config["params"]
    params["seed"] = list(seeds)
    params["noise_anneal_start"] = float(anneal_start)
    params["clip_penalty_rho"] = list(rhos)
    params["clip_penalty_weight"] = [BASELINE_CLIP_WEIGHT] + list(clip_weights)
    return grid_config


def keep_clip_sweep_points(points, prob_channel_ids, baseline_rho):
    '''The clip penalty is a statement about the probabilistic channel's gain curve, so the
    sweep runs on prob channels only. The weight-0 control is kept at a single rho, since the
    rho variants would train identical E/Ds.'''
    kept = []
    for point in points:
        if point["channel_run_id"] not in prob_channel_ids:
            continue

        params = point["params"]
        is_baseline = float(params["clip_penalty_weight"]) == BASELINE_CLIP_WEIGHT
        if is_baseline and float(params["clip_penalty_rho"]) != baseline_rho:
            continue

        kept.append(point)
    return kept


def plot_per_form_ecdf(validation_exp_dir, ed_exp_dir, out_path):
    '''Per-form ECDF of validation EVM. Prob channels get one trace per noise_anneal_start
    value, coloured by a sequential ramp over the annealing value (annealing is inert on
    nonprob channels, which stay a single black baseline trace). Marker encodes the model
    form. Colour intentionally tracks annealing, not form, since the annealing sweep is the
    comparison.'''
    anneal_by_ed = {row["run_id"]: float(row.get("noise_anneal_start"))
                    for row in _read_jsonl(Path(ed_exp_dir) / "runs.jsonl")}
    val_rows = _read_jsonl(Path(validation_exp_dir) / "runs.jsonl")

    # group EVMs by (form, anneal); nonprob folds every anneal into one trace (inert there)
    groups = {}
    for row in val_rows:
        form = row["channel_form"]
        is_prob = form.startswith("prob")
        anneal = anneal_by_ed.get(row["model"])
        key = (form, anneal if is_prob else None)
        groups.setdefault(key, []).append(float(row["evm_pct"]))

    anneal_values = sorted({anneal for (_, anneal) in groups if anneal is not None})
    cmap = colormaps[ANNEAL_CMAP]
    ramp_positions = np.linspace(0.15, 0.85, len(anneal_values)) if anneal_values else []
    color_for_anneal = dict(zip(anneal_values, (cmap(pos) for pos in ramp_positions)))

    fig = Figure(figsize=(7.5, 5))
    ax = fig.subplots()
    legend_handles = []
    # prob traces first (ascending annealing), nonprob baseline last
    for (form, anneal) in sorted(groups, key=lambda k: (k[1] is None, k[1] if k[1] is not None else 0.0, k[0])):
        marker = FORM_MARKER.get(form, "o")
        if anneal is None:
            color, linestyle = NONPROB_COLOR, "--"
            label = f"{form} (n={len(groups[(form, anneal)])})"
        else:
            color, linestyle = color_for_anneal[anneal], "-"
            label = f"{form} anneal={anneal:g} (n={len(groups[(form, anneal)])})"
        evms = np.sort(groups[(form, anneal)])
        quantiles = np.arange(1, len(evms) + 1) / len(evms)
        ax.step(np.concatenate([[evms[0]], evms]),
                np.concatenate([[0.0], quantiles]),
                where="post", color=color, lw=1.6, linestyle=linestyle)
        ax.plot(evms, quantiles, linestyle="none", marker=marker, color=color, ms=5)

        legend_handles.append(Line2D([], [], color=color, marker=marker, ms=5,
                                     lw=1.6, linestyle=linestyle, label=label))

    ax.set_xlabel("Validation EVM (%)")
    ax.set_ylabel("P(EVM <= x)")
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(handles=legend_handles, fontsize=8, frameon=False, loc="lower right")
    ax.set_title("Per-Form Validation EVM ECDF (annealing sweep on prob channels)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")


def _clip_settings(row):
    '''(rho, weight) of a validation row. An E/D trained without the penalty in its params
    records both as null, which reads here as the inert weight-0 setting.'''
    return (float(row.get("clip_penalty_rho") or 0.0),
            float(row.get("clip_penalty_weight") or 0.0))


def _clip_penalty_colormap(products):
    '''Colour scale over the rho * weight product, which is the natural ordering of penalty
    strength: rho sets how much of the gain curve is penalized and the weight sets how hard.
    '''
    cmap = ListedColormap(colormaps[CLIP_PENALTY_CMAP](np.linspace(0.12, 0.88, 256)))
    unique_products = sorted(set(products))
    if len(unique_products) < 2:
        return cmap, Normalize(vmin=0.0, vmax=1.0)
    return cmap, LogNorm(vmin=min(unique_products), vmax=max(unique_products))


def plot_clip_penalty_ecdf(validation_exp_dir, out_path):
    '''ECDF of validation EVM over the clip-penalty sweep, one trace per (rho, weight) pair.
    Colour tracks the rho * weight product and marker shape tracks rho alone, so traces
    sharing a rho stay visually grouped where the colour ramp puts them far apart. The
    weight-0 arm is the no-penalty control, a black dashed baseline as in the annealing
    figure, marked with an x since it has no rho.'''
    val_rows = _read_jsonl(Path(validation_exp_dir) / "runs.jsonl")

    groups = {}
    for row in val_rows:
        rho, weight = _clip_settings(row)
        key = (rho, weight) if weight > 0.0 else None
        groups.setdefault(key, []).append(float(row["evm_pct"]))

    penalty_keys = [key for key in groups if key is not None]
    cmap, norm = _clip_penalty_colormap([rho * weight for rho, weight in penalty_keys])
    rho_values = sorted({rho for rho, _ in penalty_keys})
    marker_for_rho = {rho: CLIP_RHO_MARKERS[index % len(CLIP_RHO_MARKERS)]
                      for index, rho in enumerate(rho_values)}

    fig = Figure(figsize=(7.5, 5))
    ax = fig.subplots()
    legend_handles = []
    # ascending penalty strength, no-penalty control last
    for key in sorted(groups, key=lambda k: (k is None, k[0] * k[1] if k is not None else 0.0)):
        evms = np.sort(groups[key])
        quantiles = np.arange(1, len(evms) + 1) / len(evms)

        if key is None:
            color, linestyle, marker = NONPROB_COLOR, "--", "x"
            label = f"no clip penalty (n={len(evms)})"
        else:
            rho, weight = key
            color, linestyle, marker = cmap(norm(rho * weight)), "-", marker_for_rho[rho]
            label = f"rho={rho:g}, weight={weight:g} (n={len(evms)})"

        ax.step(np.concatenate([[evms[0]], evms]),
                np.concatenate([[0.0], quantiles]),
                where="post", color=color, lw=1.6, linestyle=linestyle)
        ax.plot(evms, quantiles, linestyle="none", marker=marker, color=color, ms=6)

        legend_handles.append(Line2D([], [], color=color, marker=marker, ms=6,
                                     lw=1.6, linestyle=linestyle, label=label))

    ax.set_xlabel("Validation EVM (%)")
    ax.set_ylabel("P(EVM <= x)")
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(handles=legend_handles, fontsize=8, frameon=False, loc="lower right")
    ax.set_title("Validation EVM ECDF for the Clip Penalty Sweep")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")


def validated_clip_settings(validation_exp_dir):
    '''Every (rho, weight) pair with an active penalty that actually validated, ordered by
    penalty strength so the per-setting drive figures read as a sequence.'''
    settings = {_clip_settings(row) for row in _read_jsonl(Path(validation_exp_dir) / "runs.jsonl")}
    active = [(rho, weight) for rho, weight in settings if weight > 0.0]
    return sorted(active, key=lambda setting: (setting[0] * setting[1], setting))


def collect_encoder_drives(validation_exp_dir, run_ids):
    '''Every encoder drive sample measured over the validation trials of the given runs,
    flattened into one amplitude population.'''
    val_group = zarr.open_group(Path(validation_exp_dir) / "validation.zarr", mode="r")
    drives = [np.asarray(val_group[run_id]["encoder_drive"]).reshape(-1)
              for run_id in run_ids if run_id in val_group]
    if not drives:
        raise ValueError(f"no stored encoder drives for {run_ids} in {validation_exp_dir}")
    return np.concatenate(drives)


def _penalty_active_intervals(centers, normalized_gain, rho):
    '''Contiguous drive-amplitude ranges where the normalized channel gain sits below rho, so
    the one-sided penalty is pushing on the encoder. The gain is a numerical derivative of a
    binned transfer estimate, so it dips under rho over several disjoint ranges even where the
    transfer itself is monotonic; each range is returned rather than just its onset.'''
    bin_width = float(np.mean(np.diff(centers)))
    below = normalized_gain < rho
    intervals = []
    run_start = None

    for index, is_below in enumerate(below):
        if is_below and run_start is None:
            run_start = index
        elif not is_below and run_start is not None:
            intervals.append((float(centers[run_start]) - bin_width / 2,
                              float(centers[index - 1]) + bin_width / 2))
            run_start = None

    if run_start is not None:
        intervals.append((float(centers[run_start]) - bin_width / 2,
                          float(centers[-1]) + bin_width / 2))
    return intervals


def plot_clip_drive_distribution(validation_exp_dir, gain_curve_path, rho, weight,
                                 clip_threshold, out_path):
    '''What the clip penalty actually does to the encoder, in one figure: the measured drive
    amplitude distribution with and without the penalty, over the channel transfer curve that
    the penalty is derived from and the penalty curve itself. Both curves are normalized to
    unit peak, since only their shape against the drive axis matters here.'''
    val_rows = _read_jsonl(Path(validation_exp_dir) / "runs.jsonl")

    clip_run_ids = [row["run_id"] for row in val_rows if _clip_settings(row) == (rho, weight)]
    no_clip_run_ids = [row["run_id"] for row in val_rows
                       if _clip_settings(row)[1] == BASELINE_CLIP_WEIGHT]
    if not clip_run_ids:
        raise ValueError(f"no validated runs at rho={rho}, weight={weight} in {validation_exp_dir}")

    clip_drives = collect_encoder_drives(validation_exp_dir, clip_run_ids)
    no_clip_drives = collect_encoder_drives(validation_exp_dir, no_clip_run_ids)

    curve = np.load(gain_curve_path)
    centers = curve["centers"]
    normalized_transfer = curve["transfer"] / (np.max(np.abs(curve["transfer"])) + 1e-12)
    normalized_gain = curve["gain"] / (float(curve["reference_gain"]) + 1e-12)
    penalty = np.maximum(rho - normalized_gain, 0.0) ** 2
    normalized_penalty = penalty / (np.max(penalty) + 1e-12)

    cmap, norm = _clip_penalty_colormap([rho * weight])
    clip_color = cmap(norm(rho * weight))

    fig = Figure(figsize=(8, 5))
    drive_ax = fig.subplots()
    curve_ax = drive_ax.twinx()

    bins = np.linspace(-clip_threshold, clip_threshold, 121)
    drive_ax.hist(clip_drives, bins=bins, density=True, color=clip_color, alpha=0.55,
                  label=f"drive, clip penalty (rho={rho:g}, weight={weight:g})")
    drive_ax.hist(no_clip_drives, bins=bins, density=True, color=NONPROB_COLOR, alpha=0.35,
                  label="drive, no clip penalty")

    curve_ax.plot(centers, normalized_transfer, color="#0072B2", lw=1.8,
                  label="channel transfer (normalized)")
    curve_ax.plot(centers, normalized_penalty, color="#D55E00", lw=1.8, ls="-.",
                  label="clip penalty loss (normalized)")

    for index, (low, high) in enumerate(_penalty_active_intervals(centers, normalized_gain, rho)):
        drive_ax.axvspan(low, high, color="#020201", alpha=0.12, lw=0, zorder=0,
                         label=f"penalty active (gain < rho = {rho:g})" if index == 0 else None)

    drive_ax.set_xlim(-clip_threshold, clip_threshold)
    drive_ax.set_xlabel("Encoder drive amplitude (V)")
    drive_ax.set_ylabel("Drive amplitude density")
    curve_ax.set_ylabel("Normalized transfer / penalty")
    drive_ax.grid(True, alpha=0.3)

    drive_handles, drive_labels = drive_ax.get_legend_handles_labels()
    curve_handles, curve_labels = curve_ax.get_legend_handles_labels()
    drive_ax.legend(drive_handles + curve_handles, drive_labels + curve_labels,
                    fontsize=8, frameon=False, ncol=2,
                    loc="upper center", bbox_to_anchor=(0.5, -0.13))
    drive_ax.set_title("Encoder drive distribution against the channel transfer and clip penalty\n"
                       f"(rho={rho:g}, weight={weight:g})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")


if __name__ == "__main__":
    RUN_NAME = generate_run_name()

    with ExperimentalContext(CONFIG_FILE=CONFIG_FILE, create_log_file=True, run_name=RUN_NAME,
                             log_dir=LOG_DIR) as Exp:
        test_cfg = Exp.config.CONTROLLED_CDF
        if not getattr(test_cfg, "ENABLED", False):
            Exp.log("CONTROLLED_CDF.ENABLED is false, nothing to do")
            raise SystemExit(0)

        cfg = Exp.config.DATA_COLLECTION
        device = Exp.config.RUNTIME.DEVICE
        seed = int(Exp.config.RUNTIME.SEED)

        seeds = list(test_cfg.SEEDS)
        anneal_starts = list(test_cfg["ANNEAL_STARTS"])
        num_trials = int(test_cfg.N)
        noise_floor_points = int(getattr(test_cfg, "NOISE_FLOOR_POINTS", 0) or 0)

        run_annealing_sweep = bool(getattr(test_cfg, "RUN_ANNEALING_SWEEP", True))
        run_clip_penalty_sweep = bool(getattr(test_cfg, "RUN_CLIP_PENALTY_SWEEP", False))
        if not run_annealing_sweep and not run_clip_penalty_sweep:
            Exp.log("both CONTROLLED_CDF sweeps are disabled, nothing to do")
            raise SystemExit(0)

        channel_exp_dir = getattr(test_cfg, "CHANNEL_EXP_DIR", None)
        if not channel_exp_dir:
            raise ValueError("CONTROLLED_CDF needs CHANNEL_EXP_DIR set to a finished "
                             "channel_models_ directory")
        channel_exp_dir = Path(channel_exp_dir)
        dataset_path = cfg.DATASET_PATH
        exclude_models = [m.lower() for m in (getattr(test_cfg, "EXCLUDE_MODELS", None) or [])]

        # one best channel model per form: (gmp), (prob/nonprob TCN), (prob/nonprob LRU)
        best_channels = select_channel_models(channel_exp_dir, mode="best")
        if exclude_models:
            best_channels = [cm for cm in best_channels if cm["model"].lower() not in exclude_models]
            Exp.log(f"excluding channel models {exclude_models}")
        Exp.log(f"best channel per form ({len(best_channels)} forms), "
                f"{len(seeds)} seeds each -> {len(best_channels) * len(seeds)} E/Ds")
        for cm in best_channels:
            print(f"  channel {cm['model']:4s}  dist={cm.get('distribution', 'none'):12s}  "
                  f"run={cm['run_id']}")

        awg = Exp.Agilent_33250A
        osc = Exp.Tektronix_TDS2000
        pwr_supply = Exp.HP_EE3631A

        awg.set_output_load("INF")
        osc.display_channel(ch=1)
        osc.display_channel(ch=3)
        osc.set_record_length(200_000)
        osc.set_trigger(ch=1, voltage_level=0)
        osc.set_probe_gain(ch=1, gain=1)
        osc.set_probe_gain(ch=3, gain=1)
        osc.configure_channel(ch=3, scale=cfg.OSC_SCALE, offset=0)
        osc.configure_channel(ch=1, scale=1, offset=0)
        osc.set_coupling(ch=1, coupling="AC")
        osc.set_coupling(ch=3, coupling="AC")

        dc_offset_A = float(cfg.DC_OFFSETS[0])
        assert dc_offset_A < 0.4, f"DC offset {dc_offset_A} A exceeds safe range for the LED driver"
        min_freq = float(cfg.F_MINS[0])
        max_freq = float(cfg.F_MAXS[0])
        osc_fs = float(cfg.OSC_SAMPLE_RATES[0])
        subcarrier_spacing = float(cfg.SUBCARRIER_SPACING)

        pwr_supply.set_25V(voltage=4, current=dc_offset_A)
        pwr_supply.enable_output()
        Exp.log(f"DC offset set to {dc_offset_A:.3f} A")

        check_channel = CheckChannel(awg_driver=awg, osc_driver=osc, data_channel=3)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        parent_dir = EXP_DIR / f"{RUN_NAME}_controlled_cdf_{timestamp}"
        parent_dir.mkdir(parents=True, exist_ok=True)
        Exp.log(f"controlled-CDF directory: {parent_dir}")

        # OFDM structure for training comes from the collection modulator (matches the
        # dataset the channel models were fit on); the E/D itself trains on its own
        # constellation set in the ENCODER_DECODER config
        mod_collection = ModulateDataOFDM(
            constellation=get_constellation(cfg.MODULATION_FORMAT),
            f_min=min_freq, f_max=max_freq,
            subcarrier_spacing=subcarrier_spacing,
            preamble_method="zadoff_chu",
            awg_table_fraction=cfg.AWG_TABLE_FRACTION,
            cyclic_prefix_fraction=cfg.CP_LENGTH_FRACTION,
            upsample_factor=cfg.UPSAMPLE_FACTOR,
            preamble_length=cfg.PREAMBLE_LENGTH,
            power_min=getattr(cfg, "POWER_MIN", None),
            power_max=getattr(cfg, "POWER_MAX", None),
            jitter_power=getattr(cfg, "JITTER_POWER", 0.0),
            clip_threshold=float(cfg.CLIP_THRESHOLD),
        )

        prob_channel_ids = {cm["run_id"] for cm in best_channels
                            if cm.get("distribution", "none") not in (None, "none")}

        # validation transmits its own clean constellation (no power sweep or jitter)
        val_constellation = get_constellation(Exp.config.ENCODER_DECODER_VALIDATION.CONSTELLATION)
        mod_val = ModulateDataOFDM(
            constellation=val_constellation,
            f_min=min_freq, f_max=max_freq,
            subcarrier_spacing=subcarrier_spacing,
            preamble_method="zadoff_chu",
            awg_table_fraction=cfg.AWG_TABLE_FRACTION,
            cyclic_prefix_fraction=cfg.CP_LENGTH_FRACTION,
            upsample_factor=cfg.UPSAMPLE_FACTOR,
            preamble_length=cfg.PREAMBLE_LENGTH,
            power_min=None, power_max=None, jitter_power=0.0,
            clip_threshold=float(cfg.CLIP_THRESHOLD),
        )
        demod_val = DemodulateDataOFDM(
            constellation=val_constellation,
            f_min=min_freq, f_max=max_freq, subcarrier_spacing=subcarrier_spacing,
            preamble_method="zadoff_chu",
            baseband_fft_length=mod_val.baseband_fft_length,
            cyclic_prefix_length=mod_val.cyclic_prefix_length,
            upsample_factor=cfg.UPSAMPLE_FACTOR, debug=False)

        f_AWG = mod_val.awg_frequency
        osc.set_horizontal_scale(0.2 / f_AWG)
        send_waveform = SendWaveform(
            fs=mod_val.fs_out, awg_driver=awg, freq=f_AWG, amplitude=18, offset=0)
        measure_waveform = MeasureWaveform(
            fs_in=mod_val.fs_out, fs_out=osc_fs, osc_driver=osc,
            input_signal_frequency=f_AWG, trigger_channel=1, data_channel=3, debug=False)
        resample_waveform = ResampleMeasuredWaveform(
            fs_in=osc_fs, fs_out=mod_val.baseband_sampling_rate, debug=False)
        fractional_sync = FractionalSync(
            fs=mod_val.baseband_sampling_rate, f_min=min_freq, f_max=max_freq, debug=False)

        def train_and_validate(grid_config, select_points, sweep_name):
            '''Train one E/D grid and validate every survivor on the live channel, returning
            the two experiment directories the plots read from.'''
            ed_gs = EncoderDecoderGridSearch(
                grid_config, channel_models=best_channels, dataset_path=dataset_path,
                experiments_dir=parent_dir, experiment_name=f"ed_train_{sweep_name}",
                preamble_length=int(cfg.PREAMBLE_LENGTH),
                clip_threshold=float(cfg.CLIP_THRESHOLD),
                device=device, seed=seed)
            ed_gs.points = select_points(ed_gs.points)
            Exp.log(f"{sweep_name} sweep -> {len(ed_gs.points)} E/Ds")
            ed_exp_dir = ed_gs.run(ofdm_config=OFDMConfig.from_modulator(mod_collection))

            ed_models = select_encoder_decoders(ed_exp_dir, channel_exp_dir=channel_exp_dir)
            # shuffle so grid position (hence the swept value) is decorrelated from validation
            # time: any slow channel drift over the run spreads as noise, not as a sweep bias
            random.Random(seed).shuffle(ed_models)

            check_channel.run(f"pre-validation ({sweep_name})")
            osc.set_record_length(200_000)
            osc.set_horizontal_scale(0.2 / f_AWG)
            osc.configure_channel(ch=3, scale=cfg.OSC_SCALE, offset=0)

            validation = EncoderDecoderValidation(
                ed_models,
                (mod_val, send_waveform, measure_waveform, resample_waveform,
                 fractional_sync, demod_val),
                num_trials=num_trials,
                noise_floor_points=noise_floor_points,
                constellation=val_constellation,
                clip_value=float(cfg.CLIP_THRESHOLD),
                device=device, seed=seed, experiments_dir=parent_dir,
                experiment_name=f"ed_val_{sweep_name}", debug=False)
            val_exp_dir = validation.run()
            check_channel.run(f"post-validation ({sweep_name})")
            return ed_exp_dir, val_exp_dir

        if run_annealing_sweep:
            # annealing only affects probabilistic channels, so drop the redundant annealing
            # variants on nonprob channels (they would retrain identical E/Ds and waste hardware)
            def keep_anneal_points(points):
                return [pt for pt in points
                        if pt["channel_run_id"] in prob_channel_ids
                        or float(pt["params"]["noise_anneal_start"]) == BASELINE_ANNEAL]

            Exp.log(f"annealing sweep {anneal_starts} (nonprob channels pinned to {BASELINE_ANNEAL})")
            anneal_ed_dir, anneal_val_dir = train_and_validate(
                build_seed_swept_grid(seeds, anneal_starts), keep_anneal_points, "anneal")
            plot_per_form_ecdf(anneal_val_dir, anneal_ed_dir, parent_dir / "per_form_evm_ecdf.png")
        else:
            Exp.log("annealing sweep disabled, going straight to the clip-penalty sweep")

        if run_clip_penalty_sweep:
            rhos = list(test_cfg["RHOS"])
            clip_weights = list(test_cfg["CLIP_WEIGHTS"])
            clip_sweep_anneal_start = float(test_cfg.CLIP_SWEEP_ANNEAL_START)
            Exp.log(f"clip-penalty sweep rhos={rhos} weights={clip_weights} at "
                    f"anneal={clip_sweep_anneal_start} (prob channels only, plus a "
                    f"weight-{BASELINE_CLIP_WEIGHT:g} control)")
            clip_ed_dir, clip_val_dir = train_and_validate(
                build_clip_swept_grid(seeds, clip_sweep_anneal_start, rhos, clip_weights),
                lambda points: keep_clip_sweep_points(points, prob_channel_ids, float(rhos[0])),
                "clip")

            plot_clip_penalty_ecdf(clip_val_dir, parent_dir / "clip_penalty_evm_ecdf.png")

            # one drive figure per penalty setting, each against the same weight-0 control,
            # so the settings can be compared by flipping between the files
            for plot_rho, plot_weight in validated_clip_settings(clip_val_dir):
                out_path = parent_dir / f"clip_drive_rho{plot_rho:g}_weight{plot_weight:g}.png"
                plot_clip_drive_distribution(
                    clip_val_dir, clip_ed_dir / "gain_curve.npz",
                    rho=plot_rho, weight=plot_weight,
                    clip_threshold=float(cfg.CLIP_THRESHOLD),
                    out_path=out_path)
                Exp.log(f"drive distribution rho={plot_rho:g} weight={plot_weight:g} -> {out_path.name}")

        Exp.log(f"controlled-CDF complete: {parent_dir}")
