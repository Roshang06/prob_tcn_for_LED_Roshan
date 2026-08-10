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
    ChannelModelGridSearch, EncoderDecoderGridSearch, EncoderDecoderValidation,
    select_channel_models, select_encoder_decoders,
)
from modules.grid_search.base import generate_run_name
from modules.utils import OFDMConfig
from plot_annealing_interaction import (load_sweep_table, plot_annealing_interaction,
                                        plot_penalty_sweep_figures)

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

BASELINE_PENALTY_WEIGHT = 0.0
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
    and annealing are the only sources of variation.'''
    grid_config = _fixed_ed_grid_config()
    grid_config["params"]["seed"] = list(seeds)
    grid_config["params"]["noise_anneal_start"] = list(anneal_starts)
    return grid_config


# penalty families the sweep can compare
PENALTY_FAMILIES = (
    ("DRIVE_MEAN_POWER_WEIGHTS", "drive_mean_power_weight", "drive mean power weight"),
    ("KURTOSIS_WEIGHTS", "kurtosis_weight", "kurtosis weight"),
)


def build_penalty_swept_grid(seeds, anneal_starts, family_weights):
    '''Seed x noise_anneal_start x one penalty at a time, plus a plain no-penalty reference.
    family_weights maps each E/D penalty param to its weights.'''
    grid_config = _fixed_ed_grid_config()
    params = grid_config["params"]
    params["seed"] = list(seeds)
    params["noise_anneal_start"] = list(anneal_starts)
    for param, weights in family_weights.items():
        params[param] = [BASELINE_PENALTY_WEIGHT] + list(weights)
    return grid_config


def keep_penalty_sweep_points(points, prob_channel_ids, penalty_params,
                              combined_mean_power_weight=None):
    '''Prob channels only, since annealing is inert on a deterministic channel. At most one
    penalty is active per run, so the families stay separable.

    combined_mean_power_weight additionally keeps kurtosis runs that carry that one mean-power weight. The
    kurtosis penalty is scale free, so on its own it leaves the drive amplitude to the data
    loss; pinning mean power fixes the amplitude and isolates drive shape as the only variable.'''
    kept = []
    for point in points:
        if point["channel_run_id"] not in prob_channel_ids:
            continue

        params = point["params"]
        active = [name for name in penalty_params if float(params.get(name, 0.0)) > 0]
        combined = (combined_mean_power_weight is not None
                    and sorted(active) == ["drive_mean_power_weight", "kurtosis_weight"]
                    and float(params["drive_mean_power_weight"]) == combined_mean_power_weight)
        if len(active) > 1 and not combined:
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
        run_penalty_sweep = bool(getattr(test_cfg, "RUN_PENALTY_SWEEP", False))
        if not any((run_annealing_sweep, run_penalty_sweep)):
            Exp.log("every CONTROLLED_CDF sweep is disabled, nothing to do")
            raise SystemExit(0)

        dataset_path = cfg.DATASET_PATH
        exclude_models = [m.lower() for m in (getattr(test_cfg, "EXCLUDE_MODELS", None) or [])]

        channel_exp_dir = getattr(test_cfg, "CHANNEL_EXP_DIR", None)
        if channel_exp_dir:
            channel_exp_dir = Path(channel_exp_dir)
        else:
            # no finished grid to reuse, so train one here. 
            with open(CONFIG_FILE, encoding="utf-8") as f:
                channel_grid_config = yaml.safe_load(f)["CHANNEL_GRID_SEARCH"]
            channel_grid_config["models"] = [entry for entry in channel_grid_config["models"]
                                             if entry["model"].lower() not in exclude_models]
            if not channel_grid_config["models"]:
                raise ValueError(f"EXCLUDE_MODELS {exclude_models} removed every channel model "
                                 "from CHANNEL_GRID_SEARCH, nothing left to train")

            trained_forms = sorted({entry["model"] for entry in channel_grid_config["models"]})
            Exp.log(f"no CHANNEL_EXP_DIR set: training channel models here ({trained_forms})")
            channel_exp_dir = ChannelModelGridSearch(
                channel_grid_config, dataset_path=dataset_path, experiments_dir=EXP_DIR,
                device=device, seed=seed, run_prefix=RUN_NAME).run()
            Exp.log(f"trained channel models: {channel_exp_dir}")

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
            Exp.log("annealing sweep disabled, going straight to the penalty sweep")

        if run_penalty_sweep:
            families = {param: [float(w) for w in (test_cfg[key] or [])]
                        for key, param, _ in PENALTY_FAMILIES if test_cfg[key]}
            if not families:
                raise ValueError("the penalty sweep needs weights for at least one of "
                                 f"{[key for key, _, _ in PENALTY_FAMILIES]}")
            combined_mean_power_weight = getattr(test_cfg, "COMBINED_MEAN_POWER_WEIGHT", None)
            combined_mean_power_weight = float(combined_mean_power_weight) if combined_mean_power_weight else None
            Exp.log(f"penalty sweep anneal={anneal_starts} families={families}, against a "
                    f"plain no-penalty reference (one penalty at a time, prob channels only). "
                    f"kurtosis is also swept on top of mean power pinned at "
                    f"{combined_mean_power_weight}")
            penalty_ed_dir, penalty_val_dir = train_and_validate(
                build_penalty_swept_grid(seeds, anneal_starts, families),
                lambda points: keep_penalty_sweep_points(points, prob_channel_ids,
                                                              families, combined_mean_power_weight),
                "penalty")

            plot_penalty_sweep_figures(load_sweep_table(penalty_val_dir), parent_dir)

        Exp.log(f"controlled-CDF complete: {parent_dir}")
