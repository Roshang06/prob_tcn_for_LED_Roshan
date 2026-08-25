import json
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

from pyflux.core.experiment import ExperimentalContext
from modules.experimental_blocks import *
from modules.constellation_diagram import get_constellation
from modules.grid_search import (EncoderDecoderGridSearch, EncoderDecoderValidation,
                                 select_channel_models, select_encoder_decoders)
from modules.grid_search.base import generate_run_name, run_id
from modules.utils import OFDMConfig
from plot_find_regularization import (plot_delta_vs_anneal, plot_delta_pvalue_heatmap,
                                       write_summary_table)

HERE = Path(__file__).resolve().parent
CONFIG_FILE = HERE / (sys.argv[1] if len(sys.argv) > 1 else "find_regularization.yml")
EXP_DIR = HERE.parent / "data/experiments/train_and_validate"
LOG_DIR = HERE.parent / "data/logs"
BOOTSTRAP_N = 10000


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def fixed_ed_grid_config(config):
    grid = {k.lower(): v for k, v in dict(config.ENCODER_DECODER).items()}
    grid["params"] = {k: (v[0] if isinstance(v, list) else v) for k, v in dict(grid["params"]).items()}
    return grid


def build_sweep_points(base_params, channel_run_id, seeds, anneal_starts, power_weights,
                       kurtosis_weight, kurtosis_target):
    points = []
    for seed in seeds:
        for anneal in anneal_starts:
            for power in power_weights:
                params = {**base_params, "seed": int(seed), "noise_anneal_start": float(anneal),
                          "drive_mean_power_weight": float(power),
                          "kurtosis_weight": float(kurtosis_weight),
                          "kurtosis_target": float(kurtosis_target)}
                points.append({"model": "tcn_ae", "params": params, "channel_run_id": channel_run_id})
        ctrl = {**base_params, "seed": int(seed), "noise_anneal_start": 1.0,
                "drive_mean_power_weight": 0.0, "kurtosis_weight": 0.0,
                "kurtosis_target": float(kurtosis_target), "deterministic_channel": True}
        points.append({"model": "tcn_ae", "params": ctrl, "channel_run_id": channel_run_id})
    return points


def point_label(params):
    if params.get("deterministic_channel"):
        return "ctrl", 1.0, 0.0
    return "reg", float(params["noise_anneal_start"]), float(params["drive_mean_power_weight"])


def compute_stats(pairs):
    rows = []
    for dc_ma, group in pairs.groupby("dc_ma"):
        ctrl = group[group.condition == "ctrl"].set_index("seed")["evm"]
        reg = group[group.condition == "reg"]
        for (anneal, power), cell in reg.groupby(["anneal", "power"]):
            cell = cell.set_index("seed")["evm"]
            shared = ctrl.index.intersection(cell.index)
            ctrl_evm = ctrl.loc[shared].to_numpy()
            cell_evm = cell.loc[shared].to_numpy()
            delta = ctrl_evm - cell_evm
            rng = np.random.default_rng(0)
            boot = np.array([np.median(rng.choice(delta, size=len(delta), replace=True))
                             for _ in range(BOOTSTRAP_N)])
            ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
            try:
                p_value = stats.wilcoxon(ctrl_evm, cell_evm).pvalue
            except ValueError:
                p_value = float("nan")
            rows.append({"dc_ma": int(dc_ma), "power": float(power), "anneal": float(anneal),
                         "n": int(len(delta)), "median_delta": float(np.median(delta)),
                         "ci_lo": float(ci_lo), "ci_hi": float(ci_hi), "p_value": float(p_value)})
    return pd.DataFrame.from_records(rows)


if __name__ == "__main__":
    RUN_NAME = generate_run_name()

    with ExperimentalContext(CONFIG_FILE=CONFIG_FILE, create_log_file=True, run_name=RUN_NAME,
                             log_dir=LOG_DIR) as Exp:
        cfg = Exp.config.TEST_REGULARIZATION
        device = Exp.config.RUNTIME.DEVICE
        seed = int(Exp.config.RUNTIME.SEED)

        seeds = list(range(1, int(cfg.SEEDS) + 1))
        anneal_starts = [float(a) for a in cfg.ANNEAL_STARTS]
        power_weights = [float(w) for w in cfg.POWER_WEIGHTS]
        num_trials = int(cfg.VALIDATIONS_PER_SEED)
        kurtosis_weight = float(cfg.KURTOSIS_WEIGHT)
        kurtosis_target = float(cfg.KURTOSIS_TARGET)
        excluded = [m.lower() for m in cfg.EXCLUDE_MODELS]

        dc_offsets = [float(x) for x in cfg.DC_OFFSETS]
        f_min = float(cfg.F_MINS[0])
        f_maxs = [float(x) for x in cfg.F_MAXS]
        channel_dirs = [EXP_DIR / d for d in cfg.CHANNEL_DIRS]
        osc_fs = float(cfg.OSC_SAMPLE_RATES[0])
        subcarrier_spacing = float(cfg.SUBCARRIER_SPACING)

        ed_grid_config = fixed_ed_grid_config(Exp.config)
        base_params = ed_grid_config["params"]
        val_constellation = get_constellation(Exp.config.ENCODER_DECODER_VALIDATION.CONSTELLATION)

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

        check_channel = CheckChannel(awg_driver=awg, osc_driver=osc, data_channel=3)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        parent_dir = EXP_DIR / f"{RUN_NAME}_find_regularization_{timestamp}"
        parent_dir.mkdir(parents=True, exist_ok=True)
        Exp.log(f"find_regularization directory: {parent_dir}")

        records = []
        for dc_offset, f_max, channel_dir in zip(dc_offsets, f_maxs, channel_dirs):
            dc_ma = int(round(dc_offset * 1000))
            Exp.log(f"Starting sweep at {dc_ma} mA ({dc_offset} A). . .")

            assert dc_offset < 0.4, f"DC offset {dc_offset} A exceeds safe range for the LED driver"
            pwr_supply.set_25V(voltage=4, current=dc_offset)
            pwr_supply.enable_output()

            best_channels = select_channel_models(channel_dir, mode="best")
            prob_channels = [cm for cm in best_channels
                             if cm["model"].lower() not in excluded
                             and cm.get("distribution", "none") not in (None, "none")]
            best_prob = prob_channels[0]
            Exp.log(f"{dc_ma} mA best prob channel {best_prob['run_id']} "
                    f"dist={best_prob.get('distribution')}")

            dataset_path = yaml.safe_load((channel_dir / "experiment_config.yaml").read_text())["dataset_path"]

            mod_collection = ModulateDataOFDM(
                constellation=get_constellation(cfg.MODULATION_FORMAT),
                f_min=f_min, f_max=f_max, subcarrier_spacing=subcarrier_spacing,
                preamble_method="zadoff_chu", awg_table_fraction=cfg.AWG_TABLE_FRACTION,
                cyclic_prefix_fraction=cfg.CP_LENGTH_FRACTION, upsample_factor=cfg.UPSAMPLE_FACTOR,
                preamble_length=cfg.PREAMBLE_LENGTH, power_min=cfg.POWER_MIN, power_max=cfg.POWER_MAX,
                jitter_power=cfg.JITTER_POWER, clip_threshold=float(cfg.CLIP_THRESHOLD))

            mod_val = ModulateDataOFDM(
                constellation=val_constellation, f_min=f_min, f_max=f_max,
                subcarrier_spacing=subcarrier_spacing, preamble_method="zadoff_chu",
                awg_table_fraction=cfg.AWG_TABLE_FRACTION, cyclic_prefix_fraction=cfg.CP_LENGTH_FRACTION,
                upsample_factor=cfg.UPSAMPLE_FACTOR, preamble_length=cfg.PREAMBLE_LENGTH,
                power_min=None, power_max=None, jitter_power=0.0, clip_threshold=float(cfg.CLIP_THRESHOLD))
            demod_val = DemodulateDataOFDM(
                constellation=val_constellation, f_min=f_min, f_max=f_max,
                subcarrier_spacing=subcarrier_spacing, preamble_method="zadoff_chu",
                baseband_fft_length=mod_val.baseband_fft_length,
                cyclic_prefix_length=mod_val.cyclic_prefix_length,
                upsample_factor=cfg.UPSAMPLE_FACTOR, debug=False)

            f_AWG = mod_val.awg_frequency
            osc.set_horizontal_scale(0.2 / f_AWG)
            send_waveform = SendWaveform(fs=mod_val.fs_out, awg_driver=awg, freq=f_AWG, amplitude=18, offset=0)
            measure_waveform = MeasureWaveform(
                fs_in=mod_val.fs_out, fs_out=osc_fs, osc_driver=osc,
                input_signal_frequency=f_AWG, trigger_channel=1, data_channel=3, debug=False)
            resample_waveform = ResampleMeasuredWaveform(
                fs_in=osc_fs, fs_out=mod_val.baseband_sampling_rate, debug=False)
            fractional_sync = FractionalSync(
                fs=mod_val.baseband_sampling_rate, f_min=f_min, f_max=f_max, debug=False)

            ed_gs = EncoderDecoderGridSearch(
                ed_grid_config, channel_models=[best_prob], dataset_path=dataset_path,
                experiments_dir=parent_dir, experiment_name=f"ed_train_{dc_ma}mA",
                preamble_length=int(cfg.PREAMBLE_LENGTH), clip_threshold=float(cfg.CLIP_THRESHOLD),
                device=device, seed=seed)
            points = build_sweep_points(base_params, best_prob["run_id"], seeds, anneal_starts,
                                        power_weights, kurtosis_weight, kurtosis_target)
            ed_gs.points = points
            label_by_run = {run_id(pt): (pt["params"]["seed"], point_label(pt["params"])) for pt in points}
            Exp.log(f"{dc_ma} mA -> {len(points)} E/Ds "
                    f"({len(seeds)} seeds x [{len(anneal_starts)}x{len(power_weights)} + ctrl])")
            ed_exp_dir = ed_gs.run(ofdm_config=OFDMConfig.from_modulator(mod_collection))

            ed_models = select_encoder_decoders(ed_exp_dir, channel_exp_dir=channel_dir)
            random.Random(seed).shuffle(ed_models)

            check_channel.run(f"{dc_ma} mA pre-validation")
            osc.set_record_length(200_000)
            osc.set_horizontal_scale(0.2 / f_AWG)
            osc.configure_channel(ch=3, scale=cfg.OSC_SCALE, offset=0)

            validation = EncoderDecoderValidation(
                ed_models,
                (mod_val, send_waveform, measure_waveform, resample_waveform, fractional_sync, demod_val),
                num_trials=num_trials, constellation=val_constellation,
                clip_value=float(cfg.CLIP_THRESHOLD), device=device, seed=seed,
                experiments_dir=parent_dir, experiment_name=f"ed_val_{dc_ma}mA", debug=False)
            val_exp_dir = validation.run()

            for row in read_jsonl(Path(val_exp_dir) / "runs.jsonl"):
                seed_val, (kind, anneal, power) = label_by_run[row["model"]]
                records.append({"dc_ma": dc_ma, "seed": int(seed_val), "condition": kind,
                                "anneal": anneal, "power": power, "evm": float(row["evm_pct"])})

        pairs = pd.DataFrame.from_records(records)
        pairs_path = parent_dir / "pairs.csv"
        pairs.to_csv(pairs_path, index=False)
        Exp.log(f"saved pairs -> {pairs_path}")

        stats_df = compute_stats(pairs)
        stats_path = parent_dir / "stats.csv"
        stats_df.to_csv(stats_path, index=False)
        Exp.log(f"saved stats -> {stats_path}")

        plot_delta_vs_anneal(stats_df, parent_dir / "delta_vs_anneal.png")
        plot_delta_pvalue_heatmap(stats_df, parent_dir / "delta_pvalue_heatmap.png")
        write_summary_table(stats_df, parent_dir / "summary_table.csv")
        Exp.log(f"find_regularization complete: {parent_dir}")
        Exp.log(f"replot: python {HERE / 'plot_find_regularization.py'} {stats_path}")
