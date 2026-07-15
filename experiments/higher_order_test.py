'''
Higher-order spectral-efficiency test, run after a finished train_and_validate.

Takes the encoder/decoder that reached the best validation EVM (or a pinned run via
HIGHER_ORDER_TEST.ED_RUN_ID), reads its architecture and the channel model it was
trained against, and for each higher-order constellation trains a fresh E/D of the
same architecture on that channel model, then validates it on the live hardware.
The fresh training is deliberate: weights cannot transfer across constellations.

Each constellation produces a received-equalized constellation diagram annotated with
the received EVM, BER and the spectral efficiency it sustains. Everything lands in a
single {run}_higher_order_test_{timestamp} experiment directory.
'''
import json
import yaml
from datetime import datetime
from pathlib import Path

import numpy as np

from pyflux.core.experiment import ExperimentalContext
from modules.experimental_blocks import *
from modules.constellation_diagram import get_constellation
from modules.grid_search import (
    EncoderDecoderGridSearch, EncoderDecoderValidation,
    select_channel_models, select_encoder_decoders,
)
from modules.grid_search.encoder_decoder import ARCH_KEYS
from modules.grid_search.base import generate_run_name
from modules.utils import OFDMConfig

HERE = Path(__file__).resolve().parent

CONFIG_FILE = HERE / "train_and_validate.yml"
EXP_DIR = HERE.parent / "data/experiments/train_and_validate"


def _read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def select_best_ed(validation_exp_dir, ed_run_id=None):
    '''Pick the E/D to base the higher-order test on: the pinned run_id if given,
    otherwise the lowest validation EVM. Returns its architecture and originating
    channel run_id from the validation run table.'''
    rows = _read_jsonl(Path(validation_exp_dir) / "runs.jsonl")
    if ed_run_id:
        chosen = next((r for r in rows if r["run_id"] == ed_run_id), None)
        if chosen is None:
            raise ValueError(f"ED_RUN_ID {ed_run_id} not found in {validation_exp_dir}")
    else:
        chosen = min(rows, key=lambda r: r["evm_pct"])

    arch = {key: chosen[key] for key in ARCH_KEYS}
    return chosen["run_id"], arch, chosen["channel_run_id"]


def build_single_ed_grid(constellation_name, arch):
    '''ENCODER_DECODER config collapsed to one training point: the best E/D's
    architecture on the higher-order constellation, sweeps flattened to their first value.'''
    with open(CONFIG_FILE, encoding="utf-8") as f:
        full = yaml.safe_load(f)
    grid_config = {k.lower(): v for k, v in full["ENCODER_DECODER"].items()}

    params = dict(grid_config["params"])
    for key in ARCH_KEYS:
        params[key] = arch[key]

    def first(value):
        return value[0] if isinstance(value, list) else value

    params["noise_anneal_start"] = first(params.get("noise_anneal_start", 1.0))
    params["weight_decay"] = first(params.get("weight_decay", 0.0))

    grid_config["params"] = params
    grid_config["constellation"] = constellation_name
    return grid_config


if __name__ == "__main__":
    RUN_NAME = generate_run_name()

    with ExperimentalContext(CONFIG_FILE=CONFIG_FILE, create_log_file=True, run_name=RUN_NAME) as Exp:
        test_cfg = Exp.config.ENCODER_DECODER_VALIDATION.HIGHER_ORDER_TEST
        if not getattr(test_cfg, "ENABLED", False):
            Exp.log("HIGHER_ORDER_TEST.ENABLED is false, nothing to do")
            raise SystemExit(0)

        cfg = Exp.config.DATA_COLLECTION
        device = Exp.config.RUNTIME.DEVICE
        seed = int(Exp.config.RUNTIME.SEED)

        constellation_names = test_cfg.CONSTELLATION
        if isinstance(constellation_names, str):
            constellation_names = [constellation_names]
        num_trials = int(test_cfg.N)
        ed_run_id = getattr(test_cfg, "ED_RUN_ID", None)

        channel_exp_dir = getattr(test_cfg, "CHANNEL_EXP_DIR", None)
        validation_exp_dir = getattr(test_cfg, "VALIDATION_EXP_DIR", None)
        if not channel_exp_dir or not validation_exp_dir:
            raise ValueError("HIGHER_ORDER_TEST needs CHANNEL_EXP_DIR and VALIDATION_EXP_DIR "
                             "set to the finished channel_models_ and ed_validation_ directories")
        channel_exp_dir = Path(channel_exp_dir)
        validation_exp_dir = Path(validation_exp_dir)
        dataset_path = cfg.DATASET_PATH

        best_ed_run_id, arch, channel_run_id = select_best_ed(validation_exp_dir, ed_run_id)
        Exp.log(f"basing higher-order test on E/D {best_ed_run_id} "
                f"(arch {arch}, channel {channel_run_id})")

        channel_models = select_channel_models(channel_exp_dir, run_ids=[channel_run_id])

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

        pwr_supply.set_6V(voltage=4, current=dc_offset_A)
        pwr_supply.enable_output()
        Exp.log(f"DC offset set to {dc_offset_A:.3f} A")

        check_channel = CheckChannel(awg_driver=awg, osc_driver=osc, data_channel=3)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        parent_dir = EXP_DIR / f"{RUN_NAME}_higher_order_test_{timestamp}"
        parent_dir.mkdir(parents=True, exist_ok=True)
        Exp.log(f"higher-order test directory: {parent_dir}")

        check_channel.run("start")

        for constellation_name in constellation_names:
            constellation = get_constellation(constellation_name)
            modulation_order = 2 ** constellation.bits_per_symbol
            stage = f"{modulation_order}apsk"
            Exp.log(f"=== {constellation_name} ({modulation_order} APSK) ===")

            mod_ofdm = ModulateDataOFDM(
                constellation=constellation,
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
            demod_ofdm = DemodulateDataOFDM(
                constellation=constellation,
                f_min=min_freq, f_max=max_freq, subcarrier_spacing=subcarrier_spacing,
                preamble_method="zadoff_chu",
                baseband_fft_length=mod_ofdm.baseband_fft_length,
                cyclic_prefix_length=mod_ofdm.cyclic_prefix_length,
                upsample_factor=cfg.UPSAMPLE_FACTOR, debug=False)

            f_AWG = mod_ofdm.awg_frequency
            osc.set_horizontal_scale(0.2 / f_AWG)
            send_waveform = SendWaveform(
                fs=mod_ofdm.fs_out, awg_driver=awg, freq=f_AWG, amplitude=18, offset=0)
            measure_waveform = MeasureWaveform(
                fs_in=mod_ofdm.fs_out, fs_out=osc_fs, osc_driver=osc,
                input_signal_frequency=f_AWG, trigger_channel=1, data_channel=3, debug=False)
            resample_waveform = ResampleMeasuredWaveform(
                fs_in=osc_fs, fs_out=mod_ofdm.baseband_sampling_rate, debug=False)
            fractional_sync = FractionalSync(
                fs=mod_ofdm.baseband_sampling_rate, f_min=min_freq, f_max=max_freq, debug=False)

            # 1. train a fresh E/D of the best architecture on the chosen channel model
            grid_config = build_single_ed_grid(constellation_name, arch)
            ed_gs = EncoderDecoderGridSearch(
                grid_config, channel_models=channel_models, dataset_path=dataset_path,
                experiments_dir=parent_dir, experiment_name=f"{stage}_train",
                preamble_length=int(cfg.PREAMBLE_LENGTH),
                clip_threshold=float(cfg.CLIP_THRESHOLD),
                device=device, seed=seed)
            ed_exp_dir = ed_gs.run(ofdm_config=OFDMConfig.from_modulator(mod_ofdm))

            # 2. validate it on the live channel, reporting spectral efficiency
            ed_models = select_encoder_decoders(ed_exp_dir, channel_exp_dir=channel_exp_dir)
            symbol_period_s = ((mod_ofdm.baseband_fft_length + mod_ofdm.cyclic_prefix_length)
                               / mod_ofdm.baseband_sampling_rate)
            higher_order_context = {"f_min": min_freq, "f_max": max_freq,
                                    "symbol_period_s": symbol_period_s}

            check_channel.run(f"pre-validation {stage}")
            osc.set_record_length(200_000)
            osc.set_horizontal_scale(0.2 / f_AWG)
            osc.configure_channel(ch=3, scale=cfg.OSC_SCALE, offset=0)

            validation = EncoderDecoderValidation(
                ed_models,
                (mod_ofdm, send_waveform, measure_waveform, resample_waveform,
                 fractional_sync, demod_ofdm),
                num_trials=num_trials,
                constellation=constellation,
                clip_value=float(cfg.CLIP_THRESHOLD),
                higher_order_context=higher_order_context,
                device=device, seed=seed, experiments_dir=parent_dir,
                experiment_name=f"{stage}_val", debug=False)
            val_exp_dir = validation.run()
            Exp.log(f"{stage} validation written to {val_exp_dir}")

        check_channel.run("post-test")
        Exp.log(f"higher-order test complete: {parent_dir}")
