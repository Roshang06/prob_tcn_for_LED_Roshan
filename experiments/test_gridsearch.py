'''
Synthetic end-to-end test of the channel model and encoder/decoder grid searches.
No hardware, no zarr dataset — all data is generated here.

Run from the repo root:
    python experiments/test_gridsearch.py
'''
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

# Ensure repo root is on the path regardless of launch location
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.chdir(Path(__file__).resolve().parents[1])

from modules.constellation_diagram import get_constellation
from modules.grid_search import (
    ChannelModelGridSearch, EncoderDecoderGridSearch, select_channel_models,
    MODEL_REGISTRY,
)
from modules.models import TCN
from modules.utils import OFDMConfig
from modules.experimental_blocks import ModulateDataOFDM

from pyflux.core.block import Signal

# ── Experiment output ────────────────────────────────────────────────────────
EXP_DIR   = Path("data/experiments/test_gridsearch")
DEVICE    = "cpu"
SEED      = 42

# ── Synthetic OFDM geometry ──────────────────────────────────────────────────
K_MIN            = 30
K_MAX            = 100
SUBCARRIER_SPACING_HZ = 10e3


ofdm_modulator = ModulateDataOFDM(
    constellation=get_constellation("qpsk"),
    f_min=K_MIN * SUBCARRIER_SPACING_HZ,
    f_max=K_MAX * SUBCARRIER_SPACING_HZ,
    subcarrier_spacing=SUBCARRIER_SPACING_HZ,
    preamble_method="zadoff_chu",
    awg_table_fraction=0.5,
    cyclic_prefix_fraction=0.125,
    upsample_factor=2,
    preamble_length=256
)

OFDM_CONFIG = OFDMConfig.from_modulator(ofdm_modulator)


def make_synthetic_dataset(n_frames: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    '''Generate random QPSK frames in the time domain and pass them through a
    synthetic channel (short FIR + additive noise) to produce (X_sent, Y_received).'''
    torch.manual_seed(SEED)

    sent_time_frames = []
    for _ in range(n_frames):
        x = Signal(data=np.zeros(1), sampling_rate=ofdm_modulator.fs_out)
        x = ofdm_modulator.run(x)
        sent_time_frames.append(torch.from_numpy(x.artifact_container["sent_baseband"]))
        

    sent_time = torch.stack(sent_time_frames).float()

    fir = torch.tensor([[[0.8, 0.3, -0.1]]], dtype=torch.float32)
    recv_time = F.conv1d(sent_time.unsqueeze(1), fir, padding=2).squeeze(1)[:, :sent_time.shape[1]]
    recv_time = recv_time + 0.02 * torch.randn_like(recv_time)

    return sent_time, recv_time

CHANNEL_GRID = {
    "VAL_FRACTION": 0.2,
    "models": [
        {
            "model": "tcn",
            "params": {
                "nlayers":          [2, 3],
                "dilation_base":    2,
                "num_taps":         5,
                "hidden_channels":  [8, 16],
                "learn_noise":      False,
                "gaussian":         True,
                "epochs":           100,
                "lr":               1e-3,
                "batch_size":       16,
            },
        },
        {
            "model": "gmp",
            "params": {
                "memory_linear":       [4, 8],
                "memory_nonlinear":    4,
                "nonlinearity_order":  3,
                "cross_term_depth":    [0, 1],
                "ridge":               1e-3,
            },
        },
    ]
}

ENCODER_DECODER_GRID = {
    "constellation":      "qpsk",
    "preamble_amplitude": 3.0,
    "params": {
        "nlayers":         [2, 3],
        "dilation_base":   2,
        "num_taps":        5,
        "hidden_channels": 8,
        "epochs":          500,
        "lr":              1e-3,
        "weight_decay":    1e-5,
        "batch_size":      8,
    },
}


if __name__ == "__main__":
    print("=" * 60)
    print("Stage 1: channel model grid search")
    print("=" * 60)

    X, Y = make_synthetic_dataset(n_frames=1000)
    print(f"Synthetic dataset: X{tuple(X.shape)}  Y{tuple(Y.shape)}")

    channel_gs = ChannelModelGridSearch(
        CHANNEL_GRID,
        dataset_path="synthetic",
        experiments_dir=EXP_DIR,
        device=DEVICE,
        seed=SEED,
        experiment_name="channel_models",
    )
    channel_exp_dir = channel_gs.run(data=(X, Y))
    print(f"\nChannel grid done → {channel_exp_dir}")

    print("\n" + "=" * 60)
    print("Stage 2: select channel models (best per architecture, by val loss)")
    print("=" * 60)

    # SELECTED_RUN_IDS = ["tcn_xxxxxxxx"]  # pin specific runs instead of best-per-family
    SELECTED_RUN_IDS = None
    best_channels = select_channel_models(channel_exp_dir, mode="best", run_ids=SELECTED_RUN_IDS)
    all_channels  = select_channel_models(channel_exp_dir, mode="all")
    channel_models_by_id = {cm["run_id"]: cm for cm in all_channels}

    for cm in best_channels:
        print(f"  {cm['model']:4s}  run={cm['run_id']}")
    print(f"\n  {len(all_channels)} total channel runs available; using best per arch for E/D stage.")

    print("\n" + "=" * 60)
    print("Stage 3: encoder/decoder grid search")
    print("=" * 60)

    ed_gs = EncoderDecoderGridSearch(
        ENCODER_DECODER_GRID,
        channel_models=best_channels,
        dataset_path="synthetic",
        experiments_dir=EXP_DIR,
        device=DEVICE,
        seed=SEED,
        experiment_name="encoder_decoder",
    )
    ed_exp_dir = ed_gs.run(ofdm_config=OFDM_CONFIG)
    print(f"\nEncoder/decoder grid done → {ed_exp_dir}")

    print("\n" + "=" * 60)
    print("Results summary")
    print("=" * 60)

    for label, summary_dir in [("channel", channel_exp_dir), ("encoder/decoder", ed_exp_dir)]:
        lb = Path(summary_dir) / "summary" / "leaderboard.csv"
        rows = list(csv.DictReader(open(lb)))
        print(f"\n  {label} leaderboard (sorted by rrmse_pct):")
        for row in sorted(rows, key=lambda r: float(r["rrmse_pct"])):
            extra = f"  ber={float(row['ber']):.4f}" if row.get("ber") else ""
            print(f"    {row['run_id']}  rrmse_pct={float(row['rrmse_pct']):.4f}{extra}  "
                  f"params={row['num_params']}  t={row['train_seconds']}s")
