"""
Where do the encoders actually operate relative to the channel-model training range?

For every run in an E/D validation experiment, this replays the exact validation sent
bursts through the stored encoder (deterministic, clamp +-CLIP_THRESHOLD) and computes
the per-trial encoder-output symbol power mean(x^2), the same units as the sweep's
POWER_MIN/POWER_MAX spec (specified power = RMS^2, not RMS). It plots the distribution
across all validated runs as a histogram stacked by channel family, with vertical lines
at the dataset's actual min/max per-burst training power.

Edit the configuration below, then:
    cd <repo_root>/experiments
    python ed_validation_encoder_power.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import zarr
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from modules.models import TCN


# Configuration
SWEEP_PATH = "data/sweeps/dc0.05A_fmin300000_fmax7.6e+06_20260708_1822.zarr"
ED_EXP = "data/experiments/train_and_validate/amber_shore_encoder_decoder_20260711_0627"
VAL_EXP = "data/experiments/train_and_validate/amber_shore_ed_validation_20260711_0804"
CLIP_THRESHOLD = 3.0
N_HIST_BINS = 60
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PLOT_PATH = HERE.parent / "data/plots"

ARCHITECTURE_KEYS = ("nlayers", "dilation_base", "kernel_size", "hidden_channels")


def read_jsonl(path):
    lines = Path(path).read_text().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def main():
    PLOT_PATH.mkdir(parents=True, exist_ok=True)

    sweep = zarr.open_group(HERE.parent / SWEEP_PATH, mode="r")
    attrs = dict(sweep.attrs)
    cyclic_prefix_length = int(attrs["cyclic_prefix_length"])
    symbol_offset = int(attrs["preamble_length"]) + cyclic_prefix_length

    training_power = (sweep["sent_burst"][:, symbol_offset:].astype(np.float64) ** 2).mean(axis=1)
    training_min, training_max = training_power.min(), training_power.max()
    print(f"dataset training power: min={training_min:.3f} max={training_max:.3f} "
          f"(config spec {attrs.get('power_min', '?')}-{attrs.get('power_max', '?')})")

    validation = zarr.open_group(HERE.parent / VAL_EXP / "validation.zarr", mode="r")
    validation_rows = read_jsonl(HERE.parent / VAL_EXP / "runs.jsonl")
    encoder_decoder_rows = {row["run_id"]: row for row in read_jsonl(HERE.parent / ED_EXP / "runs.jsonl")}

    powers_by_form = {}
    for row in validation_rows:
        run_id = row["run_id"]
        ed_run_id = row["model"]  # the validation row's "model" field holds the E/D grid run_id

        architecture = {key: encoder_decoder_rows[ed_run_id][key] for key in ARCHITECTURE_KEYS}
        encoder = TCN(**architecture).to(DEVICE)
        checkpoint = torch.load(HERE.parent / ED_EXP / "runs" / ed_run_id / "model.pt",
                                map_location=DEVICE, weights_only=True)
        encoder.load_state_dict(checkpoint["encoder"])
        encoder.eval()

        symbol = torch.tensor(validation[run_id]["sent_time"][:].astype(np.float32), device=DEVICE)
        symbol_length = symbol.shape[1]
        burst = torch.hstack([symbol[:, -cyclic_prefix_length:], symbol])  # [CP | symbol]
        with torch.no_grad():
            encoded = encoder(burst).clamp(-CLIP_THRESHOLD, CLIP_THRESHOLD)

        trial_powers = (encoded[:, -symbol_length:] ** 2).mean(dim=1).cpu().numpy()
        powers_by_form.setdefault(row["channel_form"], []).append(trial_powers)

    all_powers = []
    print(f"\nencoder-output symbol power across {len(validation_rows)} validated runs:")
    for form, power_chunks in powers_by_form.items():
        powers = np.concatenate(power_chunks)
        all_powers.append(powers)
        fraction_below_min = (powers < training_min).mean() * 100
        print(f"  {form:14s} n={powers.size:5d}  mean={powers.mean():.3f}  "
              f"range=({powers.min():.3f}, {powers.max():.3f})  below train min: {fraction_below_min:.1f}%")
    all_powers = np.concatenate(all_powers)
    print(f"  {'ALL':14s} n={all_powers.size:5d}  mean={all_powers.mean():.3f}  "
          f"below train min: {(all_powers < training_min).mean() * 100:.1f}%")

    fig, ax = plt.subplots(figsize=(9, 5))
    histogram_bins = np.histogram_bin_edges(
        np.concatenate([all_powers, [training_min, training_max]]), bins=N_HIST_BINS)
    ax.hist([np.concatenate(chunks) for chunks in powers_by_form.values()],
            bins=histogram_bins, stacked=True, label=list(powers_by_form.keys()))
    ax.axvline(training_min, color="k", ls="--", lw=1.5, label=f"train min ({training_min:.2f})")
    ax.axvline(training_max, color="k", ls=":", lw=1.5, label=f"train max ({training_max:.2f})")
    ax.set_xlabel("Encoder-output symbol power  mean(x²)")
    ax.set_ylabel("Validation trials")
    ax.set_title(f"Encoder operating power during validation "
                 f"({len(validation_rows)} runs × {validation[validation_rows[0]['run_id']]['sent_time'].shape[0]} trials)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(PLOT_PATH / "ed_validation_encoder_power.png", dpi=130, bbox_inches="tight")
    fig.savefig(PLOT_PATH / "ed_validation_encoder_power.svg", format="svg", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
