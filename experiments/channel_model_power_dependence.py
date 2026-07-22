"""
Mirror of channel_power_dependence.py, but for a trained channel model: how does the
MODEL's predicted H(f) move with drive power?

Two regimes:
    1. in-band (0.5 to 3.0): the sweep's own U(POWER_MIN, POWER_MAX) bursts, quantile-binned
       by per-burst symbol power, model prediction vs the real received data per bin;
    2. below-band (< 0.5): the same sweep bursts rescaled to fixed target powers below the
       training range (including the ~0.24 E/D operating point), model prediction only.
       This is pure extrapolation, there is no real data to compare against.

Everything is referenced to the highest-power in-band bin, matching
channel_power_dependence.py, so the two figures are directly comparable. The model's
predicted mean is used (probabilistic runs return (noisy, mean, std, ...)).

Edit the configuration below, then:
    cd <repo_root>/experiments
    python channel_model_power_dependence.py
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

from modules.grid_search.adapters import MODEL_REGISTRY
from modules.grid_search.orchestrator import select_channel_models


# Configuration
CHANNEL_EXP_DIR = "data/experiments/train_and_validate/amber_shore_channel_models_20260710_1638"
DATASET_PATH = "data/sweeps/dc0.05A_fmin300000_fmax7.6e+06_20260708_1822.zarr"
RUN_ID = None                               # None -> best probabilistic TCN by val per-burst rRMSE
N_POWER_BINS = 4                            # quantile bins over the in-band per-burst power
LOW_POWER_TARGETS = [0.4, 0.24, 0.1, 0.05]  # symbol powers below the training range
CHUNK_SIZE = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PLOT_PATH = HERE.parent / "data/plots"


def choose_best_prob_tcn(channel_exp_dir):
    lines = (channel_exp_dir / "runs.jsonl").read_text().splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    prob_tcn_runs = [row for row in rows
                     if row["model"] == "tcn" and row.get("distribution", "none") != "none"]

    def score(row):
        for key in ("val_per_burst_rrmse_pct", "val_rrmse_pct"):
            if row.get(key) is not None:
                return row[key]
        return float("inf")

    best = min(prob_tcn_runs, key=score)
    print(f"best probabilistic TCN: {best['run_id']}  dist={best['distribution']}  "
          f"val_rrmse={score(best):.2f}%")
    return best["run_id"]


def predict_mean(adapter, inputs):
    predictions = []
    for start in range(0, inputs.shape[0], CHUNK_SIZE):
        batch = torch.tensor(inputs[start:start + CHUNK_SIZE], dtype=torch.float32, device=DEVICE)
        output = adapter.predict(batch)
        if isinstance(output, tuple):
            output = output[1]
        predictions.append(output.detach().cpu().numpy())
    return np.concatenate(predictions, axis=0)


def carrier_ratio(sent_symbol, received_symbol, active_carrier_indices):
    sent_spectrum = np.fft.fft(sent_symbol.astype(np.float64), norm="ortho", axis=1)[:, active_carrier_indices]
    received_spectrum = np.fft.fft(received_symbol.astype(np.float64), norm="ortho", axis=1)[:, active_carrier_indices]
    return received_spectrum / sent_spectrum


def main():
    PLOT_PATH.mkdir(parents=True, exist_ok=True)
    channel_exp_dir = HERE.parent / CHANNEL_EXP_DIR

    run_id = RUN_ID or choose_best_prob_tcn(channel_exp_dir)
    selected = select_channel_models(channel_exp_dir, run_ids=[run_id])[0]
    adapter = MODEL_REGISTRY[selected["model"]].load(selected["params"], selected["checkpoint"], DEVICE)

    root = zarr.open_group(HERE.parent / DATASET_PATH, mode="r")
    attrs = dict(root.attrs)
    active_carrier_indices = np.array(attrs["active_carrier_indices"])
    cyclic_prefix_length = int(attrs["cyclic_prefix_length"])
    preamble_length = int(attrs["preamble_length"])

    burst_input = root["sent_burst"][:, preamble_length:].astype(np.float64)   # [CP | symbol]
    received_symbol = root["received_burst"][:, preamble_length + cyclic_prefix_length:].astype(np.float64)
    sent_symbol = burst_input[:, cyclic_prefix_length:]
    burst_power = (sent_symbol ** 2).mean(axis=1)

    subcarrier_spacing_hz = float(attrs["f_min_hz"]) / active_carrier_indices[0]
    frequencies_mhz = active_carrier_indices * subcarrier_spacing_hz / 1e6

    predicted_symbol = predict_mean(adapter, burst_input)[:, cyclic_prefix_length:]
    model_ratio = carrier_ratio(sent_symbol, predicted_symbol, active_carrier_indices)
    real_ratio = carrier_ratio(sent_symbol, received_symbol, active_carrier_indices)

    # in-band: quantile-bin the sweep bursts by per-burst power, model vs real per bin
    power_bin_edges = np.quantile(burst_power, np.linspace(0, 1, N_POWER_BINS + 1))
    print(f"in-band per-burst power quantile edges: {np.round(power_bin_edges, 2)}")

    model_response_by_bin = []
    real_response_by_bin = []
    bin_labels = []
    for bin_index in range(N_POWER_BINS):
        in_bin = (burst_power >= power_bin_edges[bin_index]) & (burst_power <= power_bin_edges[bin_index + 1])
        model_response_by_bin.append(np.mean(model_ratio[in_bin], axis=0))
        real_response_by_bin.append(np.mean(real_ratio[in_bin], axis=0))
        bin_labels.append(f"P∈[{power_bin_edges[bin_index]:.2f},{power_bin_edges[bin_index + 1]:.2f}]")

    model_phase_by_bin = [np.unwrap(np.angle(response)) for response in model_response_by_bin]
    real_phase_by_bin = [np.unwrap(np.angle(response)) for response in real_response_by_bin]
    model_magnitude_by_bin = [20 * np.log10(np.abs(response)) for response in model_response_by_bin]

    # below-band: rescale the same bursts to fixed target powers, model prediction only
    low_power_response = []
    low_power_labels = []
    for target_power in LOW_POWER_TARGETS:
        scale = np.sqrt(target_power / burst_power)[:, None]
        low_predicted = predict_mean(adapter, burst_input * scale)[:, cyclic_prefix_length:]
        low_power_response.append(
            np.mean(carrier_ratio(sent_symbol * scale, low_predicted, active_carrier_indices), axis=0))
        low_power_labels.append(f"P={target_power:.2f}")

    low_power_phase = [np.unwrap(np.angle(response)) for response in low_power_response]
    low_power_magnitude = [20 * np.log10(np.abs(response)) for response in low_power_response]

    reference_phase = model_phase_by_bin[-1]
    reference_magnitude = model_magnitude_by_bin[-1]

    print("\nmodel phase(bin) - phase(highest in-band bin)   [real channel in brackets]:")
    for label, model_phase, real_phase in zip(bin_labels[:-1], model_phase_by_bin[:-1], real_phase_by_bin[:-1]):
        model_delta = model_phase - reference_phase
        real_delta = real_phase - real_phase_by_bin[-1]
        print(f"  {label}: span={model_delta.max() - model_delta.min():.3f} rad "
              f"[{real_delta.max() - real_delta.min():.3f}], "
              f"at f_max={model_delta[-1]:+.3f} rad [{real_delta[-1]:+.3f}]")

    print("\nmodel extrapolation below training range (rel. highest in-band bin):")
    for label, phase in zip(low_power_labels, low_power_phase):
        delta = phase - reference_phase
        print(f"  {label}: span={delta.max() - delta.min():.3f} rad, at f_max={delta[-1]:+.3f} rad")

    fig, axes = plt.subplots(3, 1, figsize=(8, 12))
    in_band_ax, extrapolation_ax, magnitude_ax = axes

    for bin_index, (label, model_phase, real_phase) in enumerate(
            zip(bin_labels, model_phase_by_bin, real_phase_by_bin)):
        in_band_ax.plot(frequencies_mhz, model_phase - reference_phase, lw=1.5, color=f"C{bin_index}",
                        label=f"model {label}")
        in_band_ax.plot(frequencies_mhz, real_phase - real_phase_by_bin[-1], lw=1.2, ls="--",
                        color=f"C{bin_index}", label=f"real {label}")
    in_band_ax.set_title(f"{run_id}: in-band phase by drive power (rel. highest-power bin)")
    in_band_ax.set_ylabel("Δ∠H (rad)")

    extrapolation_ax.plot(frequencies_mhz, model_phase_by_bin[0] - reference_phase, lw=1.5, color="k",
                          label=f"model {bin_labels[0]} (lowest in-band)")
    for offset, (label, phase) in enumerate(zip(low_power_labels, low_power_phase)):
        extrapolation_ax.plot(frequencies_mhz, phase - reference_phase, lw=1.5, color=f"C{offset + 4}",
                              label=f"model {label}")
    extrapolation_ax.set_title("Model extrapolation below training range (rel. highest in-band bin)")
    extrapolation_ax.set_ylabel("Δ∠H (rad)")

    for bin_index, (label, magnitude) in enumerate(zip(bin_labels, model_magnitude_by_bin)):
        magnitude_ax.plot(frequencies_mhz, magnitude - reference_magnitude, lw=1.5, color=f"C{bin_index}",
                          label=f"model {label}")
    for offset, (label, magnitude) in enumerate(zip(low_power_labels, low_power_magnitude)):
        magnitude_ax.plot(frequencies_mhz, magnitude - reference_magnitude, lw=1.5, ls=":",
                          color=f"C{offset + 4}", label=f"model {label}")
    magnitude_ax.set_title("Model magnitude by drive power (rel. highest in-band bin)")
    magnitude_ax.set_ylabel("Δ|H| (dB)")

    for ax in axes:
        ax.set_xlabel("Frequency (MHz)")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(PLOT_PATH / "channel_model_power_dependence.png", dpi=130, bbox_inches="tight")
    fig.savefig(PLOT_PATH / "channel_model_power_dependence.svg", format="svg", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
