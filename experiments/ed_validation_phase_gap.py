"""
Diagnose the systematic phase error of hardware-validated encoder/decoder pairs by
separating what was already wrong in simulation from the sim-to-real gap.

For every run in an E/D validation experiment this computes the end-to-end per-carrier
response H_ed = mean(FFT(received) / FFT(sent)) from validation.zarr and regresses its
phase onto the raw channel phase. The coupling coefficient measures how much channel
curvature survived equalization (0 means the E/D fully equalized it, 1 means it ignored
it). For the best run per channel family it then:

    1. replays the exact validation sent bursts through encoder -> frozen channel model
       -> decoder (the training-time environment) to get the simulated phase error,
    2. evaluates the channel model at the encoded operating point against the sweep's
       training-distribution inputs,
    3. decomposes hardware EVM into systematic (mean per-carrier gain error) and random
       (scatter) parts, and estimates per-trial residual delay jitter from the linear
       phase of each trial.

Edit the configuration below, then:
    cd <repo_root>/experiments
    python ed_validation_phase_gap.py
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
from modules.models import TCN


# Configuration
SWEEP_PATH = "data/sweeps/dc0.05A_fmin300000_fmax7.6e+06_20260708_1822.zarr"
CHANNEL_EXP = "data/experiments/train_and_validate/amber_shore_channel_models_20260710_1638"
ED_EXP = "data/experiments/train_and_validate/amber_shore_encoder_decoder_20260711_0627"
VAL_EXP = "data/experiments/train_and_validate/amber_shore_ed_validation_20260711_0804"
CLIP_THRESHOLD = 3.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PLOT_PATH = HERE.parent / "data/plots"

ARCHITECTURE_KEYS = ("nlayers", "dilation_base", "kernel_size", "hidden_channels")


def read_jsonl(path):
    lines = Path(path).read_text().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def mean_response(sent, received, active_carrier_indices):
    '''Per-carrier transfer function averaged over trials: mean(FFT(received) / FFT(sent)).
    Returns the averaged response and the two spectra it was built from.'''
    sent_spectrum = np.fft.fft(sent.astype(np.float64), norm="ortho", axis=1)[:, active_carrier_indices]
    received_spectrum = np.fft.fft(received.astype(np.float64), norm="ortho", axis=1)[:, active_carrier_indices]

    average_response = np.mean(received_spectrum / sent_spectrum, axis=0)
    return average_response, sent_spectrum, received_spectrum


def load_channel_model(channel_exp_dir, run_id):
    selection = select_channel_models(channel_exp_dir, run_ids=[run_id])[0]
    adapter = MODEL_REGISTRY[selection["model"]].load(selection["params"], selection["checkpoint"], DEVICE)

    model = adapter.model if hasattr(adapter, "model") else adapter
    if hasattr(model, "parameters"):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    return model


def run_model(model, x):
    with torch.no_grad():
        output = model(x)
    return output[0] if isinstance(output, tuple) else output


def main():
    PLOT_PATH.mkdir(parents=True, exist_ok=True)

    sweep = zarr.open_group(HERE.parent / SWEEP_PATH, mode="r")
    attrs = dict(sweep.attrs)
    active_carrier_indices = np.array(attrs["active_carrier_indices"])
    cyclic_prefix_length = int(attrs["cyclic_prefix_length"])
    symbol_offset = int(attrs["preamble_length"]) + cyclic_prefix_length

    sweep_sent = sweep["sent_burst"][:, symbol_offset:].astype(np.float64)
    sweep_received = sweep["received_burst"][:, symbol_offset:].astype(np.float64)
    symbol_length = sweep_sent.shape[1]

    subcarrier_spacing_hz = float(attrs["f_min_hz"]) / active_carrier_indices[0]
    frequencies_mhz = active_carrier_indices * subcarrier_spacing_hz / 1e6
    sample_rate_hz = symbol_length * subcarrier_spacing_hz

    raw_response, _, _ = mean_response(sweep_sent, sweep_received, active_carrier_indices)
    raw_phase = np.unwrap(np.angle(raw_response))

    validation = zarr.open_group(HERE.parent / VAL_EXP / "validation.zarr", mode="r")
    validation_rows = {row["run_id"]: row for row in read_jsonl(HERE.parent / VAL_EXP / "runs.jsonl")}
    ed_rows = {row["run_id"]: row for row in read_jsonl(HERE.parent / ED_EXP / "runs.jsonl")}

    # hardware end-to-end phase error for every validated run
    hardware_phase = {}
    for run_id in validation.group_keys():
        response, _, _ = mean_response(validation[run_id]["sent_time"][:],
                                       validation[run_id]["received_time"][:],
                                       active_carrier_indices)
        hardware_phase[run_id] = np.unwrap(np.angle(response))

    # regress each phase error onto [1, f, raw_phase]; the raw_phase coefficient reports
    # how much channel curvature survived equalization
    design_matrix = np.column_stack([np.ones_like(frequencies_mhz), frequencies_mhz, raw_phase])
    coupling_coefficients = np.array([
        np.linalg.lstsq(design_matrix, phase_error, rcond=None)[0][2]
        for phase_error in hardware_phase.values()
    ])
    print(f"{len(hardware_phase)} validated runs; coupling a to raw channel phase: "
          f"mean={coupling_coefficients.mean():.3f} std={coupling_coefficients.std():.3f} "
          f"range=({coupling_coefficients.min():.3f}, {coupling_coefficients.max():.3f})")

    # keep the lowest-EVM validated run for each distinct channel family
    best_runs_per_family = []
    seen_families = set()
    for row in sorted(validation_rows.values(), key=lambda row: row["evm_pct"]):
        if row["channel_form"] not in seen_families:
            best_runs_per_family.append(row)
            seen_families.add(row["channel_form"])

    results = {}
    for row in best_runs_per_family:
        run_id = row["run_id"]
        ed_run_id = row["model"]  # the validation row's "model" field holds the E/D grid run_id
        channel_form = row["channel_form"]

        architecture = {key: ed_rows[ed_run_id][key] for key in ARCHITECTURE_KEYS}
        encoder = TCN(**architecture).to(DEVICE)
        decoder = TCN(**architecture).to(DEVICE)

        checkpoint = torch.load(HERE.parent / ED_EXP / "runs" / ed_run_id / "model.pt",
                                map_location=DEVICE, weights_only=True)
        encoder.load_state_dict(checkpoint["encoder"])
        decoder.load_state_dict(checkpoint["decoder"])
        encoder.eval()
        decoder.eval()

        channel = load_channel_model(HERE.parent / CHANNEL_EXP, row["channel_run_id"])

        # replay the validation sent symbols through encoder -> channel -> decoder
        sent_symbol = validation[run_id]["sent_time"][:].astype(np.float32)
        symbol = torch.tensor(sent_symbol, device=DEVICE)
        burst = torch.hstack([symbol[:, -cyclic_prefix_length:], symbol])
        with torch.no_grad():
            encoded = encoder(burst).clamp(-CLIP_THRESHOLD, CLIP_THRESHOLD)
            decoded = decoder(run_model(channel, encoded))

        simulated_response, _, _ = mean_response(sent_symbol, decoded[:, -symbol_length:].cpu().numpy(),
                                                 active_carrier_indices)
        simulated_phase = np.unwrap(np.angle(simulated_response))

        # channel-model response at the encoded operating point vs the training inputs
        encoded_symbol = encoded[:, -symbol_length:].cpu().numpy()
        encoded_output = run_model(channel, encoded)[:, -symbol_length:].cpu().numpy()
        operating_point_response, _, _ = mean_response(encoded_symbol, encoded_output, active_carrier_indices)

        training_input = torch.tensor(
            sweep_sent[:256, -(symbol_length + cyclic_prefix_length):].astype(np.float32), device=DEVICE)
        training_output = run_model(channel, training_input)[:, -symbol_length:].cpu().numpy()
        training_response, _, _ = mean_response(sweep_sent[:256], training_output, active_carrier_indices)

        # split hardware EVM into a systematic (mean gain error) and a random (scatter) part
        mean_gain, sent_spectrum, received_spectrum = mean_response(
            validation[run_id]["sent_time"][:], validation[run_id]["received_time"][:], active_carrier_indices)
        signal_power = np.mean(np.abs(sent_spectrum) ** 2)
        systematic_evm = np.sqrt(np.mean(np.abs((mean_gain[None, :] - 1) * sent_spectrum) ** 2) / signal_power) * 100
        random_evm = np.sqrt(np.mean(np.abs(received_spectrum - mean_gain[None, :] * sent_spectrum) ** 2) / signal_power) * 100

        # per-trial residual delay from the linear phase left after removing the mean gain
        delay_jitter_ns = np.array([
            -np.polyfit(frequencies_mhz * 1e6,
                        np.unwrap(np.angle(received_spectrum[trial] / (sent_spectrum[trial] * mean_gain))), 1)[0]
            / (2 * np.pi) * 1e9
            for trial in range(received_spectrum.shape[0])
        ])
        phase_scatter = np.std(np.angle(received_spectrum / (sent_spectrum * mean_gain[None, :])), axis=0)
        encoded_power = (encoded[:, -symbol_length:] ** 2).mean(dim=1).cpu().numpy()

        operating_point_phase_gap = (np.unwrap(np.angle(operating_point_response))
                                     - np.unwrap(np.angle(training_response)))
        results[run_id] = dict(form=channel_form, phi_sim=simulated_phase,
                               dphi_op=operating_point_phase_gap, phase_scatter=phase_scatter)

        print(f"\n{run_id} ({channel_form}, channel={row['channel_run_id']})")
        print(f"  EVM: sim={ed_rows[ed_run_id]['evm_pct']:.2f}%  hw={row['evm_pct']:.2f}% "
              f"= systematic {systematic_evm:.2f}% + random {random_evm:.2f}%")
        print(f"  phase-err span: sim={np.ptp(simulated_phase):.3f} rad  hw={np.ptp(hardware_phase[run_id]):.3f} rad")

        sweep_power = (sweep_sent ** 2).mean(axis=1)
        print(f"  encoder-out symbol power: {encoded_power.mean():.2f} "
              f"(sweep bursts span {sweep_power.min():.2f}-{sweep_power.max():.2f})")
        print(f"  per-trial delay jitter: std={delay_jitter_ns.std():.2f} ns "
              f"({delay_jitter_ns.std() * sample_rate_hz / 1e9:.3f} samples @ {sample_rate_hz / 1e6:.1f} MS/s)")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    phase_vs_channel_ax = axes[0, 0]
    for phase_error in hardware_phase.values():
        phase_vs_channel_ax.plot(frequencies_mhz, phase_error - phase_error.mean(),
                                 color="C0", alpha=0.15, lw=0.8)
    mean_phase_error = np.mean(list(hardware_phase.values()), axis=0)
    phase_vs_channel_ax.plot(frequencies_mhz, mean_phase_error - mean_phase_error.mean(),
                             "C0", lw=2, label="hw E/D phase err (mean)")
    phase_vs_channel_ax.plot(frequencies_mhz, raw_phase - raw_phase.mean(),
                             "k--", lw=1.5, label="raw channel phase")
    phase_vs_channel_ax.set_title("Hardware E/D phase error vs channel phase (demeaned)")
    phase_vs_channel_ax.set_ylabel("phase (rad)")

    sim_vs_hardware_ax = axes[0, 1]
    for run_id, result in results.items():
        sim_vs_hardware_ax.plot(frequencies_mhz, hardware_phase[run_id] - hardware_phase[run_id].mean(),
                                lw=1.5, label=f"hw {result['form']}")
        sim_vs_hardware_ax.plot(frequencies_mhz, result["phi_sim"] - result["phi_sim"].mean(),
                                lw=1.5, ls="--", label=f"sim {result['form']}")
    sim_vs_hardware_ax.set_title("Best run per family: hardware vs sim-replay phase error")
    sim_vs_hardware_ax.set_ylabel("phase (rad)")

    operating_point_ax = axes[1, 0]
    for result in results.values():
        demeaned_gap = result["dphi_op"] - result["dphi_op"].mean()
        operating_point_ax.plot(frequencies_mhz, demeaned_gap, lw=1.5, label=result["form"])
    operating_point_ax.set_title("Channel-model phase: encoded operating point vs training inputs")
    operating_point_ax.set_ylabel("phase gap (rad)")

    scatter_ax = axes[1, 1]
    for result in results.values():
        scatter_ax.plot(frequencies_mhz, result["phase_scatter"], lw=1.5, label=result["form"])
    scatter_ax.set_title("Per-carrier phase scatter across trials (sync jitter + noise)")
    scatter_ax.set_ylabel("std (rad)")

    for ax in axes.flat:
        ax.set_xlabel("Frequency (MHz)")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(PLOT_PATH / "ed_validation_phase_gap.png", dpi=130, bbox_inches="tight")
    fig.savefig(PLOT_PATH / "ed_validation_phase_gap.svg", format="svg", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
