"""
Measure how the REAL channel's H(f) depends on the per-burst drive power, using the
random power scaling U(POWER_MIN, POWER_MAX) baked into a sweep dataset, plus a
block-wise intra-dataset drift check.

Motivation: the E/D chain modulates at symbol power ~0.24 (unit QPSK through the
ortho IFFT) while sweep datasets are collected at 0.5 to 3.0, so the channel models
are extrapolating below their training range. This script quantifies how much the
physical channel's phase/magnitude actually move with drive power, i.e. how wrong
that extrapolation can be.

Edit the configuration below, then:
    cd <repo_root>/experiments
    python channel_power_dependence.py
"""
import sys
from pathlib import Path

import numpy as np
import zarr
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))


# Configuration
DATASET_PATH = "data/sweeps/dc0.05A_fmin300000_fmax7.6e+06_20260703_1631.zarr"
N_POWER_BINS = 4        # quantile bins over per-burst sent symbol power
N_DRIFT_BLOCKS = 3      # evenly spaced burst blocks for the drift check
PLOT_PATH = HERE.parent / "data/plots"


def load_ratio(dataset_path):
    """Per-burst per-carrier R/S on the active subcarriers, plus power and freqs."""
    root = zarr.open_group(HERE.parent / dataset_path, mode="r")
    attrs = dict(root.attrs)
    active_carrier_indices = np.array(attrs["active_carrier_indices"])
    symbol_offset = int(attrs["preamble_length"]) + int(attrs["cyclic_prefix_length"])

    sent = root["sent_burst"][:, symbol_offset:].astype(np.float64)
    received = root["received_burst"][:, symbol_offset:].astype(np.float64)
    sent_spectrum = np.fft.fft(sent, norm="ortho", axis=1)[:, active_carrier_indices]
    received_spectrum = np.fft.fft(received, norm="ortho", axis=1)[:, active_carrier_indices]

    subcarrier_spacing_hz = float(attrs["f_min_hz"]) / active_carrier_indices[0]
    burst_power = (sent ** 2).mean(axis=1)
    frequencies_mhz = active_carrier_indices * subcarrier_spacing_hz / 1e6
    return received_spectrum / sent_spectrum, burst_power, frequencies_mhz


def demean_linear(phase, frequencies):
    return phase - np.polyval(np.polyfit(frequencies, phase, 1), frequencies)


def main():
    PLOT_PATH.mkdir(parents=True, exist_ok=True)
    carrier_ratio, burst_power, frequencies_mhz = load_ratio(DATASET_PATH)

    # power-binned channel estimates, referenced to the highest-power bin
    power_bin_edges = np.quantile(burst_power, np.linspace(0, 1, N_POWER_BINS + 1))
    print(f"per-burst power quantile edges: {np.round(power_bin_edges, 2)}")

    response_by_bin = []
    bin_labels = []
    for bin_index in range(N_POWER_BINS):
        in_bin = (burst_power >= power_bin_edges[bin_index]) & (burst_power <= power_bin_edges[bin_index + 1])
        response_by_bin.append(np.mean(carrier_ratio[in_bin], axis=0))
        bin_labels.append(f"P∈[{power_bin_edges[bin_index]:.2f},{power_bin_edges[bin_index + 1]:.2f}] (n={in_bin.sum()})")

    phase_by_bin = [np.unwrap(np.angle(response)) for response in response_by_bin]
    magnitude_by_bin = [20 * np.log10(np.abs(response)) for response in response_by_bin]

    print("\nphase(power bin) - phase(highest-power bin):")
    for label, phase in zip(bin_labels[:-1], phase_by_bin[:-1]):
        delta = phase - phase_by_bin[-1]
        print(f"  {label}: span={delta.max() - delta.min():.3f} rad, "
              f"linear-removed span={np.ptp(demean_linear(delta, frequencies_mhz)):.3f} rad, "
              f"at f_max={delta[-1]:+.3f} rad")

    # intra-dataset drift: H per burst block vs the first block
    num_bursts = carrier_ratio.shape[0]
    block_width = num_bursts // (2 * N_DRIFT_BLOCKS - 1)  # leave gaps between blocks
    blocks = [(2 * i * block_width, 2 * i * block_width + block_width) for i in range(N_DRIFT_BLOCKS)]
    phase_by_block = [np.unwrap(np.angle(np.mean(carrier_ratio[start:stop], axis=0))) for start, stop in blocks]

    print("\nintra-dataset drift (block phase - first block):")
    for (start, stop), phase in zip(blocks[1:], phase_by_block[1:]):
        delta = phase - phase_by_block[0]
        print(f"  bursts {start}-{stop}: span={delta.max() - delta.min():.3f} rad, "
              f"linear-removed span={np.ptp(demean_linear(delta, frequencies_mhz)):.3f} rad")

    fig, axes = plt.subplots(3, 1, figsize=(8, 11))
    phase_ax, magnitude_ax, drift_ax = axes

    for label, phase in zip(bin_labels, phase_by_bin):
        phase_ax.plot(frequencies_mhz, phase - phase_by_bin[-1], lw=1.4, label=label)
    phase_ax.set_title("Channel phase by drive power (rel. to highest-power bin)")
    phase_ax.set_ylabel("Δ∠H (rad)")

    for label, magnitude in zip(bin_labels, magnitude_by_bin):
        magnitude_ax.plot(frequencies_mhz, magnitude - magnitude_by_bin[-1], lw=1.4, label=label)
    magnitude_ax.set_title("Channel magnitude by drive power (rel. to highest-power bin)")
    magnitude_ax.set_ylabel("Δ|H| (dB)")

    for (start, stop), phase in zip(blocks, phase_by_block):
        drift_ax.plot(frequencies_mhz, phase - phase_by_block[0], lw=1.4, label=f"bursts {start}-{stop}")
    drift_ax.set_title("Intra-dataset drift: block phase - first block")
    drift_ax.set_ylabel("Δ∠H (rad)")

    for ax in axes:
        ax.set_xlabel("Frequency (MHz)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(PLOT_PATH / "channel_power_dependence.png", dpi=130, bbox_inches="tight")
    fig.savefig(PLOT_PATH / "channel_power_dependence.svg", format="svg", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
