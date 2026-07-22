"""
Estimate the per-subcarrier channel H(f_k) from a zarr dataset and plot magnitude and
unwrapped phase (with a linear-delay fit / residual split) to separate LPF dispersion
from symbol-timing offset.

Edit the configuration at the top, then:
    cd <repo_root>/experiments
    python channel_response.py
"""
import sys
from pathlib import Path

import numpy as np
import zarr
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))


# Configuration
DATASET_PATH = "data/sweeps/dc0.05A_fmin300000_fmax7.6e+06_20260708_1822.zarr"
PLOT_PATH = HERE.parent / "data/plots"


def load_symbols(dataset_path):
    """Return (sent, received, active_carrier_indices, sample_rate_hz) with preamble + CP stripped."""
    root = zarr.open_group(HERE.parent / dataset_path, mode="r")
    attrs = dict(root.attrs)
    active_carrier_indices = np.array(attrs["active_carrier_indices"])
    cyclic_prefix_length = int(attrs["cyclic_prefix_length"])
    symbol_offset = int(attrs.get("preamble_length")) + cyclic_prefix_length

    sent = root["sent_burst"][:, symbol_offset:]
    received = root["received_burst"][:, symbol_offset:]
    subcarrier_spacing_hz = float(attrs.get("f_min_hz")) / active_carrier_indices[0]
    sample_rate_hz = sent.shape[1] * subcarrier_spacing_hz
    return sent.astype(np.float64), received.astype(np.float64), active_carrier_indices, sample_rate_hz


def main():
    PLOT_PATH.mkdir(parents=True, exist_ok=True)
    sent, received, active_carrier_indices, sample_rate_hz = load_symbols(DATASET_PATH)

    # per-frame LS channel estimate on the active subcarriers, then average
    sent_spectrum = np.fft.rfft(sent, norm="ortho", axis=1)[:, active_carrier_indices]
    received_spectrum = np.fft.rfft(received, norm="ortho", axis=1)[:, active_carrier_indices]
    response = np.mean(received_spectrum / sent_spectrum, axis=0)

    frequencies_khz = active_carrier_indices * (sample_rate_hz / sent.shape[1]) / 1e3
    magnitude_db = 10 * np.log10(np.abs(response))
    phase = np.unwrap(np.angle(response))

    # linear-delay fit: slope = bulk group delay (STO + bulk channel delay)
    linear_coeffs = np.polyfit(frequencies_khz, phase, 1)
    linear_phase = np.polyval(linear_coeffs, frequencies_khz)
    residual = phase - linear_phase

    # slope is rad per kHz -> group delay tau = -slope / (2*pi*df_Hz)
    bulk_delay_ns = -linear_coeffs[0] / (2 * np.pi * 1e3) * 1e9
    print(f"Bulk group delay (linear phase fit): {bulk_delay_ns:.2f} ns "
          f"({bulk_delay_ns * sample_rate_hz / 1e9:.2f} samples)")
    print(f"Max dispersive residual: {np.abs(residual).max():.3f} rad")

    # frequency-resolved group delay tau_g(f) = -dphase/domega
    frequencies_hz = active_carrier_indices * (sample_rate_hz / sent.shape[1])
    group_delay_ns = -np.gradient(phase, frequencies_hz) / (2 * np.pi) * 1e9

    # -3 dB bandwidth relative to the low-frequency value
    reference_db = magnitude_db[0]
    below_3db = np.where(magnitude_db <= reference_db - 3)[0]
    f_3db_khz = frequencies_khz[below_3db[0]] if below_3db.size else float("nan")
    print(f"f_3dB (rel. to lowest carrier): {f_3db_khz / 1e3:.2f} MHz")

    fig, axes = plt.subplots(3, 1, figsize=(7, 9), sharex=True)

    axes[0].plot(frequencies_khz, magnitude_db, "C0-")
    axes[0].set_ylabel("|H(f)| (dB)")
    axes[0].set_title("Channel Magnitude Response (LPF roll-off)")
    axes[0].grid(True)

    axes[1].plot(frequencies_khz, phase, "C1-", label="unwrapped phase")
    axes[1].plot(frequencies_khz, linear_phase, "k--", alpha=0.7,
                 label=f"linear fit (τ = {bulk_delay_ns:.1f} ns)")
    axes[1].set_ylabel("∠H(f) (rad)")
    axes[1].set_title("Channel Phase Response")
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(frequencies_khz, residual, "C3-")
    axes[2].axhline(0, color="gray", linewidth=0.8)
    axes[2].set_ylabel("Residual phase (rad)")
    axes[2].set_xlabel("Frequency (kHz)")
    axes[2].set_title("Dispersive Residual (phase - linear-delay fit)")
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(PLOT_PATH / "channel_response.png", bbox_inches="tight")
    plt.savefig(PLOT_PATH / "channel_response.svg", format="svg", bbox_inches="tight")

    # Bode-style log-frequency view + frequency-resolved group delay
    fig2, bode_axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)

    bode_axes[0].semilogx(frequencies_khz, magnitude_db, "C0-")
    bode_axes[0].axhline(reference_db - 3, color="gray", linestyle="--", linewidth=0.8)
    if np.isfinite(f_3db_khz):
        bode_axes[0].axvline(f_3db_khz, color="C2", linestyle=":", alpha=0.8,
                             label=f"f_3dB = {f_3db_khz / 1e3:.1f} MHz")
        bode_axes[0].legend()
    bode_axes[0].set_ylabel("|H(f)| (dB)")
    bode_axes[0].set_title("Channel Magnitude (log frequency / Bode)")
    bode_axes[0].grid(True, which="both")

    bode_axes[1].semilogx(frequencies_khz, group_delay_ns, "C4-")
    bode_axes[1].set_ylabel("Group delay (ns)")
    bode_axes[1].set_xlabel("Frequency (kHz)")
    bode_axes[1].set_title("Group Delay  τ_g(f) = −dϕ/dω")
    bode_axes[1].grid(True, which="both")

    fig2.tight_layout()
    fig2.savefig(PLOT_PATH / "channel_response_bode.png", bbox_inches="tight")
    fig2.savefig(PLOT_PATH / "channel_response_bode.svg", format="svg", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
