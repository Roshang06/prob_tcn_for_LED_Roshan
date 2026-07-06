"""
channel_response.py — estimate the per-subcarrier channel H(f_k) from a zarr
dataset and plot magnitude + unwrapped phase (with a linear-delay fit / residual
split) to separate LPF dispersion from symbol-timing offset.

Edit CONFIG at the top, then:
    cd <repo_root>/experimentsp
    python channel_response.py
"""
import sys
from pathlib import Path

import numpy as np
import zarr
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

# ─── CONFIG ─────────────────────────────────────────────────────────────────────
DATASET_PATH = "data/sweeps/dc0.122A_fmin300000_fmax1.3e+07_20260625_1419.zarr"
PLOT_PATH    = HERE.parent / "data/plots"
# ────────────────────────────────────────────────────────────────────────────────


def load_symbols(dataset_path):
    """Return (sent_sym, recv_sym, ks, fs) with preamble + CP stripped."""
    root  = zarr.open_group(HERE.parent / dataset_path, mode="r")
    attrs = dict(root.attrs)
    ks    = np.array(attrs["active_carrier_indices"])
    cp    = int(attrs["cyclic_prefix_length"])
    offset = int(attrs.get("preamble_length")) + cp
    x = root["sent_burst"][:, offset:]
    y = root["received_burst"][:, offset:]
    spacing_hz = float(attrs.get("f_min_hz")) / ks[0]
    fs = x.shape[1] * spacing_hz
    return x.astype(np.float64), y.astype(np.float64), ks, fs


def main():
    PLOT_PATH.mkdir(parents=True, exist_ok=True)
    x, y, ks, fs = load_symbols(DATASET_PATH)

    # per-frame LS channel estimate on the active subcarriers, then average
    S = np.fft.rfft(x, norm="ortho", axis=1)[:, ks]
    R = np.fft.rfft(y, norm="ortho", axis=1)[:, ks]
    H = np.mean(R / S, axis=0)

    f_khz = ks * (fs / x.shape[1]) / 1e3
    mag_db = 20 * np.log10(np.abs(H))
    phase  = np.unwrap(np.angle(H))

    # linear-delay fit: slope = bulk group delay (STO + bulk channel delay)
    coeffs    = np.polyfit(f_khz, phase, 1)
    phase_lin = np.polyval(coeffs, f_khz)
    residual  = phase - phase_lin
    # slope is rad per kHz -> group delay tau = -slope / (2*pi*df_Hz)
    tau_ns = -coeffs[0] / (2 * np.pi * 1e3) * 1e9
    print(f"Bulk group delay (linear phase fit): {tau_ns:.2f} ns "
          f"({tau_ns * fs / 1e9:.2f} samples)")
    print(f"Max dispersive residual: {np.abs(residual).max():.3f} rad")

    # frequency-resolved group delay tau_g(f) = -dphase/domega
    f_hz = ks * (fs / x.shape[1])
    group_delay_ns = -np.gradient(phase, f_hz) / (2 * np.pi) * 1e9

    # -3 dB bandwidth relative to the low-frequency value
    ref_db = mag_db[0]
    below  = np.where(mag_db <= ref_db - 3)[0]
    f3db_khz = f_khz[below[0]] if below.size else float("nan")
    print(f"f_3dB (rel. to lowest carrier): {f3db_khz / 1e3:.2f} MHz")

    fig, axs = plt.subplots(3, 1, figsize=(7, 9), sharex=True)

    axs[0].plot(f_khz, mag_db, "C0-")
    axs[0].set_ylabel("|H(f)| (dB)")
    axs[0].set_title("Channel Magnitude Response (LPF roll-off)")
    axs[0].grid(True)

    axs[1].plot(f_khz, phase, "C1-", label="unwrapped phase")
    axs[1].plot(f_khz, phase_lin, "k--", alpha=0.7,
                label=f"linear fit (τ = {tau_ns:.1f} ns)")
    axs[1].set_ylabel("∠H(f) (rad)")
    axs[1].set_title("Channel Phase Response")
    axs[1].legend()
    axs[1].grid(True)

    axs[2].plot(f_khz, residual, "C3-")
    axs[2].axhline(0, color="gray", linewidth=0.8)
    axs[2].set_ylabel("Residual phase (rad)")
    axs[2].set_xlabel("Frequency (kHz)")
    axs[2].set_title("Dispersive Residual (phase − linear-delay fit)")
    axs[2].grid(True)

    plt.tight_layout()
    plt.savefig(PLOT_PATH / "channel_response.png", bbox_inches="tight")
    plt.savefig(PLOT_PATH / "channel_response.svg", format="svg", bbox_inches="tight")

    # Bode-style log-frequency view + frequency-resolved group delay
    fig2, axs2 = plt.subplots(2, 1, figsize=(7, 6), sharex=True)

    axs2[0].semilogx(f_khz, mag_db, "C0-")
    axs2[0].axhline(ref_db - 3, color="gray", linestyle="--", linewidth=0.8)
    if np.isfinite(f3db_khz):
        axs2[0].axvline(f3db_khz, color="C2", linestyle=":", alpha=0.8,
                        label=f"f_3dB = {f3db_khz / 1e3:.1f} MHz")
        axs2[0].legend()
    axs2[0].set_ylabel("|H(f)| (dB)")
    axs2[0].set_title("Channel Magnitude (log frequency / Bode)")
    axs2[0].grid(True, which="both")

    axs2[1].semilogx(f_khz, group_delay_ns, "C4-")
    axs2[1].set_ylabel("Group delay (ns)")
    axs2[1].set_xlabel("Frequency (kHz)")
    axs2[1].set_title("Group Delay  τ_g(f) = −dϕ/dω")
    axs2[1].grid(True, which="both")

    fig2.tight_layout()
    fig2.savefig(PLOT_PATH / "channel_response_bode.png", bbox_inches="tight")
    fig2.savefig(PLOT_PATH / "channel_response_bode.svg", format="svg", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
