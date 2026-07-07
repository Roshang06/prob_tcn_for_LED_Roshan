"""
channel_power_dependence.py — measure how the REAL channel's H(f) depends on the
per-burst drive power, using the random power scaling U(POWER_MIN, POWER_MAX)
baked into a sweep dataset, plus a block-wise intra-dataset drift check.

Motivation: the E/D chain modulates at symbol power ~0.24 (unit QPSK through the
ortho IFFT) while sweep datasets are collected at 0.5-3.0, so the channel models
are extrapolating below their training range. This script quantifies how much
the physical channel's phase/magnitude actually move with drive power, i.e. how
wrong that extrapolation can be.

Edit CONFIG, then:
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

# ─── CONFIG ─────────────────────────────────────────────────────────────────────
DATASET_PATH   = "data/sweeps/dc0.05A_fmin300000_fmax7.6e+06_20260703_1631.zarr"
N_POWER_BINS   = 4        # quantile bins over per-burst sent symbol power
N_DRIFT_BLOCKS = 3        # evenly spaced burst blocks for the drift check
PLOT_PATH      = HERE.parent / "data/plots"
# ────────────────────────────────────────────────────────────────────────────────


def load_ratio(dataset_path):
    """Per-burst per-carrier R/S on the active subcarriers, plus power and freqs."""
    root = zarr.open_group(HERE.parent / dataset_path, mode="r")
    attrs = dict(root.attrs)
    ks = np.array(attrs["active_carrier_indices"])
    offset = int(attrs["preamble_length"]) + int(attrs["cyclic_prefix_length"])
    x = root["sent_burst"][:, offset:].astype(np.float64)
    y = root["received_burst"][:, offset:].astype(np.float64)
    S = np.fft.fft(x, norm="ortho", axis=1)[:, ks]
    R = np.fft.fft(y, norm="ortho", axis=1)[:, ks]
    spacing = float(attrs["f_min_hz"]) / ks[0]
    return R / S, (x ** 2).mean(axis=1), ks * spacing / 1e6


def demean_linear(phi, f):
    return phi - np.polyval(np.polyfit(f, phi, 1), f)


def main():
    PLOT_PATH.mkdir(parents=True, exist_ok=True)
    ratio, power, f_mhz = load_ratio(DATASET_PATH)

    # power-binned channel estimates, referenced to the highest-power bin
    edges = np.quantile(power, np.linspace(0, 1, N_POWER_BINS + 1))
    print(f"per-burst power quantile edges: {np.round(edges, 2)}")
    H_bins, labels = [], []
    for i in range(N_POWER_BINS):
        m = (power >= edges[i]) & (power <= edges[i + 1])
        H_bins.append(np.mean(ratio[m], axis=0))
        labels.append(f"P∈[{edges[i]:.2f},{edges[i + 1]:.2f}] (n={m.sum()})")
    phi_bins = [np.unwrap(np.angle(H)) for H in H_bins]
    mag_bins = [20 * np.log10(np.abs(H)) for H in H_bins]

    print("\nphase(power bin) − phase(highest-power bin):")
    for lab, ph in zip(labels[:-1], phi_bins[:-1]):
        d = ph - phi_bins[-1]
        print(f"  {lab}: span={d.max() - d.min():.3f} rad, "
              f"linear-removed span={np.ptp(demean_linear(d, f_mhz)):.3f} rad, "
              f"at f_max={d[-1]:+.3f} rad")

    # intra-dataset drift: H per burst block vs the first block
    nb = ratio.shape[0]
    width = nb // (2 * N_DRIFT_BLOCKS - 1)  # blocks with gaps between them
    blocks = [(2 * i * width, 2 * i * width + width) for i in range(N_DRIFT_BLOCKS)]
    phi_blk = [np.unwrap(np.angle(np.mean(ratio[a:b], axis=0))) for a, b in blocks]
    print("\nintra-dataset drift (block phase − first block):")
    for (a, b), ph in zip(blocks[1:], phi_blk[1:]):
        d = ph - phi_blk[0]
        print(f"  bursts {a}-{b}: span={d.max() - d.min():.3f} rad, "
              f"linear-removed span={np.ptp(demean_linear(d, f_mhz)):.3f} rad")

    fig, axs = plt.subplots(3, 1, figsize=(8, 11))

    for lab, ph in zip(labels, phi_bins):
        axs[0].plot(f_mhz, ph - phi_bins[-1], lw=1.4, label=lab)
    axs[0].set_title("Channel phase by drive power (rel. to highest-power bin)")
    axs[0].set_ylabel("Δ∠H (rad)")

    for lab, mg in zip(labels, mag_bins):
        axs[1].plot(f_mhz, mg - mag_bins[-1], lw=1.4, label=lab)
    axs[1].set_title("Channel magnitude by drive power (rel. to highest-power bin)")
    axs[1].set_ylabel("Δ|H| (dB)")

    for (a, b), ph in zip(blocks, phi_blk):
        axs[2].plot(f_mhz, ph - phi_blk[0], lw=1.4, label=f"bursts {a}-{b}")
    axs[2].set_title("Intra-dataset drift: block phase − first block")
    axs[2].set_ylabel("Δ∠H (rad)")

    for ax in axs:
        ax.set_xlabel("Frequency (MHz)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(PLOT_PATH / "channel_power_dependence.png", dpi=130, bbox_inches="tight")
    fig.savefig(PLOT_PATH / "channel_power_dependence.svg", format="svg", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
