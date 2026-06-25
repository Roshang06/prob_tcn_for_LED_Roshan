"""
fit a ridge FIR to a zarr dataset and diagnose channel memory length.
"""
import sys
from pathlib import Path

import numpy as np
import zarr
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

# ─── CONFIG ─────────────────────────────────────────────────────────────────────
DATASET_PATH = "data/sweeps/dc0.122A_fmin300000_fmax1.299e+07_20260623_1752.zarr"
L            = 128    # FIR tap count
VAL_FRACTION = 0.2
RIDGE        = 1e-3
INV_ENERGY_LOSS = 1e-2  # tolerated relative tail-energy loss for the equalizer estimate
PLOT_PATH    = HERE.parent / "data/plots"
# ────────────────────────────────────────────────────────────────────────────────


def load(dataset_path):
    root  = zarr.open_group(HERE.parent / dataset_path, mode="r")
    attrs = dict(root.attrs)
    if "sent_burst" in root:
        x = root["sent_burst"][:]
        y = root["received_burst"][:]
        offset = int(attrs.get("preamble_length", 0)) + int(attrs["cyclic_prefix_length"])
        x, y = x[:, offset:], y[:, offset:]
    else:
        x = root["sent_baseband"][:]
        y = root["received_baseband"][:]
    ks         = np.array(attrs["active_carrier_indices"])
    spacing_hz = float(attrs.get("f_min_hz", 300e3)) / ks[0]
    fs         = x.shape[1] * spacing_hz
    return x.astype(np.float32), y.astype(np.float32), fs


def build_design(frames, L):
    """Stack sliding-window Toeplitz rows from [N, T] array → (N*(T-L+1), L)."""
    parts = []
    for x in frames:
        w = np.lib.stride_tricks.sliding_window_view(x, L)[:, ::-1].copy()
        parts.append(w)
    return np.vstack(parts)


def inversion_taps(rho, e):
    """
    Estimate the equalizer length needed to invert a minimum-phase channel.

    The min-phase inverse decays as rho^n where rho = max|zero|, so the energy
    left in the tail beyond N taps is e = rho^(2N). Solving for N:

        N = ln(e) / (2 ln(rho))

    rho : dominant zero magnitude (must be < 1; >= 1 => non-min-phase, no
          convergent causal inverse).
    e   : tolerated relative tail-energy loss (e.g. 1e-2 for 99%).
    """
    if rho >= 1.0:
        return float("inf")
    return np.log(e) / (2 * np.log(rho))


def main():
    PLOT_PATH.mkdir(parents=True, exist_ok=True)
    x, y, fs = load(DATASET_PATH)
    Ts_ns = 1e9 / fs

    n_val = max(1, int(len(x) * VAL_FRACTION))
    x_tr, y_tr = x[:-n_val], y[:-n_val]
    x_val, y_val = x[-n_val:], y[-n_val:]

    X_tr = build_design(x_tr, L)
    t_tr = np.concatenate([row[L - 1:] for row in y_tr])

    # ridge least squares: h = (X^T X + λI)^{-1} X^T y
    A = X_tr.T @ X_tr + RIDGE * np.eye(L, dtype=np.float64)
    b = X_tr.T @ t_tr
    h = np.linalg.solve(A.astype(np.float64), b.astype(np.float64))

    # validation RRMSE
    X_val = build_design(x_val, L)
    t_val = np.concatenate([row[L - 1:] for row in y_val])
    y_hat = X_val @ h
    rrmse = np.sqrt(np.mean((t_val - y_hat) ** 2) / np.mean(t_val ** 2)) * 100

    energy     = h ** 2
    cum_energy = np.cumsum(energy) / energy.sum()
    lags_ns    = np.arange(L) * Ts_ns

    m95 = int(np.searchsorted(cum_energy, 0.95))
    m99 = int(np.searchsorted(cum_energy, 0.99))
    print(f"Val RRMSE : {rrmse:.3f}%")
    print(f"95% energy: tap {m95}  ({lags_ns[min(m95, L-1)]:.1f} ns)")
    print(f"99% energy: tap {m99}  ({lags_ns[min(m99, L-1)]:.1f} ns)")

    # minimum-phase test: roots (zeros) of the FIR polynomial H(z) = sum h[k] z^-k.
    # Truncate to the 99%-energy span first — the ridge-regularized tail is noise
    # and would otherwise spawn spurious zeros near the unit circle.
    h_sig = h[:max(m99 + 1, 2)]
    roots = np.roots(h_sig)            # zeros of H(z) in the z-plane
    max_radius = float(np.abs(roots).max()) if roots.size else 0.0
    is_min_phase = max_radius < 1.0
    n_outside = int((np.abs(roots) > 1.0).sum())
    print(f"FIR zeros: {roots.size} (from {h_sig.size} taps), "
          f"max |root| = {max_radius:.4f}")
    print(f"Minimum phase: {is_min_phase}  ({n_outside} zero(s) outside unit circle)")

    n_inv = inversion_taps(max_radius, INV_ENERGY_LOSS)
    if np.isfinite(n_inv):
        print(f"Equalizer taps for {(1 - INV_ENERGY_LOSS) * 100:.0f}% inversion "
              f"energy (rho={max_radius:.4f}): {n_inv:.1f} taps "
              f"({n_inv * Ts_ns:.1f} ns)")
    else:
        print("Non-minimum phase: no convergent causal inverse "
              "(equalizer needs bulk delay + anti-causal taps)")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.stem(lags_ns, h, markerfmt="C0o", linefmt="C0-", basefmt="k-")
    ax1.set_xlabel("Lag (ns)")
    ax1.set_ylabel("Tap weight")
    ax1.set_title(f"FIR Impulse Response  L={L},  ridge={RIDGE}")
    ax1.grid(True)

    ax2.plot(lags_ns, cum_energy, "C1-")
    ax2.axhline(0.95, color="gray", linestyle="--", linewidth=0.8)
    ax2.axhline(0.99, color="gray", linestyle=":",  linewidth=0.8)
    if m95 < L:
        ax2.axvline(lags_ns[m95], color="C2", linestyle="--", alpha=0.8,
                    label=f"95%: {lags_ns[m95]:.0f} ns (tap {m95})")
    if m99 < L:
        ax2.axvline(lags_ns[m99], color="C3", linestyle=":",  alpha=0.8,
                    label=f"99%: {lags_ns[m99]:.0f} ns (tap {m99})")
    ax2.set_xlabel("Lag (ns)")
    ax2.set_ylabel("Cumulative Energy Fraction")
    ax2.set_title("Cumulative FIR Energy")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(PLOT_PATH / "fir_analysis.png", bbox_inches="tight")
    plt.savefig(PLOT_PATH / "fir_analysis.svg", format="svg", bbox_inches="tight")

    # z-plane: FIR zeros vs the unit circle (inside = minimum phase)
    fig2, ax3 = plt.subplots(figsize=(5, 5))
    theta = np.linspace(0, 2 * np.pi, 400)
    ax3.plot(np.cos(theta), np.sin(theta), "k-", linewidth=0.8)
    inside  = np.abs(roots) < 1.0
    ax3.scatter(roots[inside].real,  roots[inside].imag,  marker="o",
                facecolors="none", edgecolors="C0", label="inside (min-phase)")
    ax3.scatter(roots[~inside].real, roots[~inside].imag, marker="x",
                color="C3", label="outside (non-min-phase)")
    ax3.set_xlabel("Real")
    ax3.set_ylabel("Imag")
    ax3.set_title(f"FIR Zeros (z-plane)  max|root|={max_radius:.3f}")
    ax3.set_aspect("equal")
    ax3.grid(True)
    ax3.legend()

    fig2.tight_layout()
    fig2.savefig(PLOT_PATH / "fir_zeros.png", bbox_inches="tight")
    fig2.savefig(PLOT_PATH / "fir_zeros.svg", format="svg", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
