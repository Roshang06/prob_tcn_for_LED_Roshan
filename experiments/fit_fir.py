"""
Diagnose channel memory from a zarr dataset and recommend receptive-field tap counts.

Everything is computed from the IN-BAND channel (per-carrier H_k), which is the only part
the data actually constrains, then converted to taps at the dataset's own sample rate. The
broadband time-domain FIR's tap count is deliberately NOT reported: it is dominated by
out-of-band extrapolation and inflates at lower oversampling (see git history).

Prints:
  * channel memory (taps)        - in-band group-delay dispersion x fs
  * recommended channel-model RF - must reproduce the channel
  * recommended encoder/decoder RF - must invert it (longer; from the channel's dominant zero)
"""
import sys
from pathlib import Path

import numpy as np
import zarr
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

# ─── CONFIG ─────────────────────────────────────────────────────────────────────
# DATASET_PATH = "data/sweeps/dc0.122A_fmin300000_fmax1.299e+07_20260623_1752.zarr"
DATASET_PATH = "data/sweeps/dc0.052A_fmin300000_fmax7.6e+06_20260630_1743.zarr"
L            = 128    # FIR tap count used only to extract the channel's dominant zero
VAL_FRACTION = 0.2
RIDGE        = 1e-3
INV_ENERGY_LOSS = 1e-2  # tolerated tail-energy loss for the equalizer-length estimate (99%)
MARGIN       = 2.0      # safety factor applied to the raw tap counts for the RF recommendation
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
    ks = np.array(attrs["active_carrier_indices"])
    spacing_hz = float(attrs.get("f_min_hz", 300e3)) / ks[0]
    fs = x.shape[1] * spacing_hz
    return x.astype(np.float32), y.astype(np.float32), fs, ks


def build_design(frames, L):
    """Stack sliding-window Toeplitz rows from [N, T] array → (N*(T-L+1), L)."""
    parts = []
    for x in frames:
        w = np.lib.stride_tricks.sliding_window_view(x, L)[:, ::-1].copy()
        parts.append(w)
    return np.vstack(parts)


def inversion_taps(rho, e):
    """Equalizer length to invert a minimum-phase channel: the inverse decays as rho^n,
    so tail energy beyond N taps is e = rho^(2N) -> N = ln(e)/(2 ln rho). rho = dominant
    zero magnitude (<1 for a convergent causal inverse)."""
    if rho >= 1.0:
        return float("inf")
    return np.log(e) / (2 * np.log(rho))


def inband_channel(x, y, fs, ks):
    """Per-carrier in-band channel H_k = <Y_k,X_k>/<X_k,X_k> (LS over bursts), free of the
    out-of-band extrapolation that dominates a time-domain FIR. Returns H_k, the active
    frequencies, the soft-band-limited impulse response (for plotting), and the in-band
    group-delay dispersion in ns (the real memory an equaliser must undo)."""
    Nfft = x.shape[1]
    X = np.fft.rfft(x, axis=1)
    Y = np.fft.rfft(y, axis=1)
    Hk = np.zeros(X.shape[1], dtype=complex)
    Hk[ks] = (Y * np.conj(X)).sum(0)[ks] / np.maximum((np.abs(X) ** 2).sum(0)[ks], 1e-12)
    f_hz = np.fft.rfftfreq(Nfft, 1.0 / fs)
    binhz = fs / Nfft

    # group-delay span = -dphase/domega across the band; smooth phase with a low-order
    # polynomial first so per-bin noise on fine carrier grids doesn't inflate the gradient
    ph = np.unwrap(np.angle(Hk[ks]))
    ph_s = np.polyval(np.polyfit(f_hz[ks], ph, deg=min(6, len(ks) - 1)), f_hz[ks])
    gd_ns = -np.gradient(ph_s, 2 * np.pi * binhz) * 1e9
    gd_span = float(gd_ns.max() - gd_ns.min())

    # soft-tapered band-limited impulse response (for the plot only)
    win = np.zeros_like(Hk, dtype=float)
    k0, k1 = int(ks[0]), int(ks[-1])
    win[k0:k1 + 1] = 1.0
    n_edge = max(1, int(0.08 * (k1 - k0)))
    ramp = 0.5 * (1 - np.cos(np.linspace(0, np.pi, n_edge)))
    win[k0:k0 + n_edge] *= ramp
    win[k1 - n_edge + 1:k1 + 1] *= ramp[::-1]
    h_bl = np.fft.fftshift(np.fft.irfft(Hk * win, n=Nfft))

    nlo = max(1, len(ks) // 5)
    lo = float(np.abs(Hk[ks[:nlo]]).mean())
    hi = float(np.abs(Hk[ks[-nlo:]]).mean())
    return dict(Hk=Hk, f_hz=f_hz, h_bl=h_bl, gd_span_ns=gd_span, lo=lo, hi=hi)


def channel_zero(x, y):
    """Dominant zero magnitude rho of the channel main lobe, and the linear-fit RRMSE.
    A ridge FIR is fit then TRUNCATED to the real channel memory before taking roots, so
    the out-of-band-extrapolation tail cannot inflate rho."""
    n_val = max(1, int(len(x) * VAL_FRACTION))
    X_tr = build_design(x[:-n_val], L)
    t_tr = np.concatenate([row[L - 1:] for row in y[:-n_val]])
    A = X_tr.T @ X_tr + RIDGE * np.eye(L)
    h = np.linalg.solve(A.astype(np.float64), (X_tr.T @ t_tr).astype(np.float64))
    X_val = build_design(x[-n_val:], L)
    t_val = np.concatenate([row[L - 1:] for row in y[-n_val:]])
    rrmse = float(np.sqrt(np.mean((t_val - X_val @ h) ** 2) / np.mean(t_val ** 2)) * 100)
    return h, rrmse


def main():
    PLOT_PATH.mkdir(parents=True, exist_ok=True)
    x, y, fs, ks = load(DATASET_PATH)
    Ts_ns = 1e9 / fs
    Nfft = x.shape[1]
    f_min, f_max = ks[0] * fs / Nfft, ks[-1] * fs / Nfft

    ib = inband_channel(x, y, fs, ks)
    rolloff = ib["hi"] / ib["lo"]

    # channel memory: in-band dispersion converted to taps at THIS sample rate
    channel_taps = max(1, int(np.ceil(ib["gd_span_ns"] / Ts_ns)))

    # equalizer length: invert the channel's dominant zero (from the truncated main lobe)
    h, rrmse = channel_zero(x, y)
    roots = np.roots(h[:max(channel_taps, 2)])
    rho = float(np.abs(roots).max()) if roots.size else 0.0
    eq_taps = int(np.ceil(inversion_taps(rho, INV_ENERGY_LOSS))) if rho < 1.0 else None

    rf_channel = int(np.ceil(channel_taps * MARGIN))
    rf_ed = int(np.ceil(eq_taps * MARGIN)) if eq_taps else None

    # full-band broadband FIR, for comparison only: its tap count is inflated/biased by
    # out-of-band extrapolation (the data does not constrain |f| outside the passband)
    h_energy = np.cumsum(h ** 2) / np.sum(h ** 2)
    fb_95 = int(np.searchsorted(h_energy, 0.95) + 1)
    fb_99 = int(np.searchsorted(h_energy, 0.99) + 1)
    H_full = np.fft.rfft(h, n=Nfft)

    print(f"dataset      : {Path(DATASET_PATH).name}")
    print(f"sample rate  : {fs / 1e6:.1f} MHz   (Ts = {Ts_ns:.2f} ns/tap)")
    print(f"band         : {f_min / 1e6:.2f} - {f_max / 1e6:.2f} MHz")
    print(f"in-band |H|  : {ib['lo']:.2f} -> {ib['hi']:.2f}  ({rolloff:.2f}x rolloff)"
          f"   linear-fit RRMSE {rrmse:.1f}%")
    print("-" * 60)
    print(f"channel memory (in-band)      : {channel_taps:4d} taps  ({ib['gd_span_ns']:.0f} ns)")
    print(f"full-band FIR (comparison)    : 95% energy {fb_95:2d} taps, 99% energy {fb_99:2d} taps  "
          f"(out-of-band-biased — see plot)")
    print(f"recommended channel-model RF  : {rf_channel:4d} taps   (memory x{MARGIN:g})")
    if rf_ed:
        print(f"recommended encoder/decoder RF: {rf_ed:4d} taps   "
              f"(equalizer {eq_taps} taps x{MARGIN:g}, channel zero rho={rho:.2f})")
    else:
        print("recommended encoder/decoder RF: channel non-minimum-phase (rho>=1) — "
              "needs bulk delay + anti-causal taps")

    # one figure: in-band data vs the full-band broadband FIR, in frequency and time
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4))
    axA.plot(ib["f_hz"][ks] / 1e6, 20 * np.log10(np.abs(ib["Hk"][ks]) + 1e-12),
             "C0.-", ms=3, label="in-band H_k (data)")
    axA.plot(ib["f_hz"] / 1e6, 20 * np.log10(np.abs(H_full) + 1e-12),
             "C3-", alpha=0.7, label="full-band FIR fit")
    axA.axvspan(f_min / 1e6, f_max / 1e6, color="C2", alpha=0.10, label="data passband")
    axA.set_xlabel("Frequency (MHz)")
    axA.set_ylabel("|H| (dB)")
    axA.set_title(f"Frequency: in-band data vs full-band FIR ({rolloff:.1f}x rolloff)")
    axA.grid(True, alpha=0.3)
    axA.legend(fontsize=8)

    peak = int(np.argmax(np.abs(ib["h_bl"])))
    tb = (np.arange(len(ib["h_bl"])) - peak) * Ts_ns
    axB.plot(tb, ib["h_bl"] / np.max(np.abs(ib["h_bl"])), "C1-", label="in-band IR (band-limited)")
    hpk = int(np.argmax(np.abs(h)))
    th = (np.arange(len(h)) - hpk) * Ts_ns
    axB.plot(th, h / np.max(np.abs(h)), "C3.-", ms=3, label=f"full-band FIR taps (99%@{fb_99})")
    axB.set_xlim(-channel_taps * Ts_ns * 4, channel_taps * Ts_ns * 4)
    axB.axvspan(-ib["gd_span_ns"] / 2, ib["gd_span_ns"] / 2, color="C2", alpha=0.15,
                label=f"in-band memory ~{channel_taps} taps")
    axB.set_xlabel("Lag (ns)")
    axB.set_ylabel("Normalized amplitude")
    axB.set_title("Impulse response (normalized)")
    axB.grid(True, alpha=0.3)
    axB.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOT_PATH / "fir_channel.png", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
