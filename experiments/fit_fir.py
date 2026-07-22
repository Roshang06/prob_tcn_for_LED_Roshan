"""
Diagnose channel memory from a zarr dataset and recommend receptive-field tap counts.

Everything is computed from the IN-BAND channel (per-carrier H_k), which is the only part
the data actually constrains, then converted to taps at the dataset's own sample rate. The
broadband time-domain FIR's tap count is deliberately NOT reported: it is dominated by
out-of-band extrapolation and inflates at lower oversampling (see git history).

Prints:
    * channel memory (taps)          in-band group-delay dispersion x fs
    * recommended channel-model RF   must reproduce the channel
    * recommended encoder/decoder RF must invert it (longer; from the channel's dominant zero)
"""
import sys
from pathlib import Path

import numpy as np
import zarr
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))


# Configuration
# DATASET_PATH = "data/sweeps/dc0.122A_fmin300000_fmax1.299e+07_20260623_1752.zarr"
DATASET_PATH = "data/sweeps/dc0.052A_fmin300000_fmax7.6e+06_20260630_1743.zarr"
FIR_TAPS = 128           # FIR tap count used only to extract the channel's dominant zero
VAL_FRACTION = 0.2
RIDGE = 1e-3
INV_ENERGY_LOSS = 1e-2   # tolerated tail-energy loss for the equalizer-length estimate (99%)
MARGIN = 2.0             # safety factor applied to the raw tap counts for the RF recommendation
PLOT_PATH = HERE.parent / "data/plots"


def load(dataset_path):
    root = zarr.open_group(HERE.parent / dataset_path, mode="r")
    attrs = dict(root.attrs)

    if "sent_burst" in root:
        sent = root["sent_burst"][:]
        received = root["received_burst"][:]
        symbol_offset = int(attrs.get("preamble_length", 0)) + int(attrs["cyclic_prefix_length"])
        sent, received = sent[:, symbol_offset:], received[:, symbol_offset:]
    else:
        sent = root["sent_baseband"][:]
        received = root["received_baseband"][:]

    active_carrier_indices = np.array(attrs["active_carrier_indices"])
    subcarrier_spacing_hz = float(attrs.get("f_min_hz", 300e3)) / active_carrier_indices[0]
    sample_rate_hz = sent.shape[1] * subcarrier_spacing_hz
    return sent.astype(np.float32), received.astype(np.float32), sample_rate_hz, active_carrier_indices


def build_design(frames, tap_count):
    """Stack sliding-window Toeplitz rows from [N, T] array into (N*(T-tap_count+1), tap_count)."""
    windows = []
    for frame in frames:
        window = np.lib.stride_tricks.sliding_window_view(frame, tap_count)[:, ::-1].copy()
        windows.append(window)
    return np.vstack(windows)


def inversion_taps(zero_magnitude, energy_loss):
    """Equalizer length to invert a minimum-phase channel: the inverse decays as rho^n,
    so tail energy beyond N taps is e = rho^(2N) -> N = ln(e)/(2 ln rho). rho = dominant
    zero magnitude (<1 for a convergent causal inverse)."""
    if zero_magnitude >= 1.0:
        return float("inf")
    return np.log(energy_loss) / (2 * np.log(zero_magnitude))


def inband_channel(sent, received, sample_rate_hz, active_carrier_indices):
    """Per-carrier in-band channel H_k = <Y_k,X_k>/<X_k,X_k> (LS over bursts), free of the
    out-of-band extrapolation that dominates a time-domain FIR. Returns H_k, the active
    frequencies, the soft-band-limited impulse response (for plotting), and the in-band
    group-delay dispersion in ns (the real memory an equaliser must undo)."""
    fft_length = sent.shape[1]
    sent_spectrum = np.fft.rfft(sent, axis=1)
    received_spectrum = np.fft.rfft(received, axis=1)

    channel_response = np.zeros(sent_spectrum.shape[1], dtype=complex)
    channel_response[active_carrier_indices] = (
        (received_spectrum * np.conj(sent_spectrum)).sum(0)[active_carrier_indices]
        / np.maximum((np.abs(sent_spectrum) ** 2).sum(0)[active_carrier_indices], 1e-12))

    frequencies_hz = np.fft.rfftfreq(fft_length, 1.0 / sample_rate_hz)
    bin_spacing_hz = sample_rate_hz / fft_length

    # group-delay span = -dphase/domega across the band; smooth phase with a low-order
    # polynomial first so per-bin noise on fine carrier grids doesn't inflate the gradient
    phase = np.unwrap(np.angle(channel_response[active_carrier_indices]))
    smoothed_phase = np.polyval(
        np.polyfit(frequencies_hz[active_carrier_indices], phase, deg=min(6, len(active_carrier_indices) - 1)),
        frequencies_hz[active_carrier_indices])
    group_delay_ns = -np.gradient(smoothed_phase, 2 * np.pi * bin_spacing_hz) * 1e9
    group_delay_span_ns = float(group_delay_ns.max() - group_delay_ns.min())

    # soft-tapered band-limited impulse response (for the plot only)
    window = np.zeros_like(channel_response, dtype=float)
    first_carrier, last_carrier = int(active_carrier_indices[0]), int(active_carrier_indices[-1])
    window[first_carrier:last_carrier + 1] = 1.0
    edge_width = max(1, int(0.08 * (last_carrier - first_carrier)))
    ramp = 0.5 * (1 - np.cos(np.linspace(0, np.pi, edge_width)))
    window[first_carrier:first_carrier + edge_width] *= ramp
    window[last_carrier - edge_width + 1:last_carrier + 1] *= ramp[::-1]
    band_limited_ir = np.fft.fftshift(np.fft.irfft(channel_response * window, n=fft_length))

    edge_carrier_count = max(1, len(active_carrier_indices) // 5)
    low_band_gain = float(np.abs(channel_response[active_carrier_indices[:edge_carrier_count]]).mean())
    high_band_gain = float(np.abs(channel_response[active_carrier_indices[-edge_carrier_count:]]).mean())
    return dict(response=channel_response, frequencies_hz=frequencies_hz, band_limited_ir=band_limited_ir,
                group_delay_span_ns=group_delay_span_ns, low_band_gain=low_band_gain, high_band_gain=high_band_gain)


def channel_zero(sent, received):
    """Dominant zero magnitude rho of the channel main lobe, and the linear-fit RRMSE.
    A ridge FIR is fit then TRUNCATED to the real channel memory before taking roots, so
    the out-of-band-extrapolation tail cannot inflate rho."""
    validation_count = max(1, int(len(sent) * VAL_FRACTION))

    train_design = build_design(sent[:-validation_count], FIR_TAPS)
    train_target = np.concatenate([row[FIR_TAPS - 1:] for row in received[:-validation_count]])
    gram_matrix = train_design.T @ train_design + RIDGE * np.eye(FIR_TAPS)
    taps = np.linalg.solve(gram_matrix.astype(np.float64), (train_design.T @ train_target).astype(np.float64))

    val_design = build_design(sent[-validation_count:], FIR_TAPS)
    val_target = np.concatenate([row[FIR_TAPS - 1:] for row in received[-validation_count:]])
    rrmse = float(np.sqrt(np.mean((val_target - val_design @ taps) ** 2) / np.mean(val_target ** 2)) * 100)
    return taps, rrmse


def main():
    PLOT_PATH.mkdir(parents=True, exist_ok=True)
    sent, received, sample_rate_hz, active_carrier_indices = load(DATASET_PATH)
    sample_period_ns = 1e9 / sample_rate_hz
    fft_length = sent.shape[1]
    f_min_hz = active_carrier_indices[0] * sample_rate_hz / fft_length
    f_max_hz = active_carrier_indices[-1] * sample_rate_hz / fft_length

    channel = inband_channel(sent, received, sample_rate_hz, active_carrier_indices)
    rolloff = channel["high_band_gain"] / channel["low_band_gain"]

    # channel memory: in-band dispersion converted to taps at THIS sample rate
    channel_taps = max(1, int(np.ceil(channel["group_delay_span_ns"] / sample_period_ns)))

    # equalizer length: invert the channel's dominant zero (from the truncated main lobe)
    taps, rrmse = channel_zero(sent, received)
    roots = np.roots(taps[:max(channel_taps, 2)])
    zero_magnitude = float(np.abs(roots).max()) if roots.size else 0.0
    equalizer_taps = int(np.ceil(inversion_taps(zero_magnitude, INV_ENERGY_LOSS))) if zero_magnitude < 1.0 else None

    channel_rf = int(np.ceil(channel_taps * MARGIN))
    encoder_decoder_rf = int(np.ceil(equalizer_taps * MARGIN)) if equalizer_taps else None

    # full-band broadband FIR, for comparison only: its tap count is inflated/biased by
    # out-of-band extrapolation (the data does not constrain |f| outside the passband)
    tap_energy = np.cumsum(taps ** 2) / np.sum(taps ** 2)
    full_band_95 = int(np.searchsorted(tap_energy, 0.95) + 1)
    full_band_99 = int(np.searchsorted(tap_energy, 0.99) + 1)
    full_band_response = np.fft.rfft(taps, n=fft_length)

    print(f"dataset      : {Path(DATASET_PATH).name}")
    print(f"sample rate  : {sample_rate_hz / 1e6:.1f} MHz   (Ts = {sample_period_ns:.2f} ns/tap)")
    print(f"band         : {f_min_hz / 1e6:.2f} - {f_max_hz / 1e6:.2f} MHz")
    print(f"in-band |H|  : {channel['low_band_gain']:.2f} -> {channel['high_band_gain']:.2f}  "
          f"({rolloff:.2f}x rolloff)   linear-fit RRMSE {rrmse:.1f}%")
    print("-" * 60)
    print(f"channel memory (in-band)      : {channel_taps:4d} taps  ({channel['group_delay_span_ns']:.0f} ns)")
    print(f"full-band FIR (comparison)    : 95% energy {full_band_95:2d} taps, 99% energy {full_band_99:2d} taps  "
          f"(out-of-band-biased, see plot)")
    print(f"recommended channel-model RF  : {channel_rf:4d} taps   (memory x{MARGIN:g})")
    if encoder_decoder_rf:
        print(f"recommended encoder/decoder RF: {encoder_decoder_rf:4d} taps   "
              f"(equalizer {equalizer_taps} taps x{MARGIN:g}, channel zero rho={zero_magnitude:.2f})")
    else:
        print("recommended encoder/decoder RF: channel non-minimum-phase (rho>=1), "
              "needs bulk delay + anti-causal taps")

    # one figure: in-band data vs the full-band broadband FIR, in frequency and time
    fig, (freq_ax, time_ax) = plt.subplots(1, 2, figsize=(11, 4))
    freq_ax.plot(channel["frequencies_hz"][active_carrier_indices] / 1e6,
                 20 * np.log10(np.abs(channel["response"][active_carrier_indices]) + 1e-12),
                 "C0.-", ms=3, label="in-band H_k (data)")
    freq_ax.plot(channel["frequencies_hz"] / 1e6, 20 * np.log10(np.abs(full_band_response) + 1e-12),
                 "C3-", alpha=0.7, label="full-band FIR fit")
    freq_ax.axvspan(f_min_hz / 1e6, f_max_hz / 1e6, color="C2", alpha=0.10, label="data passband")
    freq_ax.set_xlabel("Frequency (MHz)")
    freq_ax.set_ylabel("|H| (dB)")
    freq_ax.set_title(f"Frequency: in-band data vs full-band FIR ({rolloff:.1f}x rolloff)")
    freq_ax.grid(True, alpha=0.3)
    freq_ax.legend(fontsize=8)

    ir_peak = int(np.argmax(np.abs(channel["band_limited_ir"])))
    ir_lags_ns = (np.arange(len(channel["band_limited_ir"])) - ir_peak) * sample_period_ns
    time_ax.plot(ir_lags_ns, channel["band_limited_ir"] / np.max(np.abs(channel["band_limited_ir"])),
                 "C1-", label="in-band IR (band-limited)")
    tap_peak = int(np.argmax(np.abs(taps)))
    tap_lags_ns = (np.arange(len(taps)) - tap_peak) * sample_period_ns
    time_ax.plot(tap_lags_ns, taps / np.max(np.abs(taps)), "C3.-", ms=3,
                 label=f"full-band FIR taps (99%@{full_band_99})")
    time_ax.set_xlim(-channel_taps * sample_period_ns * 4, channel_taps * sample_period_ns * 4)
    time_ax.axvspan(-channel["group_delay_span_ns"] / 2, channel["group_delay_span_ns"] / 2,
                    color="C2", alpha=0.15, label=f"in-band memory ~{channel_taps} taps")
    time_ax.set_xlabel("Lag (ns)")
    time_ax.set_ylabel("Normalized amplitude")
    time_ax.set_title("Impulse response (normalized)")
    time_ax.grid(True, alpha=0.3)
    time_ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(PLOT_PATH / "fir_channel.png", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
