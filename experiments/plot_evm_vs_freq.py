"""
Model-independent noise-floor estimate for an OFDM dataset.

The same preamble is transmitted on every burst, so the cross-burst variance of the
*received* preamble is pure additive noise, independent of any channel model and of
whether the channel is linear or nonlinear (the deterministic channel output is
identical each burst and cancels in the mean). This yields per-frequency EVM% across
the OFDM band and a mean EVM% that serves as the noise-floor estimate.

Usage:
    python plot_evm_vs_freq.py <dataset.zarr> [--save out.png]
"""
import argparse

import numpy as np
import zarr
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", help="path to the zarr training dataset")
    parser.add_argument("--save", default=None, help="optional path to save the figure (PNG/SVG)")
    args = parser.parse_args()

    root = zarr.open_group(args.dataset, mode="r")
    attrs = dict(root.attrs)
    if "received_burst" not in root or "preamble_length" not in attrs:
        raise SystemExit("need burst-format capture with 'received_burst' + 'preamble_length'")

    preamble_length = int(attrs["preamble_length"])
    cyclic_prefix_length = int(attrs["cyclic_prefix_length"])
    active_carrier_indices = np.asarray(attrs["active_carrier_indices"])
    received_burst = np.asarray(root["received_burst"][:])
    sent_burst = np.asarray(root["sent_burst"][:]) if "sent_burst" in root else None

    # baseband sampling rate: fs = FFT_length * subcarrier_spacing
    symbol_length = received_burst.shape[1] - preamble_length - cyclic_prefix_length
    f_min_hz = float(attrs.get("f_min_hz"))
    f_max_hz = float(attrs.get("f_max_hz"))
    subcarrier_spacing_hz = f_min_hz / active_carrier_indices.min()
    sample_rate_hz = symbol_length * subcarrier_spacing_hz

    # this only works if the preamble is truly identical across bursts
    if sent_burst is not None:
        max_deviation = float(np.abs(sent_burst[:, :preamble_length] - sent_burst[0, :preamble_length]).max())
        if max_deviation > 1e-9:
            print(f"WARNING: sent preamble varies across bursts (max dev {max_deviation:.2e}); "
                  "noise estimate may be contaminated.")

    received_preamble = received_burst[:, :preamble_length]
    mean_preamble = received_preamble.mean(axis=0)   # deterministic channel output (noise averages out)
    noise = received_preamble - mean_preamble         # per-burst deviation = additive noise

    signal_spectrum = np.fft.rfft(mean_preamble)
    noise_spectrum = np.fft.rfft(noise, axis=1)
    signal_amplitude = np.abs(signal_spectrum)
    noise_amplitude = np.sqrt((np.abs(noise_spectrum) ** 2).mean(axis=0))
    frequencies_hz = np.fft.rfftfreq(preamble_length, d=1.0 / sample_rate_hz)
    evm_pct = 100.0 * noise_amplitude / (signal_amplitude + 1e-20)

    in_band = (frequencies_hz >= f_min_hz) & (frequencies_hz <= f_max_hz)
    band_frequencies = frequencies_hz[in_band]
    band_evm = evm_pct[in_band]
    mean_evm_pct = float(band_evm.mean())   # uniform average over band = noise-floor estimate

    print(f"dataset    : {args.dataset}")
    print(f"OFDM band  : {f_min_hz/1e6:.2f}-{f_max_hz/1e6:.2f} MHz  "
          f"({int(in_band.sum())} bins, fs={sample_rate_hz/1e6:.1f} MHz)")
    print(f"mean EVM%  : {mean_evm_pct:.1f}   (noise-floor estimate, uniform over band)")

    fig, ax = plt.subplots(figsize=(5, 3), constrained_layout=True)
    ax.plot(band_frequencies / 1e6, band_evm, marker=".", ms=4, lw=1)
    ax.axhline(mean_evm_pct, ls="--", c="k", lw=0.8, label=f"mean {mean_evm_pct:.1f}%")
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("EVM (%)")
    ax.set_title("Per-frequency noise floor (repeated preamble)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    if args.save:
        fig.savefig(args.save, dpi=200)
        print(f"saved figure to {args.save}")
    plt.show()


if __name__ == "__main__":
    main()
