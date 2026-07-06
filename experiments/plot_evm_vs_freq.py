"""
plot_evm_vs_freq.py — model-independent noise-floor estimate for an OFDM dataset.

The same preamble is transmitted on every burst, so the cross-burst variance of the
*received* preamble is pure additive noise — independent of any channel model and of whether
the channel is linear or nonlinear (the deterministic channel output is identical each burst
and cancels in the mean). This yields per-frequency EVM% across the OFDM band and a mean EVM%
that serves as the noise-floor estimate.

Usage:
    python plot_evm_vs_freq.py <dataset.zarr> [--save out.png]
"""
import argparse

import numpy as np
import zarr
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset", help="path to the zarr training dataset")
    ap.add_argument("--save", default=None, help="optional path to save the figure (PNG/SVG)")
    args = ap.parse_args()

    root = zarr.open_group(args.dataset, mode="r")
    attrs = dict(root.attrs)
    if "received_burst" not in root or "preamble_length" not in attrs:
        raise SystemExit("need burst-format capture with 'received_burst' + 'preamble_length'")

    pre = int(attrs["preamble_length"])
    cp  = int(attrs["cyclic_prefix_length"])
    ks  = np.asarray(attrs["active_carrier_indices"])
    R   = np.asarray(root["received_burst"][:])
    S   = np.asarray(root["sent_burst"][:]) if "sent_burst" in root else None

    # baseband sampling rate: fs = FFT_length * subcarrier_spacing
    sym_len = R.shape[1] - pre - cp
    f_min   = float(attrs.get("f_min_hz"))
    spacing = f_min / ks.min()
    fs      = sym_len * spacing
    f_max   = float(attrs.get("f_max_hz"))

    # this only works if the preamble is truly identical across bursts
    if S is not None:
        dev = float(np.abs(S[:, :pre] - S[0, :pre]).max())
        if dev > 1e-9:
            print(f"WARNING: sent preamble varies across bursts (max dev {dev:.2e}); "
                  "noise estimate may be contaminated.")

    Rp     = R[:, :pre]
    mean_p = Rp.mean(axis=0)     # deterministic channel output (noise averages out)
    noise  = Rp - mean_p         # per-burst deviation = additive noise

    Sf        = np.fft.rfft(mean_p)
    Nf        = np.fft.rfft(noise, axis=1)
    sig_amp   = np.abs(Sf)
    noise_amp = np.sqrt((np.abs(Nf) ** 2).mean(axis=0))
    freq      = np.fft.rfftfreq(pre, d=1.0 / fs)
    evm       = 100.0 * noise_amp / (sig_amp + 1e-20)

    band       = (freq >= f_min) & (freq <= f_max)
    fb, eb     = freq[band], evm[band]
    mean_evm   = float(eb.mean())     # uniform average over band = noise-floor estimate


    print(f"dataset    : {args.dataset}")
    print(f"OFDM band  : {f_min/1e6:.2f}-{f_max/1e6:.2f} MHz  ({int(band.sum())} bins, fs={fs/1e6:.1f} MHz)")
    print(f"mean EVM%  : {mean_evm:.1f}   (noise-floor estimate, uniform over band)")

    fig, ax = plt.subplots(figsize=(5, 3), constrained_layout=True)
    ax.plot(fb / 1e6, eb, marker=".", ms=4, lw=1)
    ax.axhline(mean_evm, ls="--", c="k", lw=0.8, label=f"mean {mean_evm:.1f}%")
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
