"""
Measure how the channel response depends on the AWG's zero-order-hold granularity
(AWG_TABLE_FRACTION).

The baseband burst is always resampled to table_length = 16384 * fraction points, and
the AWG plays the whole table once per burst period (f_AWG, fraction-independent), so
the implied table clock is table_length * f_AWG. Fraction 0.5 therefore holds each
point 2x longer than fraction 1, 0.25 holds 4x longer, and so on. NOTE the Agilent
33250A DAC tops out at 200 MS/s: at this burst rate every fraction above 0.125 implies
a table clock past that ceiling, so the instrument is internally decimating or
interpolating. The measured differences capture whatever it actually does, not just
ideal ZOH sinc rolloff.

For each fraction in AWG_TABLE_FRACTIONS the same N_BURSTS symbol bursts (identical
seed, fixed power, no jitter) are sent through the standard OFDM chain and H(f) is
fit per carrier. Everything is compared against fraction 1.0 (finest table); a repeat
of 1.0 at the end separates real ZOH effects from channel drift. FractionalSync aligns
each capture on its own preamble (which passes the same ZOH), so ideal ZOH's linear
phase cancels, and any surviving phase difference is a non-ideal reconstruction effect.
Raw bursts are saved to an npz so the analysis can be re-run offline.

Edit the configuration below, then:
    cd <repo_root>/experiments
    python zoh_channel_comparison.py
"""
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from pyflux.core.experiment import ExperimentalContext
from pyflux.core.block import Signal
from modules.experimental_blocks import (
    ModulateDataOFDM, SendWaveform, MeasureWaveform, ResampleMeasuredWaveform,
    FractionalSync, DemodulateDataOFDM, SendAndReceiveOFDM, CheckChannel,
)
from modules.constellation_diagram import get_constellation

HERE = Path(__file__).resolve().parent

# Configuration
CONFIG_FILE = HERE / "train_and_validate.yml"   # OFDM settings + hardware
AWG_TABLE_FRACTIONS = [1, 0.5, 0.25, 0.125, 1]  # trailing 1 = drift control
N_BURSTS = 5
FIXED_POWER = 1.5     # constant symbol power: ZOH is the only variable
SEED = 0              # reseeded per fraction -> identical bursts everywhere
REUSE_DATA = None     # path to a previous npz -> skip hardware, re-plot
SAVE_DIR = HERE.parent / "data/experiments/zoh_comparison"
PLOT_PATH = HERE.parent / "data/plots"
MEASURED_A_OFFSET = 0.000


def fraction_label(index, fraction):
    repeat = " (repeat)" if fraction in AWG_TABLE_FRACTIONS[:index] else ""
    return f"{fraction:g}{repeat}"


def collect():
    with ExperimentalContext(CONFIG_FILE=CONFIG_FILE) as Exp:
        cfg = Exp.config.DATA_COLLECTION

        awg = Exp.Agilent_33250A
        osc = Exp.Tektronix_TDS2000
        pwr_supply = Exp.HP_EE3631A

        awg.set_output_load("INF")
        osc.display_channel(ch=1)
        osc.display_channel(ch=3)
        osc.set_record_length(200_000)
        osc.set_trigger(ch=1, voltage_level=0)
        osc.set_probe_gain(ch=1, gain=1)
        osc.set_probe_gain(ch=3, gain=1)
        osc.configure_channel(ch=3, scale=cfg.OSC_SCALE, offset=0)
        osc.configure_channel(ch=1, scale=1, offset=0)
        osc.set_coupling(ch=1, coupling="AC")
        osc.set_coupling(ch=3, coupling="AC")

        dc_offset_A = float(cfg.DC_OFFSETS[0])
        assert dc_offset_A < 0.4, f"DC offset {dc_offset_A} A exceeds safe range for the LED driver"
        min_freq = float(cfg.F_MINS[0])
        max_freq = float(cfg.F_MAXS[0])
        osc_fs = float(cfg.OSC_SAMPLE_RATES[0])
        subcarrier_spacing = float(cfg.SUBCARRIER_SPACING)

        pwr_supply.set_6V(voltage=4, current=dc_offset_A)
        pwr_supply.enable_output()
        Exp.log(f"DC offset set to {dc_offset_A - MEASURED_A_OFFSET:.3f} A")

        check_channel = CheckChannel(awg_driver=awg, osc_driver=osc, data_channel=3)

        sent, received, attrs = {}, {}, {}
        for index, fraction in enumerate(AWG_TABLE_FRACTIONS):
            label = fraction_label(index, fraction)

            mod_ofdm = ModulateDataOFDM(
                constellation=get_constellation(cfg.MODULATION_FORMAT),
                f_min=min_freq, f_max=max_freq,
                subcarrier_spacing=subcarrier_spacing,
                preamble_method="zadoff_chu",
                awg_table_fraction=fraction,
                cyclic_prefix_fraction=cfg.CP_LENGTH_FRACTION,
                upsample_factor=cfg.UPSAMPLE_FACTOR,
                preamble_length=cfg.PREAMBLE_LENGTH,
                power_min=FIXED_POWER, power_max=FIXED_POWER,
                jitter_power=0.0,
                clip_threshold=float(cfg.CLIP_THRESHOLD),
            )
            f_AWG = mod_ofdm.awg_frequency
            table_rate = mod_ofdm.output_length * f_AWG
            Exp.log(f"fraction {label}: table {mod_ofdm.output_length} pts, "
                    f"implied clock {table_rate / 1e6:.1f} MS/s "
                    f"({'exceeds' if table_rate > 200e6 else 'within'} 33250A 200 MS/s)")

            if index == 0:
                attrs = dict(
                    active_carrier_indices=mod_ofdm.subcarrier_indicies,
                    cyclic_prefix_length=mod_ofdm.cyclic_prefix_length,
                    preamble_length=mod_ofdm.preamble_length,
                    f_min_hz=min_freq, awg_frequency=f_AWG,
                )
                osc.set_horizontal_scale(0.2 / f_AWG)

            check_channel.run(f"fraction {label}")
            osc.set_record_length(200_000)
            osc.set_horizontal_scale(0.2 / f_AWG)
            osc.configure_channel(ch=3, scale=cfg.OSC_SCALE, offset=0)

            send_and_receive = SendAndReceiveOFDM(
                mod_ofdm,
                SendWaveform(fs=mod_ofdm.fs_out, awg_driver=awg,
                             freq=f_AWG, amplitude=18, offset=0),
                MeasureWaveform(fs_in=mod_ofdm.fs_out, fs_out=osc_fs, osc_driver=osc,
                                input_signal_frequency=f_AWG,
                                trigger_channel=1, data_channel=3),
                ResampleMeasuredWaveform(fs_in=osc_fs, fs_out=mod_ofdm.baseband_sampling_rate),
                DemodulateDataOFDM(
                    constellation=get_constellation(cfg.MODULATION_FORMAT),
                    f_min=min_freq, f_max=max_freq, subcarrier_spacing=subcarrier_spacing,
                    preamble_method="zadoff_chu",
                    baseband_fft_length=mod_ofdm.baseband_fft_length,
                    cyclic_prefix_length=mod_ofdm.cyclic_prefix_length,
                    upsample_factor=cfg.UPSAMPLE_FACTOR),
                fractional_sync_block=FractionalSync(
                    fs=mod_ofdm.baseband_sampling_rate, f_min=min_freq, f_max=max_freq),
            )

            np.random.seed(SEED)  # identical bits/power draws at every fraction
            sent[label], received[label] = [], []
            for burst_index in range(N_BURSTS):
                x = send_and_receive.run(Signal(data=np.zeros(1), sampling_rate=mod_ofdm.fs_out))
                sent[label].append(np.asarray(x.artifact_container["sent_burst"], dtype="float32"))
                received[label].append(np.asarray(x.artifact_container["received_burst"], dtype="float32"))
                Exp.log(f"fraction {label}: gathered {burst_index + 1}/{N_BURSTS}")

        SAVE_DIR.mkdir(parents=True, exist_ok=True)
        out_path = SAVE_DIR / f"zoh_comparison_{datetime.now().strftime('%Y%m%d_%H%M')}.npz"
        np.savez(out_path,
                 labels=list(sent.keys()),
                 **{f"sent_{label}": np.stack(sent[label]) for label in sent},
                 **{f"received_{label}": np.stack(received[label]) for label in received},
                 **attrs)
        Exp.log(f"saved raw bursts to {out_path}")
    return out_path


def analyze(data_path):
    data = np.load(data_path, allow_pickle=True)
    labels = list(data["labels"])
    active_carrier_indices = np.asarray(data["active_carrier_indices"])
    symbol_offset = int(data["preamble_length"]) + int(data["cyclic_prefix_length"])
    subcarrier_spacing_hz = float(data["f_min_hz"]) / active_carrier_indices[0]
    frequencies_hz = active_carrier_indices * subcarrier_spacing_hz
    frequencies_mhz = frequencies_hz / 1e6
    awg_frequency = float(data["awg_frequency"])

    response_by_fraction = {}
    for label in labels:
        sent = data[f"sent_{label}"][:, symbol_offset:].astype(np.float64)
        received = data[f"received_{label}"][:, symbol_offset:].astype(np.float64)
        sent_spectrum = np.fft.fft(sent, norm="ortho", axis=1)[:, active_carrier_indices]
        received_spectrum = np.fft.fft(received, norm="ortho", axis=1)[:, active_carrier_indices]
        carrier_ratio = received_spectrum / sent_spectrum

        # remove each burst's residual sync delay before averaging: a single
        # multi-ns FractionalSync miss otherwise tilts the coherent mean phase
        # and ripples its magnitude (seen 2026-07-08, fraction 0.5 burst 2).
        # ZOH linear phase is common to preamble and symbol, so it was never
        # observable here anyway; real dispersion differences survive.
        aligned_ratios = []
        delays_ns = []
        for burst_index in range(carrier_ratio.shape[0]):
            phase = np.unwrap(np.angle(carrier_ratio[burst_index]))
            slope, intercept = np.polyfit(frequencies_hz, phase, 1)
            aligned_ratios.append(carrier_ratio[burst_index] * np.exp(-1j * (slope * frequencies_hz + intercept)))
            delays_ns.append(-slope / (2 * np.pi) * 1e9)
        print(f"fraction {label}: per-burst sync delay (ns) {np.round(delays_ns, 2)}")
        response_by_fraction[label] = np.mean(aligned_ratios, axis=0)

    reference_label = labels[0]  # fraction 1: finest table
    reference_phase = np.unwrap(np.angle(response_by_fraction[reference_label]))
    reference_magnitude = 20 * np.log10(np.abs(response_by_fraction[reference_label]))

    def zoh_theory_mag_db(label):
        table_rate = 16384 * float(label.split()[0]) * awg_frequency
        reference_rate = 16384 * float(reference_label.split()[0]) * awg_frequency
        return 20 * np.log10(np.sinc(frequencies_hz / table_rate) / np.sinc(frequencies_hz / reference_rate))

    fig, axes = plt.subplots(3, 1, figsize=(8, 12))
    magnitude_ax, phase_ax, nonlinear_phase_ax = axes
    print(f"reference: fraction {reference_label}")
    for color_index, label in enumerate(labels[1:]):
        magnitude_delta = 20 * np.log10(np.abs(response_by_fraction[label])) - reference_magnitude
        phase_delta = np.unwrap(np.angle(response_by_fraction[label])) - reference_phase
        phase_delta_nonlinear = phase_delta - np.polyval(np.polyfit(frequencies_mhz, phase_delta, 1), frequencies_mhz)
        print(f"fraction {label}: Δ|H| span={magnitude_delta.max() - magnitude_delta.min():.3f} dB, "
              f"Δ∠H span={np.ptp(phase_delta):.3f} rad, "
              f"linear-removed span={np.ptp(phase_delta_nonlinear):.3f} rad, at f_max={phase_delta[-1]:+.3f} rad")

        magnitude_ax.plot(frequencies_mhz, magnitude_delta, lw=1.5, color=f"C{color_index}",
                          label=f"fraction {label}")
        magnitude_ax.plot(frequencies_mhz, zoh_theory_mag_db(label), lw=1.0, ls="--", color=f"C{color_index}",
                          label=f"ideal ZOH {label}")
        phase_ax.plot(frequencies_mhz, phase_delta, lw=1.5, color=f"C{color_index}", label=f"fraction {label}")
        nonlinear_phase_ax.plot(frequencies_mhz, phase_delta_nonlinear, lw=1.5, color=f"C{color_index}",
                                label=f"fraction {label}")

    magnitude_ax.set_title(f"Δ|H| vs AWG table fraction (rel. fraction {reference_label})")
    magnitude_ax.set_ylabel("Δ|H| (dB)")
    phase_ax.set_title("Δ∠H (ideal ZOH is removed by preamble sync: expected ≈ 0)")
    phase_ax.set_ylabel("Δ∠H (rad)")
    nonlinear_phase_ax.set_title("Δ∠H, linear component removed")
    nonlinear_phase_ax.set_ylabel("Δ∠H (rad)")
    for ax in axes:
        ax.set_xlabel("Frequency (MHz)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    PLOT_PATH.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(PLOT_PATH / "zoh_channel_comparison.png", dpi=130, bbox_inches="tight")
    fig.savefig(PLOT_PATH / "zoh_channel_comparison.svg", format="svg", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    data_path = Path(REUSE_DATA) if REUSE_DATA else collect()
    analyze(data_path)
