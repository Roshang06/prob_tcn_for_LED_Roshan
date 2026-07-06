'''
send_ofdm_symbol.py — minimal send/receive of single OFDM bursts on the live hardware.

Pick OFDM settings in send_ofdm_symbol.yml, then run this to transmit one clean OFDM symbol
at a time and plot the sent vs received constellations.
'''
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from pyflux.core.experiment import ExperimentalContext
from pyflux.core.block import Signal
from modules.experimental_blocks import *
from modules.constellation_diagram import get_constellation

HERE = Path(__file__).resolve().parent
CONFIG_FILE = HERE / "send_ofdm_symbol.yml"

MEASURED_A_OFFSET = 0.002        # stable current sits ~2 mA below the set point


if __name__ == "__main__":
    with ExperimentalContext(CONFIG_FILE=CONFIG_FILE, create_log_file=False,
                             run_name="send_ofdm_symbol") as Exp:
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
        min_freq    = float(cfg.F_MINS[0])
        max_freq    = float(cfg.F_MAXS[0])
        osc_fs      = float(cfg.OSC_SAMPLE_RATES[0])
        subcarrier_spacing = float(cfg.SUBCARRIER_SPACING)
        constellation = get_constellation(getattr(cfg, "CONSTELLATION", "qpsk"))

        pwr_supply.set_6V(voltage=4, current=dc_offset_A)  # current-limited; never reaches 4 V
        pwr_supply.enable_output()
        Exp.log(f"DC offset set to {dc_offset_A - MEASURED_A_OFFSET:.3f} A")

        check_channel = CheckChannel(awg_driver=awg, osc_driver=osc, data_channel=3)

        mod_ofdm = ModulateDataOFDM(
            constellation=constellation,
            f_min=min_freq, f_max=max_freq,
            subcarrier_spacing=subcarrier_spacing,
            preamble_method="zadoff_chu",
            awg_table_fraction=cfg.AWG_TABLE_FRACTION,
            cyclic_prefix_fraction=cfg.CP_LENGTH_FRACTION,
            upsample_factor=cfg.UPSAMPLE_FACTOR,
            preamble_length=cfg.PREAMBLE_LENGTH,
            power_min=None, power_max=None, jitter_power=0.0,
        )

        f_AWG = mod_ofdm.awg_frequency
        Exp.log(f"AWG frequency {f_AWG:.2f} Hz")
        osc.set_horizontal_scale(0.2 / f_AWG)

        send_waveform = SendWaveform(
            fs=mod_ofdm.fs_out, awg_driver=awg, freq=f_AWG, amplitude=18.0, offset=0)
        measure_waveform = MeasureWaveform(
            fs_in=mod_ofdm.fs_out, fs_out=osc_fs, osc_driver=osc,
            input_signal_frequency=f_AWG, trigger_channel=1, data_channel=3, debug=True)
        resample_waveform = ResampleMeasuredWaveform(
            fs_in=osc_fs, fs_out=mod_ofdm.baseband_sampling_rate, debug=True)
        demod_ofdm = DemodulateDataOFDM(
            constellation=constellation,
            f_min=min_freq, f_max=max_freq, subcarrier_spacing=subcarrier_spacing,
            preamble_method="zadoff_chu",
            baseband_fft_length=mod_ofdm.baseband_fft_length,
            cyclic_prefix_length=mod_ofdm.cyclic_prefix_length,
            upsample_factor=cfg.UPSAMPLE_FACTOR, debug=True)

        def _restore_ofdm_scope():
            osc.set_record_length(200_000)
            osc.set_horizontal_scale(0.2 / f_AWG)
            osc.configure_channel(ch=3, scale=cfg.OSC_SCALE, offset=0)

        check_channel.run("start")
        _restore_ofdm_scope()

        send_and_receive = SendAndReceiveOFDM(
            mod_ofdm, send_waveform, measure_waveform, resample_waveform, demod_ofdm)

        def plot_constellations(x):
            """Sent vs received constellations; the received panel overlays the original
            sent symbols as red x's so each received point can be compared to its target."""
            sent  = np.asarray(x.artifact_container['sent_symbols'])
            recv  = np.asarray(x.artifact_container['received_symbols'])
            freqs = np.asarray(x.artifact_container['subcarrier_freqs_hz'])
            fig, (ax_sent, ax_recv) = plt.subplots(1, 2, figsize=(11, 5))
            ax_sent.scatter(sent.real, sent.imag, s=10, c=freqs, cmap='viridis')
            ax_sent.set_title('Sent')
            sc = ax_recv.scatter(recv.real, recv.imag, s=10, c=freqs, cmap='viridis')
            ax_recv.scatter(sent.real, sent.imag, marker='x', c='red', s=40,
                            linewidths=1.2, label='Sent (reference)')
            ax_recv.set_title('Received')
            ax_recv.legend(loc='upper right')
            for ax in (ax_sent, ax_recv):
                ax.set_xlabel('In-Phase')
                ax.set_ylabel('Quadrature')
                ax.grid(True)
                ax.set_aspect('equal', 'box')
            fig.colorbar(sc, ax=[ax_sent, ax_recv], label='Carrier Frequency (Hz)')
            plt.show()

        # transmit one OFDM burst at a time; Enter -> send another, q -> quit
        i = 0
        while input("\n[Enter] send an OFDM symbol, [q] quit: ").strip().lower() != "q":
            i += 1
            x = send_and_receive.run(Signal(data=np.zeros(1), sampling_rate=mod_ofdm.fs_out))
            Exp.log(f"symbol {i}: {len(x.artifact_container['sent_symbols'])} carriers "
                    f"({min_freq/1e6:.2f}-{max_freq/1e6:.2f} MHz)")
            plot_constellations(x)
