'''
The purpose of this file is to contain the large helper classes used in scripts that
create results/figures for the manuscript
'''
from pyflux.core.experiment import ExperimentalContext
from pyflux.core.block import Signal, ActionBlock, FunctionalBlock, ResamplingBlock
from pyflux.core.chain import Chain
from pathlib import Path
from scipy.signal import resample_poly, find_peaks, correlate, resample
from fractions import Fraction
import matplotlib.pyplot as plt
import numpy as np
import torch
import zarr
import time
from datetime import datetime

def generate_random_bits(N) -> str:
    x = np.random.randint(0, 2, size=N)
    return ''.join(str(bit) for bit in x)

def generate_zadoff_chu(length, root=1):
    """Generate Real-valuded Zadoff-Chu sequence normalized between -1 and 1"""
    num_pad = 6
    n = np.arange(length - 2 * num_pad)
    if length % 2 == 0:
        zc = np.exp(-1j * np.pi * root * n * (n + 1) / length)
    else:
        zc = np.exp(-1j * np.pi * root * n**2 / length)

    # add 6 zeros on either side
    padding_zeros = np.zeros(num_pad, dtype=np.complex128)
    zc = np.concatenate([padding_zeros, zc, padding_zeros])
    max_val = np.max(np.abs(zc))
    zc = zc / max_val
    return np.real(zc)

def generate_zadoff_chu_freq(length, root=1):
    '''Complex constant-modulus (CAZAC) Zadoff-Chu sequence for mapping onto OFDM
    subcarriers. |zc| = 1, giving a band-limited preamble with a sharp autocorrelation'''
    n = np.arange(length)
    if length % 2 == 0:
        return np.exp(-1j * np.pi * root * n * (n + 1) / length)
    return np.exp(-1j * np.pi * root * n ** 2 / length)

def band_limited_zc_preamble(preamble_length, baseband_sampling_rate, f_min, f_max, amplitude=3.0):
    '''Band-limited Zadoff-Chu preamble: a CAZAC sequence on the passband bins only,
    IFFT'd to time and scaled to +-amplitude peak. Single source of truth shared by
    ModulateDataOFDM and encoder/decoder training so both use an identical preamble.'''
    freqs = np.fft.rfftfreq(preamble_length, d=1.0 / baseband_sampling_rate)
    in_band = np.where((freqs >= f_min) & (freqs <= f_max))[0]
    spectrum = np.zeros(preamble_length // 2 + 1, dtype=complex)
    spectrum[in_band] = generate_zadoff_chu_freq(len(in_band))
    preamble = np.fft.irfft(spectrum, n=preamble_length, norm='ortho')
    return amplitude * preamble / np.max(np.abs(preamble))

class ModulateDataOFDM(FunctionalBlock):
    def __init__(self,
                 constellation,
                 f_min,
                 f_max,
                 subcarrier_spacing: float,
                 preamble_method: str,
                 awg_table_fraction: float,
                 cyclic_prefix_fraction: float,
                 upsample_factor: int,
                 preamble_length: int = 256,
                 power_min: float = None,
                 power_max: float = None,
                 jitter_power: float = 0.0,
                 clip_threshold: float = 3.0,
                 ):
        self.AWG_TABLE_LENGTH = 16_384
        self.output_length = int(self.AWG_TABLE_LENGTH * awg_table_fraction)
        self.subcarrier_freqs_hz = np.arange(f_min, f_max, subcarrier_spacing)
        self.subcarrier_indicies = np.round(self.subcarrier_freqs_hz / subcarrier_spacing).astype(int)
        self.num_carriers = len(self.subcarrier_freqs_hz)
        self.bits_per_symbol = constellation.bits_per_symbol
        self.constellation = constellation
        self.preamble_method = preamble_method
        self.preamble_length = preamble_length
        self.leading_zero_subcarriers = np.arange(subcarrier_spacing, f_min, subcarrier_spacing) # Exclude DC
        self.baseband_fft_length = upsample_factor * 2 * (self.num_carriers + len(self.leading_zero_subcarriers) + 1)
        self.cyclic_prefix_length = int(round(cyclic_prefix_fraction * self.baseband_fft_length))
        self.baseband_sampling_rate = self.baseband_fft_length * subcarrier_spacing
        self.baseband_ofdm_symbol_length = self.baseband_fft_length + self.cyclic_prefix_length
        self.baseband_burst_length = preamble_length + self.baseband_ofdm_symbol_length

        self.fs_out = self.baseband_sampling_rate
        super().__init__(self.fs_out)

        self.awg_frequency = subcarrier_spacing * (self.baseband_fft_length / self.baseband_burst_length)

        self.upsample_factor = upsample_factor
        self.first_data_bin = len(self.leading_zero_subcarriers) + 1

        assert self.output_length in [self.AWG_TABLE_LENGTH//16, self.AWG_TABLE_LENGTH//8, self.AWG_TABLE_LENGTH//4, self.AWG_TABLE_LENGTH//2, self.AWG_TABLE_LENGTH], "Waveform length in samples must be one of [1000, 2000, 4000, 8000, 16000] for compatibility with Agilent 33250A arbitrary waveform generator"
        assert upsample_factor in [1, 2, 4, 8, 16], "Upsample factor must be one of [1, 2, 4, 8, 16]"
        nyquist_frequency = subcarrier_spacing * (self.baseband_fft_length // 2)
        assert f_max <= nyquist_frequency, f"Maximum subcarrier frequency {f_max: .2e} must be less than Nyquist frequency of {nyquist_frequency: .2e}"
        assert self.output_length >= self.baseband_burst_length, f"AWG waveform length must be at least as long as the baseband burst length of {self.baseband_burst_length} samples"

        self.power_min = power_min
        self.power_max = power_max
        self.jitter_power = jitter_power
        self.clip_threshold = clip_threshold
        self.preamble = self._build_preamble()

    def _build_preamble(self):
        '''Band-limited Zadoff-Chu preamble on the active-carrier passband, so it survives
        the in-band encoder/decoder and carries no energy in the LED's low-frequency noise
        band. Scaled to peak 3 to own the burst global peak.'''
        if self.preamble_method != "zadoff_chu":
            raise ValueError("No valid preamble method chosen!")
        return band_limited_zc_preamble(
            self.preamble_length, self.baseband_sampling_rate,
            float(self.subcarrier_freqs_hz.min()), float(self.subcarrier_freqs_hz.max()), amplitude=3.0)

    def transform(self, x:Signal) -> Signal:
        source_bits = generate_random_bits(self.bits_per_symbol * self.num_carriers)
        active_subcarriers = self.constellation.bits_to_symbols(source_bits)

        # Off-constellation jitter for channel-model training richness
        if self.jitter_power:
            jitter = np.sqrt(self.jitter_power / 2) * (
                np.random.randn(self.num_carriers) + 1j * np.random.randn(self.num_carriers))
            active_subcarriers = active_subcarriers + jitter

        # Place carriers in a fixed-length half spectrum (DC, leading bins, and all
        # bins above f_max stay zero) and irfft to a fixed-length real symbol.
        half_spectrum = np.zeros(self.baseband_fft_length // 2 + 1, dtype=complex)
        half_spectrum[self.first_data_bin : self.first_data_bin + self.num_carriers] = active_subcarriers
        time_domain_signal = np.fft.irfft(half_spectrum, n=self.baseband_fft_length, norm='ortho') # Use sinc interpolation for smooth analog waveform

        # Random power scaling: normalize to unit RMS then scale to U(power_min, power_max).
        # Increases training diversity
        if self.power_min is not None and self.power_max is not None:
            rms = np.sqrt(np.mean(time_domain_signal ** 2)) + 1e-12
            target_power = np.random.uniform(self.power_min, self.power_max)
            time_domain_signal = time_domain_signal / rms * np.sqrt(target_power)

        cyclic_prefix = time_domain_signal[-self.cyclic_prefix_length:]
        time_domain_signal_with_cp = np.concatenate((cyclic_prefix, time_domain_signal))

        preamble = self.preamble
        time_domain_signal_with_cp = np.clip(time_domain_signal_with_cp, -self.clip_threshold, self.clip_threshold)
        baseband_burst = np.concatenate((preamble, time_domain_signal_with_cp))


        # Now, upsample from baseband sampling rate to AWG sampling rate with sinc interpolation
        awg_waveform = resample(baseband_burst, self.output_length)
        awg_waveform = np.clip(awg_waveform, -self.clip_threshold, self.clip_threshold)


        payload = dict(
            preamble=preamble,
            source_bits=source_bits,
            sent_symbols=active_subcarriers,
            sent_baseband=time_domain_signal,    # symbol only (no CP/preamble); kept for validation plotting
            sent_burst=baseband_burst,           # full burst [preamble | CP | symbol]
            awg_waveform=awg_waveform,
            awg_frequency=self.awg_frequency,
        )

        # Create signal with associated payloads
        x = Signal(
            data = baseband_burst,
            artifact_container = payload,
            sampling_rate = self.fs_out
        )

        return x
    
class SendWaveform(ActionBlock):
    def __init__(
            self,
            fs: float,
            awg_driver: object,
            freq: float,
            amplitude: float,
            offset: float
        ):
        super().__init__(fs)
        self.driver = awg_driver
        self.freq = freq
        self.amplitude = amplitude
        self.offset = offset
    
    def action(self, x:Signal) -> Signal:
        awg_waveform = x.artifact_container['awg_waveform']
        assert len(awg_waveform) <= 16_384, "Agilent 33250A max arbitrary waveform length is 16384 points"
        self.driver.send_arbitrary(
            samples=awg_waveform,
            freq=self.freq,
            amplitude=self.amplitude,
            offset=self.offset)
        time.sleep(3)
        
class MeasureWaveform(ResamplingBlock):
    def __init__(
            self,
            fs_in,
            fs_out,
            osc_driver,
            input_signal_frequency,
            trigger_channel,
            data_channel,
            debug=False
        ):
        # The capture is the rate transition: signal arrives at the modulation
        # rate (fs_in) and leaves at the oscilloscope's sample rate (fs_out).
        super().__init__(fs_in, fs_out)
        self.driver = osc_driver
        self.input_signal_frequency = input_signal_frequency
        self.trigger_channel = trigger_channel
        self.data_channel = data_channel
        self.debug = debug

    def resample(self, x: Signal) -> Signal:
        timesteps, voltages = self.driver.measure_waveform(channel=self.data_channel)
        true_fs = 1.0 / float(np.mean(np.diff(timesteps)))
        if not np.isclose(true_fs, self.fs_out, rtol=1e-3):
            raise ValueError(
                f"MeasureWaveform: scope's true sample rate {true_fs:.6g} Hz (from xincr) "
                f"does not match declared fs_out {self.fs_out:.6g} Hz."
            )
        x.sampling_rate = self.fs_out
        x.data = voltages
        if self.debug:
            print(f"MeasureWaveform: measured {len(voltages)} samples at {true_fs:.2e} Hz (expected {self.fs_out:.2e} Hz)")
            plt.figure()
            plt.plot(timesteps, voltages)
            plt.title("MeasureWaveform: measured waveform")
            plt.xlabel("Time (s)")
            plt.ylabel("Voltage (V)")
            plt.show()
        return x
            
class ResampleMeasuredWaveform(ResamplingBlock):
    def __init__(self, fs_in, fs_out, debug=False):
        super().__init__(fs_in, fs_out)
        self.debug = debug

    def resample(self, x: Signal) -> Signal:
        # Estimate up and down with rational fraction
        original_len = len(x.data)
        frac = Fraction(self.fs_out / self.fs_in).limit_denominator(1000)
        up = frac.numerator
        down = frac.denominator
        resampled_data = resample_poly(x.data, up, down)
        x.data = resampled_data
        x.sampling_rate = self.fs_out
        if self.debug:
            print(f"ResampleMeasuredWaveform: resampled from {self.fs_in:.2e} Hz to {self.fs_out:.2e} Hz "
                  f"using up={up}, down={down}, resulting in length change of {original_len} to {len(resampled_data)}samples")
        return x

class DemodulateDataOFDM(FunctionalBlock):
    def __init__(self,
                    constellation,
                    f_min,
                    f_max,
                    subcarrier_spacing,
                    preamble_method,
                    baseband_fft_length,
                    cyclic_prefix_length,
                    upsample_factor,
                    debug=False):

        self.fs_in = baseband_fft_length * subcarrier_spacing
        super().__init__(self.fs_in)
        self.constellation = constellation
        self.subcarrier_freqs_hz = np.arange(f_min, f_max, subcarrier_spacing)
        self.preamble_method = preamble_method
        self.ofdm_symbol_length_with_cp = baseband_fft_length + cyclic_prefix_length
        self.cyclic_prefix_length = cyclic_prefix_length
        self.upsample_factor = upsample_factor
        self.subcarrier_indicies = np.round(self.subcarrier_freqs_hz / subcarrier_spacing).astype(int)
        self.debug = debug

    def find_preamble_start(self, y_t, preamble, chosen_peak_index=0):
        corr = correlate(y_t, preamble, mode='valid')
        peaks, _ = find_peaks(corr, height=0.95 * np.max(np.abs(corr)), distance=len(preamble))
        if len(peaks) == 0:
            raise ValueError("Preamble not found in received signal")
        needed = self.ofdm_symbol_length_with_cp
        valid = [p for p in peaks if p + len(preamble) + needed <= len(y_t)]
        if not valid:
            raise ValueError(f"No preamble has a full OFDM symbol after it: capture holds "
                            f"{len(y_t)} samples, need {needed} after each preamble. Widen the scope window.")
        return valid[chosen_peak_index], corr, peaks

    def _plot_cp_removed_symbol(self, y_t, ofdm_symbol_start, preamble_length, tail=None,
                                perfect_preamble=None):
        cp = self.cyclic_prefix_length
        win_start = ofdm_symbol_start + cp
        win_end = win_start + (self.ofdm_symbol_length_with_cp - cp)
        if tail is None:
            tail = preamble_length # show a full preamble's worth past the window
        plot_end = min(win_end + tail, len(y_t))
        n = np.arange(win_end - 30, plot_end)

        fig, (ax_sym, ax_cp) = plt.subplots(2, 1, figsize=(12, 6), sharey=True)

        # top: end of the FFT window + trailing samples (watch for next preamble leaking in)
        ax_sym.plot(n, y_t[win_end - 30:plot_end], lw=0.8)
        ax_sym.axvspan(win_end - 30, win_end, color="tab:green", alpha=0.12,
                       label="FFT window (CP removed)")
        ax_sym.axvline(win_end, color="red", lw=1.2,
                       label="window end / next-symbol boundary")
        ax_sym.set_title("DemodulateDataOFDM: CP-removed OFDM symbol")
        ax_sym.set_xlabel("Sample index")
        ax_sym.set_ylabel("Amplitude")
        ax_sym.legend(loc="upper right")
        ax_sym.grid(True, alpha=0.3)

        # bottom: original perfect preamble vs the preamble extracted from this capture,
        # so sync/distortion is obvious. On a clean, well-synced capture the extracted
        # preamble should track the perfect one sample-for-sample.
        preamble_start = ofdm_symbol_start - preamble_length
        recv_preamble = y_t[preamble_start : preamble_start + preamble_length]  # extracted preamble
        k = np.arange(preamble_length)
        ax_cp.plot(k, recv_preamble, lw=1.0, ls="--", label="extracted preamble (received)")
        if perfect_preamble is not None:
            ax_cp.plot(k, np.asarray(perfect_preamble), lw=1.2, label="original preamble (perfect)")
        ax_cp.set_title("Original vs extracted preamble (should overlap)")
        ax_cp.set_xlabel("Preamble sample index")
        ax_cp.set_ylabel("Amplitude")
        ax_cp.legend(loc="upper right")
        ax_cp.grid(True, alpha=0.3)

        fig.tight_layout()
        plt.show()

    def transform(self, x:Signal) -> Signal:
        preamble = x.artifact_container['preamble']
        y_t = x.data
        preamble_start, corr, peaks = self.find_preamble_start(y_t, preamble)
        if self.debug:
            plt.figure()
            plt.plot(corr)
            plt.plot(peaks, corr[peaks], "x")
            plt.plot(preamble_start, corr[preamble_start], "o", color="red")
            plt.title("DemodulateDataOFDM: preamble correlation (red = chosen peak)")
            plt.xlabel("Lag (samples)")
            plt.ylabel("Correlation")
            plt.show()
        ofdm_symbol_start = preamble_start + len(preamble)
        ofdm_symbol_end = ofdm_symbol_start + self.ofdm_symbol_length_with_cp
        ofdm_symbol = y_t[ofdm_symbol_start : ofdm_symbol_end]
        cyclic_prefix_removed = ofdm_symbol[self.cyclic_prefix_length :]

        if self.debug:
            self._plot_cp_removed_symbol(y_t, ofdm_symbol_start, len(preamble),
                                         perfect_preamble=preamble)

        # Preamble-aligned full burst, same length as sent_burst
        x.artifact_container['received_burst'] = y_t[preamble_start : preamble_start + len(preamble) + self.ofdm_symbol_length_with_cp]

        # Take FFT and extract subcarriers
        Y_k = np.fft.fft(cyclic_prefix_removed, norm='ortho')[self.subcarrier_indicies]
        bits_estimated = self.constellation.symbols_to_bits(Y_k)
        x.artifact_container['estimated_bits'] = bits_estimated
        x.artifact_container['received_symbols'] = Y_k
        x.artifact_container['subcarrier_freqs_hz'] = self.subcarrier_freqs_hz
        x.data = cyclic_prefix_removed
        return x
    
class PlotConstellations(ActionBlock):
    def __init__(self, fs=0):
        super().__init__(fs)

    def action(self, x: Signal):
        received = np.asarray(x.artifact_container['received_symbols'])
        sent = np.asarray(x.artifact_container['sent_symbols'])
        freqs = np.asarray(x.artifact_container['subcarrier_freqs_hz'])
        assert len(received) == len(sent) == len(freqs), "Number of received symbols does not match number of sent symbols"
        fig, (ax_sent, ax_recv) = plt.subplots(1, 2, figsize=(11, 5))
        ax_sent.scatter(sent.real, sent.imag, s=10, c=freqs, cmap='viridis')
        ax_sent.set_title('Sent')
        sc = ax_recv.scatter(received.real, received.imag, s=10, c=freqs, cmap='viridis')
        ax_recv.set_title('Received')
        for ax in (ax_sent, ax_recv):
            ax.set_xlabel('In-Phase')
            ax.set_ylabel('Quadrature')
            ax.grid(True)
            ax.set_aspect('equal', 'box')
        fig.colorbar(sc, ax=[ax_sent, ax_recv], label='Carrier Frequency (Hz)')
        plt.show()
        return x

class SendAndReceiveOFDM(Chain):
    def __init__(
                 self,
                 modulate_block: ModulateDataOFDM,
                 send_block: SendWaveform,
                 receive_block: MeasureWaveform,
                 resample_block: ResampleMeasuredWaveform,
                 demodulate_block: DemodulateDataOFDM
                 ):
        super().__init__([modulate_block, send_block, receive_block, resample_block, demodulate_block])


class ApplyEncoder(FunctionalBlock):
    '''
    Insert a frozen encoder after modulation: pre-distort only the OFDM symbol
    (CP + payload) and re-prepend the pristine preamble, so hardware sync correlates
    against an untouched template. Then rebuild the AWG waveform, clamped to +-clip_value.
    '''
    def __init__(self, encoder, modulate_block: ModulateDataOFDM, clip_value: float, device="cpu"):
        super().__init__(modulate_block.fs_out)
        self.encoder = encoder
        self.output_length = modulate_block.output_length
        self.preamble_length = modulate_block.preamble_length
        self.clip_value = clip_value
        self.device = device

    def _apply(self, signal):
        t = torch.tensor(signal, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            return self.encoder(t).squeeze(0).cpu().numpy()

    def transform(self, x: Signal) -> Signal:
        burst = np.array(x.data, copy=True)
        preamble, symbol = burst[:self.preamble_length], burst[self.preamble_length:]

        x.artifact_container['encoder_input'] = symbol

        encoded_symbol = self._apply(symbol)
        encoded_symbol = np.clip(encoded_symbol, -self.clip_value, self.clip_value)
        x.artifact_container['encoder_output'] = encoded_symbol
        encoded = np.concatenate((preamble, encoded_symbol))
        x.data = encoded
        x.artifact_container['awg_waveform'] = np.clip(resample(encoded, self.output_length),
                                                       -self.clip_value, self.clip_value)
        
        
        return x


class ApplyDecoder(FunctionalBlock):
    '''Insert a frozen decoder before demodulation: sync on the pristine preamble in the
    raw received stream, equalize only the OFDM symbol (CP + payload), and splice it back
    so demod re-syncs on the untouched preamble and demodulates normally. Sync therefore
    never depends on the encoder/decoder.'''
    def __init__(self, decoder, demodulate_block: DemodulateDataOFDM, device="cpu", debug=False):
        super().__init__(demodulate_block.fs_in)
        self.decoder = decoder
        self.demod = demodulate_block
        self.device = device
        self.debug = debug

    def _apply(self, signal):
        t = torch.tensor(signal, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            return self.decoder(t).squeeze(0).cpu().numpy()

    def transform(self, x: Signal) -> Signal:
        received = np.array(x.data, copy=True)
        preamble = x.artifact_container['preamble']
        preamble_start, _, _ = self.demod.find_preamble_start(received, preamble)
        symbol_start = preamble_start + len(preamble)
        symbol_end = symbol_start + self.demod.ofdm_symbol_length_with_cp
        if self.debug:
            print(f"ApplyDecoder: received {len(received)} samples, symbol [{symbol_start}:{symbol_end}]")

        decoder_input = received[symbol_start:symbol_end]
        x.artifact_container['decoder_input'] = decoder_input
        decoded_symbol = self._apply(received[symbol_start:symbol_end])
        x.artifact_container['decoder_output'] = decoded_symbol
        decoded = received.copy()
        decoded[symbol_start:symbol_end] = decoded_symbol
        x.data = decoded
        x.artifact_container['decoder_window'] = (self.demod.cyclic_prefix_length,)
        return x


class AppendToDataset(ActionBlock):
    def __init__(self,
                 fs_in: float,
                 dc_offset: float,
                 f_min: float,
                 f_max: float,
                 active_carrier_indices,
                 cyclic_prefix_length: int,
                 preamble_length: int,
                 modulation_format: str,
                 block_size: int = 64,
                 dataset_path: Path = None):
        super().__init__(fs_in)
        self.block_size = block_size
        self._arrays = None

        if dataset_path is not None:
            self.dataset_path = Path(dataset_path).resolve()
            self.group = zarr.open_group(self.dataset_path, mode="a")
            print(f"[AppendToDataset] appending to existing dataset: {self.dataset_path}")
        else:
            sweeps_dir = Path(__file__).resolve().parents[1] / "data" / "sweeps"
            sweeps_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            name = f"dc{dc_offset:g}A_fmin{f_min:g}_fmax{f_max:g}_{timestamp}.zarr"
            self.dataset_path = sweeps_dir / name
            self.group = zarr.open_group(self.dataset_path, mode="w-")
            print(f"[AppendToDataset] created new dataset: {self.dataset_path}")

        self.group.attrs.update({
            "dc_offset_A": float(dc_offset),
            "f_min_hz": float(f_min),
            "f_max_hz": float(f_max),
            "active_carrier_indices": np.asarray(active_carrier_indices).astype(int).tolist(),
            "cyclic_prefix_length": int(cyclic_prefix_length),
            "preamble_length": int(preamble_length),
            "modulation_format": str(modulation_format),
        })

    @staticmethod
    def _bits_to_array(bits):
        if isinstance(bits, str):
            return np.array([int(b) for b in bits], dtype="uint8")
        return np.asarray(bits, dtype="uint8")

    def _init_arrays(self, fields):
        b = self.block_size
        self._arrays = {}
        for name, row in fields.items():
            if name in self.group:
                self._arrays[name] = self.group[name]
            else:
                self._arrays[name] = self.group.create_array(
                    name, shape=(0, row.shape[0]), chunks=(b, row.shape[0]), dtype=row.dtype)

    def action(self, x: Signal):
        fields = {
            "sent_burst": np.asarray(x.artifact_container["sent_burst"], dtype="float32"),
            "received_burst": np.asarray(x.artifact_container["received_burst"], dtype="float32"),
            "source_bits": self._bits_to_array(x.artifact_container["source_bits"]),
            "decoded_bits": self._bits_to_array(x.artifact_container["estimated_bits"]),
        }
        if self._arrays is None:
            self._init_arrays(fields)
        for name, row in fields.items():
            self._arrays[name].append(row[None, :])
        return x


class CheckChannel:
    _PROBE_HZ = 1e6
    _DRIVE_VPP = 2.0
    _N_AVG = 128
    _RECORD_LEN = 20_000
    _PERIODS = 4
    _MEAS_SLOT = 1
    _MEAS_TYPE = "RMS"
    SLEEP_TIME = 3.0 # Ensure the average has long enough to settle before measuring the waveform

    def __init__(self, awg_driver, osc_driver, data_channel):
        self.awg = awg_driver
        self.osc = osc_driver

        self.data_channel = data_channel

    def run(self, label=""):
        self.awg.apply_output_fun(
            waveform="SIN", frequency=self._PROBE_HZ,
            amplitude=self._DRIVE_VPP, offset=0.0)
        self.osc.set_record_length(self._RECORD_LEN)
        self.osc.set_horizontal_scale(self._PERIODS / (10.0 * self._PROBE_HZ))
        # _, v = self.osc.measure_waveform(channel=self.data_channel)
        # pkpk0 = float(np.ptp(v))
        # self.osc.configure_channel(ch=self.data_channel, scale=max(pkpk0 / 8.0, 1e-3), offset=0)
        self.osc.set_acquire_mode("AVErage", self._N_AVG)
        self.osc.resume_acquire()  # must be actively acquiring for the average to accumulate
        self.osc.select_measurement(self._MEAS_SLOT, self.data_channel, self._MEAS_TYPE)
        time.sleep(self.SLEEP_TIME)
        vrms = self.osc.read_measurement(self._MEAS_SLOT)
        mode = self.osc.get_acquire_mode()
        self.osc.set_acquire_mode("SAMple")
        self.osc.resume_acquire()
        prefix = f"[{label}] " if label else ""
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[check_channel {ts}] {prefix}1 MHz V{self._MEAS_TYPE.lower()} = {vrms:.4f} V  ({self._N_AVG}-avg, mode={mode})")
        return vrms