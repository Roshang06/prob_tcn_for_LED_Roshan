'''
The purpose of this file is to contain the large helper classes used in scripts that
create results/figures for the manuscript
'''
from pyflux.core.experiment import ExperimentalContext
from pyflux.core.block import Signal, ActionBlock, FunctionalBlock, ResamplingBlock
from pyflux.core.chain import Chain
from scipy.signal import resample_poly, find_peaks, correlate
from fractions import Fraction
import matplotlib.pyplot as plt
import numpy as np

def generate_random_bits(N) -> str:
    x = np.random.randint(0, 2, size=N)
    return ''.join(str(bit) for bit in x)

def generate_zadoff_chu(length, root=1):
    """Generate Real-valuded Zadoff-Chu sequence"""
    num_pad = 6
    n = np.arange(length - 2 * num_pad)
    if length % 2 == 0:
        zc = np.exp(-1j * np.pi * root * n * (n + 1) / length)
    else:
        zc = np.exp(-1j * np.pi * root * n**2 / length)

    # add 6 zeros on either side
    padding_zeros = np.zeros(num_pad, dtype=np.complex128)
    zc = np.concatenate([padding_zeros, zc, padding_zeros])
    return np.real(zc)

class ModulateDataOFDM(FunctionalBlock):
    def __init__(self,
                 constellation,
                 f_min,
                 f_max,
                 subcarrier_spacing: float,
                 preamble_method: str,
                 ofdm_symbol_length: int,
                 cyclic_prefix_length: int,
                 upsample_factor: int
                 ):
        self.fs_out = f_max * upsample_factor * subcarrier_spacing
        super().__init__(self.fs_out)
        self.subcarrier_freqs_hz = np.arange(f_min, f_max, subcarrier_spacing)
        self.num_carriers = len(self.subcarrier_freqs_hz)
        self.bits_per_symbol = constellation.bits_per_symbol
        self.constellation = constellation
        self.preamble_method = preamble_method
        self.ofdm_symbol_length = ofdm_symbol_length
        self.cyclic_prefix_length = cyclic_prefix_length
        self.leading_zero_subcarriers = np.arange(subcarrier_spacing, f_min, subcarrier_spacing) # Exclude DC
        self.upsample_factor = upsample_factor
        assert ofdm_symbol_length in [1000, 2000, 4000, 8000, 16000], "OFDM symbol length in samples must be one of [1000, 2000, 4000, 8000, 16000] for compatibility with Agilent 33250A arbitrary waveform generator"
        assert ofdm_symbol_length / cyclic_prefix_length == 4.0, "Cyclic prefix length must be 1/4 of OFDM symbol length"
        assert upsample_factor in [1, 2, 4, 8, 16], "Upsample factor must be one of [1, 2, 4, 8, 16]"
        assert self.ofdm_symbol_length % self.upsample_factor == 0, "Symbol length in samples must be divisible by upsample factor"
        fft_length = (ofdm_symbol_length - cyclic_prefix_length) / upsample_factor
        nyquist_frequency = subcarrier_spacing * (fft_length / 2)
        assert f_max < nyquist_frequency, f"Maximum subcarrier frequency must be less than Nyquist frequency of {nyquist_frequency}"

    def transform(self, x:Signal) -> Signal:
        source_bits = generate_random_bits(self.bits_per_symbol * self.num_carriers)
        complex_baseband = self.constellation.bits_to_symbols(source_bits)
        # Add zeros on front where no subcarriers are present
        complex_baseband = np.concatenate((np.zeros(len(self.leading_zero_subcarriers)), complex_baseband))

        dc_carrier = np.zeros(1)
        nyquist_carrier = np.zeros(1)

        # Upsample to desired symbol length
        num_padding_zero_carriers = (self.upsample_factor - 1) * len(complex_baseband)
        padded_complex_baseband = np.concatenate((complex_baseband, np.zeros(num_padding_zero_carriers)))
        hermitian_complex_baseband = np.conjugate(np.flip(padded_complex_baseband))
        ifft_input = np.concatenate((dc_carrier, padded_complex_baseband, nyquist_carrier, hermitian_complex_baseband))
        time_domain_signal = np.fft.ifft(ifft_input, norm='ortho').real

        # Add cyclic prefix
        cyclic_prefix = time_domain_signal[-self.cyclic_prefix_length:]
        time_domain_signal = np.concatenate((cyclic_prefix, time_domain_signal))

        # 16384 is the max arbitrary waveform length for the Agilent 33250A
        preamble_length = (384 // self.upsample_factor)

        if self.preamble_method == "zadoff_chu":
            preamble = generate_zadoff_chu(preamble_length)
        else:
            raise ValueError("No valid preamble method chosen!")

        # Add trailing zeros to fill total waveform length
        N_trailing = self.ofdm_symbol_length - len(time_domain_signal)
        time_domain_signal_with_trailing_zeros = np.concatenate((time_domain_signal, np.zeros(N_trailing)))
        total_waveform = np.concatenate((preamble, time_domain_signal_with_trailing_zeros))

        payload_container = dict()
        payload_container['preamble'] = preamble
        payload_container['source_bits'] = source_bits
        payload_container['sent_symbols'] = complex_baseband

        # Create signal with associated payloads
        x = Signal(
            data = total_waveform,
            artifact_container = payload_container,
            sampling_rate = self.fs_out
        )

        return x
    
class SendWaveform(ActionBlock):
    def __init__(
            self,
            fs: float,
            awg_driver: object,
            waveform_data: np.ndarray,
            freq: float,
            amplitude: float,
            offset: float
        ):
        super().__init__(fs)
        self.driver = awg_driver
        self.waveform_data = waveform_data
        self.freq = freq
        self.amplitude = amplitude
        self.offset = offset
    
    def action(self, x:Signal) -> Signal:
        self.driver.send_arbitrary(
            samples=self.waveform_data,
            freq=self.freq,
            amplitude=self.amplitude,
            offset=self.offset)
        
class MeasureWaveform(FunctionalBlock):
    def __init__(
            self,
            osc_driver,
            input_signal_frequency,
            trigger_channel,
            data_channel
        ):
        
        self.fs = osc_driver.input_sample_rate
        super().__init__(self.fs)
        self.driver = osc_driver
        self.input_signal_frequency = input_signal_frequency
        self.trigger_channel = trigger_channel
        self.data_channel = data_channel

    def transform(self, x: Signal) -> Signal:
        timesteps, voltages = self.driver.measure_waveform(channel=self.channel)
        x.sampling_rate = self.fs
        x.data = voltages
        return x
            
class ResampleMeasuredWaveform(ResamplingBlock):
    def __init__(self, fs_in, fs_out):
        self.fs_out = fs_out
        self.fs_in = fs_in

    def resample(self, x: Signal) -> Signal:
        # Estimate up and down with rational fraction
        frac = Fraction(self.fs_out, self.fs_in).limit_denominator(1000)
        up = frac.numerator
        down = frac.denominator
        resampled_data = resample_poly(x.data, up, down)
        x.data = resampled_data
        x.sampling_rate = self.fs_out
        return x

class DemodulateDataOFDM(FunctionalBlock):
    def __init__(self,
                    constellation,
                    f_min,
                    f_max,
                    subcarrier_spacing,
                    preamble_method,
                    ofdm_symbol_length,
                    cyclic_prefix_length,
                    upsample_factor):
        
        self.fs_in = f_max * upsample_factor * subcarrier_spacing
        super().__init__(self.fs_in)
        self.constellation = constellation
        self.subcarrier_freqs_hz = np.arange(f_min, f_max, subcarrier_spacing)
        self.preamble_method = preamble_method
        self.ofdm_symbol_length = ofdm_symbol_length
        self.cyclic_prefix_length = cyclic_prefix_length
        self.upsample_factor = upsample_factor
        self.subcarrier_freqs_hz = np.arange(f_min, f_max, subcarrier_spacing)
        self.subcarrier_indicies = (self.subcarrier_freqs_hz / subcarrier_spacing).astype(int)

    def transform(self, x:Signal) -> Signal:
        preamble = x.artifact_container['preamble']
        y_t = x.data
        corr = correlate(y_t, preamble, mode='valid')
        peaks, _ = find_peaks(corr, height=0.95 * np.max(np.abs(corr)), distance=len(preamble))
        if len(peaks) == 0:
            raise ValueError("Preamble not found in received signal")
        preamble_start = peaks[0]
        ofdm_symbol_start = preamble_start + len(preamble)
        ofdm_symbol_end = ofdm_symbol_start + self.ofdm_symbol_length
        ofdm_symbol = y_t[ofdm_symbol_start : ofdm_symbol_end]
        cyclic_prefix_removed = ofdm_symbol[self.cyclic_prefix_length :]

        # Take FFT and extract subcarriers
        Y_k = np.fft.fft(cyclic_prefix_removed, norm='ortho')[self.subcarrier_indicies]
        bits_estimated = self.constellation.symbols_to_bits(Y_k)
        x.artifact_container['estimated_bits'] = bits_estimated
        x.artifact_container['received_symbols'] = Y_k
        x.data = cyclic_prefix_removed
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