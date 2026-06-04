import os
from pathlib import Path
from pyflux.core.experiment import ExperimentalContext
from pyflux.core.block import Signal, ActionBlock, FunctionalBlock, ResamplingBlock
from pyflux.core.chain import Chain
import matplotlib.pyplot as plt
import numpy as np
import time


# set the cwd to be parent folder and not workspace root
script_dir = Path(__file__).resolve().parent
os.chdir(script_dir)

# create blocks for sending sine wave a gathering waveform
class SendSine(ActionBlock):
    def __init__(self, fs, freq, fn_generator_driver):
        super().__init__(fs)
        self.driver = fn_generator_driver
        self.freq = freq
    
    def action(self, x: Signal):
        self.driver.apply_output_fun(
            waveform="SIN",
            frequency=self.freq,
            amplitude="1.0",
            offset="0.0")
        

class SendArbWaveform(ActionBlock):
    def __init__(self, fs, fn_generator_driver, waveform_data: np.ndarray, freq: float,
                 amplitude: float, offset: float):
        super().__init__(fs)
        self.driver = fn_generator_driver
        self.waveform_data = waveform_data
        self.freq = freq
        self.amplitude = amplitude
        self.offset = offset
        assert len(waveform_data) <= 16_384, "Agilent 33250A max arbitrary waveform length is 16384 points"
    
    def action(self, x: Signal):
        self.driver.send_arbitrary(
            samples=self.waveform_data,
            freq=self.freq,
            amplitude=self.amplitude,
            offset=self.offset)
        # print(f"Errors: {self.driver.drain_errors()}")
        
class MeasureSine(ActionBlock):
    def __init__(self, fs, osc_driver, input_signal_frequency, channel):
        super().__init__(fs)
        self.driver = osc_driver
        self.input_signal_frequency = input_signal_frequency
        self.channel = channel

    def action(self, x: Signal) -> Signal:
        _, x.data = self.driver.measure_waveform(channel=self.channel)
        x.sampling_rate = self.driver.input_sample_rate

class DisplayWaveform(FunctionalBlock):
    def __init__(self, fs):
        super().__init__(fs)

    def transform(self, x):
        plt.plot(x.data)
        plt.title("Measured Waveform")
        plt.xlabel("Sample Number")
        plt.ylabel("Amplitude (Volts)")
        plt.grid()
        plt.show()
        return x

class ResampleWaveform(ResamplingBlock):
    def __init__(self, fs_in, fs_out):
        super().__init__(fs_in, fs_out)

    def resample(self, x: Signal) -> Signal:
        '''Pass here as sending sine does not produce a signal with data,
           so resampling is not needed. In a real use case, this block would
           resample the input signal from fs_in to fs_out.
        '''
        x.sampling_rate = self.fs_out
        return x

if __name__ == '__main__':
    '''Without the main guard, every child process will rerun the code'''
    with ExperimentalContext(CONFIG_FILE="send_and_measure.yml") as Exp:

        FREQS = [1e4, 2e4, 5e4] # Hz

        fn_gen = Exp.Agilent_33250A
        osc = Exp.Tektronix_TDS2000
    
        osc.set_record_length(20_000)
        osc.set_trigger(ch=1, voltage_level=1)

    
        osc.set_probe_gain(ch=3, gain=1)
        osc.set_probe_gain(ch=1, gain=1)

        VERTICAL_DIVS = osc.vertical_divisions
        SINE_VPKPK = 2

        osc.configure_channel(ch=3,
                            scale=SINE_VPKPK/VERTICAL_DIVS, offset=0)
        osc.configure_channel(ch=1,
                            scale=1, offset=0)

        for f in FREQS:

            Exp.log(f"Starting Sine sending at {f}!")

            osc.set_horizontal_scale(seconds_per_div=0.1*(1/f))
            N = 16384  # max arbitrary waveform length for Agilent 33250A
            t = np.linspace(0, 1, N, endpoint=False)
            waveform_data = 1 - 2 * np.abs(2 * t - 1)

            
            main_pipe = Chain(chain=[
                SendArbWaveform(fs=fn_gen.output_sample_rate,
                              freq=f,
                              fn_generator_driver=fn_gen,
                              waveform_data=waveform_data,
                              amplitude=1.0,
                              offset=0.0
                              ), # 10 kHz
                ResampleWaveform(fs_in=fn_gen.output_sample_rate, fs_out=osc.input_sample_rate),
                MeasureSine(fs=osc.input_sample_rate, osc_driver=osc, input_signal_frequency=f, channel=3),
                DisplayWaveform(fs=osc.input_sample_rate)
            ])
            x = Signal(data=None, sampling_rate=fn_gen.output_sample_rate)
            main_pipe.run(x) 