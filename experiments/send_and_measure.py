import os
from pathlib import Path
from pyflux.core.experiment import ExperimentalContext
from pyflux.core.block import Signal, ActionBlock, FunctionalBlock, ResamplingBlock
from pyflux.core.chain import Chain
import matplotlib.pyplot as plt
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

class MeasureSine(ActionBlock):
    def __init__(self, fs, osc_driver, input_signal_frequency):
        super().__init__(fs)
        self.driver = osc_driver
        self.input_signal_frequency = input_signal_frequency

    def action(self, x: Signal) -> Signal:
        self.driver.set_trigger(ch=1, voltage_level=0.5)
        self.driver.configure_channel(ch=3,
                                      scale=1, offset=0,
                                      seconds_per_div=1/self.input_signal_frequency)
        x.data, _ = self.driver.measure_waveform(channel=3)
        x.sampling_rate = self.driver.input_sample_rate

class DisplaySine(FunctionalBlock):
    def __init__(self, fs):
        super().__init__(fs)

    def transform(self, x):
        # Exp.log(f"{x.data, x.data.max().item(), x.data.min().item()}")
        plt.plot(x.data)
        plt.title("Measured Sine Waveform")
        plt.xlabel("Sample Number")
        plt.ylabel("Amplitude (ADC Units)")
        plt.grid()
        plt.show()
        return x

class PrintSomeNums(ActionBlock):
    def __init__(self, fs, mode: str, name=None):
        super().__init__(fs, mode, name)
    
    def action(self, x):
        pid = os.getpid()
        for i in range(3):
            time.sleep(4)
            print(f"{self.mode} {i} on PID: {pid}")

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

        Exp.log("Starting Sine sending!")

        fn_gen = Exp.Agilent_33250A
        osc = Exp.Tektronix_TDS2000

        main_pipe = Chain(chain=[
            SendSine(fs=fn_gen.output_sample_rate, freq=1e4, fn_generator_driver=fn_gen), # 10 kHz
            ResampleWaveform(fs_in=fn_gen.output_sample_rate, fs_out=osc.input_sample_rate),
            MeasureSine(fs=osc.input_sample_rate, osc_driver=osc, input_signal_frequency=1e4),
            DisplaySine(fs=osc.input_sample_rate)
        ])

        async_chain = Chain(chain=[
            PrintSomeNums(fs=None, mode="thread"),
            PrintSomeNums(fs=None, mode="process")
        ])

        for i in range(Exp.config.SINE_ITERS):
            x = Signal(data=None, sampling_rate=fn_gen.output_sample_rate)
            futures = async_chain.run_async(x, exp=Exp)
            main_pipe.run(x) 

        Exp.log("Hardware finished! Finishing processing tasks. . ")
        Exp.sync(futures)