'''
The purpose of this script is to perform an end-to-end experiment to test the equalization performance of encoder/decoder equalizer models for
a variety of channel model architectures. 

Namely, a number of DC offsets, max parameter counts, and model families can be specified for a grid search. 

Then, this script initiates the automatic dataset collection, channel model training, and encoder/decoder training before
gathering final performance metrics on the original channel.
'''

import os
import time
from pathlib import Path
from pyflux.core.experiment import ExperimentalContext
from pyflux.core.block import Signal, ActionBlock, FunctionalBlock, ResamplingBlock
from pyflux.core.chain import Chain
import numpy as np
import matplotlib.pyplot as plt
from modules.experimental_blocks import *
from modules.constellation_diagram import QPSK_Constellation, get_constellation, RingShapedConstellation

# set the cwd to be parent folder and not workspace root
script_dir = Path(__file__).resolve().parent
os.chdir(script_dir)

'''Setup Experimental Blocks'''

'''OFDM Blocks'''

if __name__ == '__main__':
    '''Without the main guard, every child process will rerun the code'''
    with ExperimentalContext(CONFIG_FILE="end_to_end_experiment.yml") as Exp:

        # Build the relevant blocks for the experiment
        
        # Setup instruments
        awg = Exp.Agilent_33250A
        AWG_WAVEFORM_LENGTH = 16_384 # The Agilent 33250A has a maximum arbitrary waveform length of 16,384 samples
        osc = Exp.Tektronix_TDS2000
        pwr_supply = Exp.HP_EE3631A

        awg.set_output_load("INF") # Set output load to infinite
        osc.display_channel(ch=1)
        osc.display_channel(ch=3)
        osc.set_record_length(20_000)
        osc.set_trigger(ch=1, voltage_level=0)

        osc.set_probe_gain(ch=1, gain=1)
        osc.set_probe_gain(ch=3, gain=1)

        VERTICAL_DIVS = osc.vertical_divisions

        osc.configure_channel(ch=3,
                              scale=Exp.config.DATA_COLLECTION.OSC_SCALE,
                              offset=0)
        osc.configure_channel(ch=1,
                              scale=1,
                              offset=0)
        
        osc.set_coupling(ch=1, coupling="AC")
        osc.set_coupling(ch=3, coupling="AC")
     
        MEASURED_A_OFFSET = 0.002 # The stable current is consistently 2 mA lower than what is set
        dc_offsets = Exp.config.DATA_COLLECTION.DC_OFFSETS
        min_subcarrier_freqs = Exp.config.DATA_COLLECTION.F_MINS
        max_subcarrier_freqs = Exp.config.DATA_COLLECTION.F_MAXS
        osc_sample_rates = Exp.config.DATA_COLLECTION.OSC_SAMPLE_RATES
        subcarrier_spacing = float(Exp.config.DATA_COLLECTION.SUBCARRIER_SPACING)
        upsample_factor = Exp.config.DATA_COLLECTION.UPSAMPLE_FACTOR
        awg_table_fraction = Exp.config.DATA_COLLECTION.AWG_TABLE_FRACTION
        cp_length_fraction = Exp.config.DATA_COLLECTION.CP_LENGTH_FRACTION
        preamble_length = Exp.config.DATA_COLLECTION.PREAMBLE_LENGTH

        for dc_offset_A, min_freq, max_freq, osc_fs in zip(dc_offsets, min_subcarrier_freqs, max_subcarrier_freqs, osc_sample_rates):
            dc_offset_A = float(dc_offset_A)
            min_freq = float(min_freq)
            max_freq = float(max_freq)
            osc_fs = float(osc_fs)
            # Perform all collection and training with set DC offset
            print(f"Setting DC offset to {dc_offset_A - MEASURED_A_OFFSET: .2f}: A")
            pwr_supply.set_6V(
                voltage=4, # It will never reach 4V because of current limit
                current=dc_offset_A
            )
            # Wait 5 seconds for constant current on the power supply to stabilize before collecting data
            pwr_supply.enable_output()
            Exp.log(f"Starting Data Collection with DC offset of {dc_offset_A - MEASURED_A_OFFSET: .2f} A!")

            # Configure OFDM settings at these DC offsets
            num_carriers = len(np.arange(min_freq, max_freq, subcarrier_spacing))
            num_leading_zero_subcarriers = len(np.arange(subcarrier_spacing, min_freq, subcarrier_spacing))
            baseband_ofdm_symbol_length = upsample_factor * 2 * (num_carriers + num_leading_zero_subcarriers + 1)

            mod_ofdm = ModulateDataOFDM(
                constellation=get_constellation("qpsk"),
                f_min=min_freq,
                f_max=max_freq,
                subcarrier_spacing=subcarrier_spacing,
                preamble_method="zadoff_chu",
                awg_table_fraction=awg_table_fraction,
                cyclic_prefix_fraction=cp_length_fraction,
                upsample_factor=upsample_factor,
                preamble_length=preamble_length
            )

            f_AWG = mod_ofdm.awg_frequency
            Exp.log(f"AWG frequency for this experiment will be {f_AWG: .2f} Hz")
            T_AWG = 1 / f_AWG
            osc.set_horizontal_scale(0.2 * T_AWG)
            send_waveform = SendWaveform(
                fs=mod_ofdm.fs_out,
                awg_driver=awg,
                freq=f_AWG,
                amplitude=19.9, # just under the 33250A's 20 Vpp high-Z ceiling
                offset=0
            )

            measure_waveform = MeasureWaveform(
                fs_in=mod_ofdm.fs_out,
                fs_out=osc_fs,
                osc_driver=osc,
                input_signal_frequency=f_AWG,
                trigger_channel=1,
                data_channel=3,
                debug=True
            )

            resample_waveform = ResampleMeasuredWaveform(
                fs_in=osc_fs,
                fs_out=mod_ofdm.baseband_sampling_rate
            )

            demod_ofdm = DemodulateDataOFDM(
                constellation=get_constellation("qpsk"),
                f_min=min_freq,
                f_max=max_freq,
                subcarrier_spacing=subcarrier_spacing,
                preamble_method="zadoff_chu",
                baseband_fft_length=mod_ofdm.baseband_fft_length,
                cyclic_prefix_length=mod_ofdm.cyclic_prefix_length,
                upsample_factor=upsample_factor,
                debug=True
            )

            send_and_receive_ofdm = SendAndReceiveOFDM(mod_ofdm,
                                                       send_waveform,
                                                       measure_waveform,
                                                       resample_waveform,
                                                       demod_ofdm)
            
            seed = Signal(data=np.zeros(1), sampling_rate=mod_ofdm.fs_out)
            x = send_and_receive_ofdm.run(seed)
            Exp.log(f"Shape of received symbol frame: {x.data.shape}, sampling rate: {x.sampling_rate: .2f} Hz")
            PlotConstellations().run(x)

            # Plot waveform
            plt.plot(x.data)
            plt.title(f"Received OFDM Waveform with DC offset of {dc_offset_A - MEASURED_A_OFFSET} A")
            plt.xlabel("Sample Index")
            plt.ylabel("Voltage (V)")
            plt.grid()
            plt.show()