'''
The purpose of this script is to perform an end-to-end experiment to test the equalization performance of encoder/decoder equalizer models for
a variety of channel model architectures. 

Namely, a number of DC offsets, max parameter counts, and model families can be specified for a grid search. 

Then, this script initiates the automatic dataset collection, channel model training, and encoder/decoder training before
gathering final performance metrics on the original channel.
'''

import os
from pathlib import Path
from pyflux.core.experiment import ExperimentalContext
from pyflux.core.block import Signal, ActionBlock, FunctionalBlock, ResamplingBlock
from pyflux.core.chain import Chain
import numpy as np
from modules.experimental_blocks import *
from modules.constellation_diagram import QPSK_Constellation, get_constellation, RingShapedConstellation

# set the cwd to be parent folder and not workspace root
script_dir = Path(__file__).resolve().parent
os.chdir(script_dir)


if __name__ == '__main__':
    '''Without the main guard, every child process will rerun the code'''
    with ExperimentalContext(CONFIG_FILE="end_to_end_experiment.yml") as Exp:

        # Build the relevant blocks for the experiment
        
        # Setup instruments
        awg = Exp.Agilent_33250A
        osc = Exp.Tektronix_TDS2000
        pwr_supply = Exp.HP_EE3631A



        osc.set_record_length(20_000)
        osc.set_trigger(ch=1, voltage_level=1)

        osc.set_probe_gain(ch=1, gain=1)
        osc.set_probe_gain(ch=3, gain=1)

        VERTICAL_DIVS = osc.vertical_divisions

        osc.configure_channel(ch=3,
                              scale=Exp.config.DATA_COLLECTION.OSC_SCALE,
                              offset=0)
        osc.configure_channel(ch=1,
                              scale=1,
                              offset=0)
        
        for dc_offset_ma in Exp.config.DATA_COLLECTION.DC_OFFSETS:
            # Perform all collection and training with set DC offset
            pwr_supply.set_6V(
                voltage=4,
                current=dc_offset_ma
            )
            pwr_supply.enable_output()
            Exp.log(f"Starting Data Collection with DC offset of {dc_offset_ma} mA!")
