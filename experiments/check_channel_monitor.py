'''Probe the channel with CheckChannel on demand and print each reading to the
terminal. No dataset, no log file. Press Enter to take a reading, q to quit. Meant
for watching the LED response live while debugging the DC bias supply.'''
from pathlib import Path

from pyflux.core.experiment import ExperimentalContext
from modules.experimental_blocks import CheckChannel

HERE = Path(__file__).resolve().parent
CONFIG_FILE = HERE / "train_and_validate.yml"

DATA_CHANNEL = 3
TRIGGER_CHANNEL = 1




if __name__ == "__main__":
    with ExperimentalContext(CONFIG_FILE=CONFIG_FILE, create_log_file=False) as Exp:
        cfg = Exp.config.DATA_COLLECTION

        awg = Exp.Agilent_33250A
        osc = Exp.Tektronix_TDS2000
        pwr_supply = Exp.HP_EE3631A

        awg.set_output_load("INF")
        osc.display_channel(ch=TRIGGER_CHANNEL)
        osc.display_channel(ch=DATA_CHANNEL)
        osc.set_trigger(ch=TRIGGER_CHANNEL, voltage_level=0)
        osc.set_probe_gain(ch=TRIGGER_CHANNEL, gain=1)
        osc.set_probe_gain(ch=DATA_CHANNEL, gain=1)
        osc.configure_channel(ch=TRIGGER_CHANNEL, scale=1, offset=0)
        osc.configure_channel(ch=DATA_CHANNEL, scale=float(cfg.OSC_SCALE), offset=0)
        osc.set_coupling(ch=TRIGGER_CHANNEL, coupling="AC")
        osc.set_coupling(ch=DATA_CHANNEL, coupling="AC")

        dc_offset_A = float(cfg.DC_OFFSETS[0])
        pwr_supply.set_25V(voltage=4, current=dc_offset_A)
        pwr_supply.enable_output()
        print(f"DC supply commanded: P25V current-limit {dc_offset_A:.3f} A, output ON")

        check_channel = CheckChannel(awg_driver=awg, osc_driver=osc, data_channel=DATA_CHANNEL)

        print("press Enter to probe the channel, q then Enter to quit")
        try:
            while input("> ").strip().lower() != "q":
                check_channel.run()
        except KeyboardInterrupt:
            pass
        print("\nstopping monitor (supply left as-is)")
