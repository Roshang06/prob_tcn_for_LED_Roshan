'''Throwaway diagnostic: load the REAL OFDM burst as an arb and sweep amplitude to
find the largest value the 33250A accepts without the -222 clip. Confirms the
amplitude is limited by the asymmetric waveform's DAC span, not the output load.
Run:  python experiments/_diag_awg_load.py   (delete when done)
'''
import numpy as np
from pathlib import Path
from pyflux.core.experiment import ExperimentalContext
from pyflux.core.block import Signal
from modules.experimental_blocks import ModulateDataOFDM
from modules.constellation_diagram import get_constellation

CONFIG_FILE = Path(__file__).resolve().parent / "train_and_validate.yml"


if __name__ == "__main__":
    with ExperimentalContext(CONFIG_FILE=CONFIG_FILE) as Exp:
        cfg = Exp.config.DATA_COLLECTION
        awg = Exp.Agilent_33250A

        mod = ModulateDataOFDM(
            constellation=get_constellation("qpsk"),
            f_min=float(cfg.F_MINS[0]), f_max=float(cfg.F_MAXS[0]),
            subcarrier_spacing=float(cfg.SUBCARRIER_SPACING),
            preamble_method="zadoff_chu",
            awg_table_fraction=cfg.AWG_TABLE_FRACTION,
            cyclic_prefix_fraction=cfg.CP_LENGTH_FRACTION,
            upsample_factor=cfg.UPSAMPLE_FACTOR,
            preamble_length=cfg.PREAMBLE_LENGTH)

        x = mod.run(Signal(data=np.zeros(1), sampling_rate=mod.fs_out))
        wf = np.asarray(x.artifact_container["awg_waveform"], float)
        dac = np.int16(np.round(wf / np.max(np.abs(wf)) * 2047))[:16384]
        ptp = int(dac.max()) - int(dac.min())
        print(f"dac span: [{int(dac.min())}, {int(dac.max())}]  ptp={ptp}/4094  "
              f"=> full-scale-equiv for 19.9Vpp = {19.9 * 4094 / ptp:.2f} Vpp")

        awg.set_output_load("INF")
        awg.resource.clear(); awg.write("*CLS"); awg.write("FORM:BORD SWAP")
        awg.resource.write_binary_values("DATA:DAC VOLATILE, ", dac, datatype='h', is_big_endian=False)
        awg.query("*OPC?")
        awg.write("FUNC:USER VOLATILE")
        awg.write("OUTP:LOAD INF")
        print("VOLT?MAX with real arb loaded:", awg.query("VOLT? MAX").strip())

        for amp in (19.9, 19.0, 18.6, 18.0, 17.0):
            awg.write("*CLS")
            awg.write(f"APPL:USER {mod.awg_frequency}, {amp}, 0")
            err = awg.query("SYST:ERR?").strip()
            volt = awg.query("VOLT?").strip()
            print(f"  APPL amp={amp:5.1f} -> VOLT?={volt}  ERR={err}")
