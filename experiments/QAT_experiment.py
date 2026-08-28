import subprocess
import sys
import os
from pathlib import Path
from Execute_channel import Execute_Channel
from generate_Q88_time_frames import writeSentTime
from Plot_sv_constellation import create_plots

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.chdir(Path(__file__).resolve().parents[1])

sim_directory = "sv_tcn/tcn5"

DATA_WIDTH = 16
MODEL_TYPE = "decoder"
TEST = 8


if __name__ == "__main__":
    writeSentTime()

    result = subprocess.run(
        ["iverilog", "-g2012", "-P", f'tb.MODEL_TYPE="encoder"', "-P", f"tb.TEST={TEST}", "-P", f"tb.DATA_WIDTH={DATA_WIDTH}", "-o", "tb.vvp", "tb.sv"],
        cwd=sim_directory,
        capture_output=True,
        text=True,
    )

    result2 = subprocess.run(
        ["vvp", "tb.vvp"],
        cwd=sim_directory,
        capture_output=True,
        text=True,
    )

    Execute_Channel()

    result3 = subprocess.run(
        ["iverilog", "-g2012", "-P", f'tb.MODEL_TYPE="decoder"', "-P", f"tb.TEST={TEST}", "-P", f"tb.DATA_WIDTH={DATA_WIDTH}", "-o", "tb.vvp", "tb.sv"],
        cwd=sim_directory,
        capture_output=True,
        text=True,
    )

    result4 = subprocess.run(
        ["vvp", "tb.vvp"],
        cwd=sim_directory,
        capture_output=True,
        text=True,
    )

    create_plots(ShowTimeSeries=True)

    # Print status and output
    print(f"Return code: {result.returncode}")
    print(f"Textout:\n{result.stdout}")

    print(f"Return code: {result2.returncode}")
    print(f"Textout:\n{result2.stdout}")

    print(f"Return code: {result3.returncode}")
    print(f"Textout:\n{result3.stdout}")

    print(f"Return code: {result4.returncode}")
    print(f"Textout:\n{result4.stdout}")