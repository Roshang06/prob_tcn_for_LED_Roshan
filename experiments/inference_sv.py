import subprocess
import sys
import os
from pathlib import Path
import torch
import json
from Execute_channel import Execute_Channel
from generate_Q88_time_frames import writeSentTime
from Plot_sv_constellation import create_plots
from Plot_EVM_vs_BitWidth import plot_evm_vs_bitwidth
from QAT_experiment import read_model

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.chdir(Path(__file__).resolve().parents[1])
base_pth = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

sim_directory = "sv_tcn/tcn6"
DATA_WIDTH = 16
TEST = 20
SAVE_PATH = f"prob_tcn_for_LED/{sim_directory}"
ED_MODEL = "prob_tcn_for_LED/data/experiments/test_real_gridsearch/encoder_decoder_20260825_1207/runs/tcn_ae_6f25ae0d"


if __name__ == "__main__":
    writeSentTime(SAVE_PATH, DATA_WIDTH)
    read_model(os.path.join(base_pth, ED_MODEL, "model.pt"), os.path.join(base_pth, SAVE_PATH, "TestingData", f"Test{TEST}"), DATA_WIDTH)
    # print("Reading to:")
    # print(os.path.join(base_pth, GRID_SEARCH, "runs", run["run_id"], "model.pt"))

    result = subprocess.run(
        ["iverilog", "-g2012", "-P", f'tb.MODEL_TYPE="encoder"', "-P", f"tb.TEST={TEST}", "-P", f"tb.DATA_WIDTH={DATA_WIDTH}", "-P", f"tb.SAMPLES={3760}", "-o", "tb.vvp", "tb.sv"],
        cwd=sim_directory,
        capture_output=True,
        text=True,
        check=True,
    )

    result2 = subprocess.run(
        ["vvp", "tb.vvp"],
        cwd=sim_directory,
        capture_output=True,
        text=True,
        check=True,
    )

    Execute_Channel(DATA_WIDTH)

    result3 = subprocess.run(
        ["iverilog", "-g2012", "-P", f'tb.MODEL_TYPE="decoder"', "-P", f"tb.TEST={TEST}", "-P", f"tb.DATA_WIDTH={DATA_WIDTH}", "-P", f"tb.SAMPLES={3760}", "-o", "tb.vvp", "tb.sv"],
        cwd=sim_directory,
        capture_output=True,
        text=True,
        check=True,
    )

    result4 = subprocess.run(
        ["vvp", "tb.vvp"],
        cwd=sim_directory,
        capture_output=True,
        text=True,
        check=True,
    )

    create_plots(DATA_WIDTH, ShowTimeSeries=False, ed_model_pth=os.path.join(base_pth, ED_MODEL))