"""
This tests the SystemVerilog TCN on a prototype QAT gridsearch.

For each run in the grid search:
1. Read the model.pt file into the sim directory
2. generate time frames as a .mem file -> input_time_series.mem
3. Run the encoder, channel model, decoder
4. create a constellation plot and record the performance of the model (EVM)

Create a plot of bitwidth vs EVM

Note* This takes a really long time because of the icarus verilog simulation
"""
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
from modules.utils import q88_int_to_hex, float_to_q88_int

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.chdir(Path(__file__).resolve().parents[1])
base_pth = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def save_mem_file(path: str, values: list, comment: str = "", printDebug=True):
    """Save a list of Q8.8 integers to a .mem file (hex, one per line)."""
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)

    with open(path, "w") as f:
        if comment:
            f.write(f"// {comment}\n")
        for v in values:
            f.write(v + "\n")

    if printDebug:
        print(f"  Saved {len(values)} values → {path}")


def read_model(read_pth, save_pth, datawidth):
    output_dir = os.path.join(base_pth, save_pth)
    os.makedirs(output_dir, exist_ok=True)
    model = torch.load(os.path.join(base_pth, read_pth), weights_only=True)

    for type in ["encoder", "decoder"]:
        e_or_d = model[type]
        keys = e_or_d.keys()
        #print(f"{type} keys: {keys}")
        for key in keys:
            #print(f"Keyname: {key}")

            tensor = e_or_d[key]
            for i, hidden_channel in enumerate(tensor):
                #print(f"    Channel{i}:")
                #print(hidden_channel)

                arr = hidden_channel.detach().cpu().numpy().flatten()
                save_arr = []
                for num in arr:
                    hexa = q88_int_to_hex(float_to_q88_int(num, datawidth), datawidth)
                    #print(f"        Weight/Bias: {num} Hex: {hexa}")
                    save_arr.append(hexa)

                safe_key = key.replace('.', '_')
                save_mem_file(os.path.join(output_dir, f"{type}", safe_key, f"channel{i}.mem"), save_arr, f"contains {len(arr)} values", printDebug=False)

sim_directory = "sv_tcn/tcn5"
SAVE_PATH = f"prob_tcn_for_LED/{sim_directory}"
GRID_SEARCH = "prob_tcn_for_LED/data/experiments/test_real_gridsearch/encoder_decoder_20260825_1207"


if __name__ == "__main__":
    plot_dict = {}

    # read each ed model in the testgridsearch
    with open(os.path.join(base_pth, GRID_SEARCH, "runs.jsonl"), "r", encoding='utf-8') as file:
        for line in file:
            if line.strip():
                run = json.loads(line)
                print(run["run_id"], ":", run["quantization"])

                if run["quantization"] == None:
                    quantization = 8
                    testNum = 18
                    save_pth = f"prob_tcn_for_LED/sv_tcn/tcn5/TestingData/Test{testNum}"
                else:
                    quantization = run["quantization"]
                    testNum = quantization
                    save_pth = f"prob_tcn_for_LED/sv_tcn/tcn5/TestingData/Test{testNum}"
                
                writeSentTime(SAVE_PATH, quantization*2)
                read_model(os.path.join(base_pth, GRID_SEARCH, "runs", run["run_id"], "model.pt"), save_pth, quantization*2)
                # print("Reading to:")
                # print(os.path.join(base_pth, GRID_SEARCH, "runs", run["run_id"], "model.pt"))

                result = subprocess.run(
                    ["iverilog", "-g2012", "-P", f'tb.MODEL_TYPE="encoder"', "-P", f"tb.TEST={testNum}", "-P", f"tb.DATA_WIDTH={quantization*2}", "-P", f"tb.SAMPLES={3760}", "-o", "tb.vvp", "tb.sv"],
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
            
                Execute_Channel(quantization*2)
            
                result3 = subprocess.run(
                    ["iverilog", "-g2012", "-P", f'tb.MODEL_TYPE="decoder"', "-P", f"tb.TEST={testNum}", "-P", f"tb.DATA_WIDTH={quantization*2}", "-P", f"tb.SAMPLES={3760}", "-o", "tb.vvp", "tb.sv"],
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
            
                sv, py = create_plots(quantization*2, ShowTimeSeries=False, ed_model_pth=os.path.join(base_pth, GRID_SEARCH, "runs", run["run_id"]))

                plot_dict[run["run_id"]] = [quantization, sv, py]

        print(plot_dict)

        bit_widths = []
        py_evm = []
        sv_evm = []
        for key in plot_dict:
            bit_widths.append(plot_dict[key][0]*2)
            py_evm.append(plot_dict[key][2])
            sv_evm.append(plot_dict[key][1])

        plot_evm_vs_bitwidth(bit_widths, sv_evm, py_evm, out_path=os.path.join(sim_directory, "plots"))
