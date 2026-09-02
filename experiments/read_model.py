import torch
import numpy
import os
import math
import sys
from pathlib import Path
from modules.utils import q88_int_to_hex, float_to_q88_int

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.chdir(Path(__file__).resolve().parents[1])

def save_mem_file(path: str, values: list, comment: str = ""):
    """Save a list of Q8.8 integers to a .mem file (hex, one per line)."""
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)

    with open(path, "w") as f:
        if comment:
            f.write(f"// {comment}\n")
        for v in values:
            f.write(v + "\n")
    print(f"  Saved {len(values)} values → {path}")

def read_model(read_pth, save_pth, datawidth):
    output_dir = os.path.join(base_pth, save_pth)
    os.makedirs(output_dir, exist_ok=True)
    model = torch.load(os.path.join(base_pth, read_pth), weights_only=True)

    for type in ["encoder", "decoder"]:
        e_or_d = model[type]
        keys = e_or_d.keys()
        print(f"{type} keys: {keys}")
        for key in keys:
            print(f"Keyname: {key}")

            tensor = e_or_d[key]
            for i, hidden_channel in enumerate(tensor):
                print(f"    Channel{i}:")
                #print(hidden_channel)

                arr = hidden_channel.detach().cpu().numpy().flatten()
                save_arr = []
                for num in arr:
                    hexa = q88_int_to_hex(float_to_q88_int(num, datawidth), datawidth)
                    print(f"        Weight/Bias: {num} Hex: {hexa}")
                    save_arr.append(hexa)

                safe_key = key.replace('.', '_')
                save_mem_file(os.path.join(output_dir, f"{type}", safe_key, f"channel{i}.mem"), save_arr, f"contains {len(arr)} values")

TEST = 1
READ_PATH = "prob_tcn_for_LED/data/experiments/test_real_gridsearch/encoder_decoder_20260824_1457/runs/tcn_ae_6f25ae0d/model.pt" # file path of the model.pt
SAVE_PATH = f"prob_tcn_for_LED/sv_tcn/tcn5/TestingData/Test{TEST}" # relative
DATA_WIDTH = 16

base_pth = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

if __name__ == "__main__":
    read_model(READ_PATH, SAVE_PATH, DATA_WIDTH)