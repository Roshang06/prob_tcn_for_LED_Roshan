import torch
import os
import sys
from pathlib import Path
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.chdir(Path(__file__).resolve().parents[1])

from modules.grid_search.adapters import TCNAdapter
from modules.utils import q88_int_to_hex, q88_hex_to_float, float_to_q88_int
from generate_Q88_time_frames import DATA_WIDTH, SAVE_PATH

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

def send_through_channel(sent_time):
    with open(os.path.join(base_pth, CHANNEL_PTH, "config.yaml"), "r") as file:
            config = yaml.safe_load(file)
            params = config["params"]
            #print(f"Model config: {params}")
            adapter = TCNAdapter.load(params=params, checkpoint=os.path.join(base_pth, CHANNEL_PTH, "model.pt"), device="cpu")
    
            rec_time_tensor = adapter.model(sent_time)
    
            #print(rec_time_tensor)
    
            return rec_time_tensor[0] # 0 for noise, 1 for mean

def Execute_Channel(data_width):
    with open(os.path.join(base_pth, CHANNEL_PTH, "config.yaml"), "r") as file:
        sent_time = [[]]

        with open(os.path.join(base_pth, SAVE_PATH, "encoder_output.mem"), "r") as file:
            for line in file:
                line = line.strip()
                if line[0:2] != "//":
                    num = q88_hex_to_float(line, data_width)
                    sent_time[0].append(num)

        recieved_time = send_through_channel(torch.tensor(sent_time)).detach().cpu().numpy().flatten()

        words = []
        for sample in recieved_time:
            words.append(q88_int_to_hex(float_to_q88_int(sample, data_width), data_width))
        save_mem_file(os.path.join(base_pth, SAVE_PATH, "recieved_time.mem"), words, f"Sent through the channel model")


CHANNEL_PTH = "prob_tcn_for_LED/data/experiments/test_real_gridsearch/channel_models_20260824_1456/runs/tcn_b338d2dc"

base_pth = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

if __name__ == "__main__":

    #model = torch.load(os.path.join(base_pth, CHANNEL_PTH))
    with open(os.path.join(base_pth, CHANNEL_PTH, "config.yaml"), "r") as file:
        sent_time = [[]]

        with open(os.path.join(base_pth, SAVE_PATH, "encoder_output.mem"), "r") as file:
            for line in file:
                line = line.strip()
                if line[0:2] != "//":
                    num = q88_hex_to_float(line, DATA_WIDTH)
                    sent_time[0].append(num)

        #rec_time_tensor = model.model((torch.tensor(sent_time)))

        #print(rec_time_tensor)

        recieved_time = send_through_channel(torch.tensor(sent_time)).detach().cpu().numpy().flatten()
        #print(recieved_time)
        words = []
        for sample in recieved_time:
            words.append(q88_int_to_hex(float_to_q88_int(sample, DATA_WIDTH), DATA_WIDTH))
        save_mem_file(os.path.join(base_pth, SAVE_PATH, "recieved_time.mem"), words, f"Sent through the channel model")


