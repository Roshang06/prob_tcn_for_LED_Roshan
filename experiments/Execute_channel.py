import torch
import os
import sys
from pathlib import Path
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.chdir(Path(__file__).resolve().parents[1])

from modules.grid_search.adapters import TCNAdapter

def q88_hex_to_float(hexa: str):
    v = int(hexa, 16)
    if v & 0x8000:  # If the sign bit is set (negative)
        v -= 0x10000
    return v/256

def float_to_q88_int(x: float) -> int:
    """Convert float → 16-bit Q8.8 signed integer."""
    scaled = int(round(x * 256))
    return max(-32768, min(32767, scaled)) # question: why is this capping it at 32768


def q88_int_to_hex(v: int) -> str:
    """Format a signed Q8.8 integer as 4-digit hex (two's complement)."""
    # Python's % operator handles negative numbers differently from C,
    # so we mask to 16 bits to get two's complement representation.
    return f"{v & 0xFFFF:04x}"

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


CHANNEL_PTH = f"prob_tcn_for_LED/data/experiments/test_real_gridsearch/channel_models_20260810_1210/runs/tcn_4712cb05"
SV_PTH = "realtime-microled/tcn4"

base_pth = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

if __name__ == "__main__":

    #model = torch.load(os.path.join(base_pth, CHANNEL_PTH))
    with open(os.path.join(base_pth, CHANNEL_PTH, "config.yaml"), "r") as file:
        sent_time = [[]]

        with open(os.path.join(base_pth, SV_PTH, "encoder_output.mem"), "r") as file:
            for line in file:
                line = line.strip()
                if line[0:2] != "//":
                    num = q88_hex_to_float(line)
                    sent_time[0].append(num)

        #rec_time_tensor = model.model((torch.tensor(sent_time)))

        #print(rec_time_tensor)

        recieved_time = send_through_channel(torch.tensor(sent_time)).detach().cpu().numpy().flatten()
        #print(recieved_time)
        words = []
        for sample in recieved_time:
            words.append(q88_int_to_hex(float_to_q88_int(sample)))
        save_mem_file(os.path.join(base_pth, SV_PTH, "recieved_time.mem"), words, f"Sent through the channel model")


