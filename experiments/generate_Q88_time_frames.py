import zarr
import torch
import os
import sys
from pathlib import Path
import numpy as np
import math
# Ensure repo root is on the path regardless of launch location
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.chdir(Path(__file__).resolve().parents[1])

from modules.grid_search import EncoderDecoderGridSearch
from modules.utils import q88_int_to_hex, float_to_q88_int
from test_real_gridsearch import (
    OFDM_CONFIG, ENCODER_DECODER_GRID
)

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

def read_dataset(path, n_frames: int = 64) -> torch.Tensor:#tuple[torch.Tensor, torch.Tensor]:
    sent_arr = zarr.open(path, mode='r')
    #recieved_arr = zarr.open("prob_tcn_for_LED/data/dc0.052A_fmin300000_fmax7.6e+06_20260630_1743.zarr/received_burst", mode='r')
    """
    Shape: (5000, 940)
    Data Type: float32
    Chunks: (64, 940)
    Subset shape: (100, 940)
    """
    sent_symbols = torch.from_numpy(sent_arr[:n_frames, :]).float()
    #recv_symbols = torch.from_numpy(recieved_arr[:n_frames, :]).float()

    return sent_symbols#, recv_symbols



READ_PATH = "prob_tcn_for_LED/data/dc0.052A_fmin300000_fmax7.6e+06_20260630_1743.zarr/sent_burst"
SAVE_PATH = f"prob_tcn_for_LED/sv_tcn/tcn5"
WAVEFORMS = 4
DATA_WIDTH = 16
TYPE = "Synthetic"# Synthetic, Real, Step
base_pth = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

ed_gs = EncoderDecoderGridSearch(
            ENCODER_DECODER_GRID,
            channel_models=[],
            dataset_path="nothing",
)

def create_sent_time():
    if TYPE == "Synthetic":
        return ed_gs._sample_batch(WAVEFORMS, len(OFDM_CONFIG.active_carrier_indices)*2, OFDM_CONFIG, "please add the preamble")[1]
    elif TYPE == "Real":
        return read_dataset(os.path.join(base_pth, READ_PATH), n_frames=WAVEFORMS)
    elif TYPE == "Step":
        length = 940
        # 4 cycles (Frequency x4)
        single_waveform = (((torch.arange(length) * 2 * 1) // length) % 2 * 2.0 - 1.0).to(torch.float32)
        return single_waveform.repeat(WAVEFORMS, 1)
    elif TYPE == "Random Step":
        length = 940
        values = torch.tensor([0.0, 1.0, -1.0, 2.0, -3.0, 3.0], dtype=torch.float32)
        time_series = torch.empty((WAVEFORMS, length), dtype=torch.float32)

        for burst_idx in range(WAVEFORMS):
            position = 0
            while position < length:
                remaining = length - position
                min_step = min(80, remaining)
                max_step = min(300, remaining)
                step_length = int(torch.randint(min_step, max_step + 1, ()).item())
                value = values[torch.randint(0, len(values), ())].item()
                time_series[burst_idx, position:position + step_length] = value
                position += step_length

        return time_series
    else:
        raise ValueError("TYPE was not correctly specified.")

def writeSentTime(save_pth, datawidth):
    sent_time= create_sent_time()

    time_series = []
    tensor_time_series = sent_time.detach().cpu().numpy()
    for burst in tensor_time_series:    
            for point in burst:
                time_series.append(q88_int_to_hex(float_to_q88_int(point, data_width=datawidth), data_width=datawidth))

    save_mem_file(os.path.join(base_pth, save_pth, "input_time_series.mem"), time_series, "OFDM modulated time series - includes cyclic prefix and preamble")

if __name__ == "__main__":


    #print(OFDM_CONFIG.cyclic_prefix_length)
    #print(ed_gs.preamble_length)
    sent_time= create_sent_time()
    print(f"Dataset Shape: Sent{tuple(sent_time.shape)}")#  Rec{tuple(recieved.shape)}")

    #sent_time = symbols_to_time(sent, num_left_padding_zeros=0, num_right_padding_zeros=0) # padding varies the number of time series points.
    #rec_time = symbols_to_time(recieved, num_left_padding_zeros=0, num_right_padding_zeros=0)
    #print(f"Time Series Shape: Sent{tuple(sent_time.shape)}")#  Rec{tuple(rec_time.shape)}")

    time_series = []
    tensor_time_series = sent_time.detach().cpu().numpy()

    for burst in tensor_time_series:    
         for point in burst:
              time_series.append(q88_int_to_hex(float_to_q88_int(point, DATA_WIDTH), DATA_WIDTH))

    save_mem_file(os.path.join(base_pth, SAVE_PATH, "input_time_series.mem"), time_series, "OFDM modulated time series - includes cyclic prefix and preamble")
    