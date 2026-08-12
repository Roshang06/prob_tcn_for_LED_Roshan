import torch
import os
import sys
from pathlib import Path
import yaml
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.chdir(Path(__file__).resolve().parents[1])

from modules.models import TCN
from modules.utils import evm_pct, calculate_per_burst_rrmse_pct_loss   
from modules.grid_search.adapters import TCNAdapter
from modules.grid_search import EncoderDecoderGridSearch, base
from test_real_gridsearch import (
    OFDM_CONFIG, ENCODER_DECODER_GRID
)
from generate_Q88_time_frames import WAVEFORMS, read_dataset, READ_PATH, create_sent_time
from Execute_channel import send_through_channel
from modules.experimental_blocks import band_limited_zc_preamble

def q88_hex_to_float(hexa: str):
    v = int(hexa, 16)
    if v & 0x8000:  # If the sign bit is set (negative)
        v -= 0x10000
    return v/256

ed_gs = EncoderDecoderGridSearch(
            ENCODER_DECODER_GRID,
            channel_models=[],
            dataset_path="nothing",
)

ARCH_KEYS = ("nlayers", "dilation_base", "kernel_size", "hidden_channels", "activation")

#CHANNEL_PTH = f"prob_tcn_for_LED/data/experiments/test_real_gridsearch/channel_models_20260707_2205/runs/tcn_4712cb05"
ED_MODEL_PTH = f"prob_tcn_for_LED/data/experiments/test_real_gridsearch/encoder_decoder_20260810_1212/runs/tcn_ae_9efa5918"
SV_PTH = "realtime-microled/tcn4"
FILES = ["input_time_series.mem", "encoder_output.mem", "recieved_time.mem", "decoder_output.mem"]

base_pth = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

if __name__ == "__main__":
    py_freq = []
    with open(os.path.join(base_pth, ED_MODEL_PTH, "config.yaml"), "r") as file:
        config = yaml.safe_load(file)
        p = config["params"]
        arch = {k: p[k] for k in ARCH_KEYS}
        encoder = TCN(**arch).to("cpu")
        checkpoint = torch.load(os.path.join(base_pth, ED_MODEL_PTH, "model.pt"), map_location="cpu", weights_only=True)
        encoder.load_state_dict(checkpoint["encoder"])
        encoder.eval()

        #ed_gs.preamble = torch.tensor(band_limited_zc_preamble(256, OFDM_CONFIG.subcarrier_spacing*OFDM_CONFIG.baseband_fft_length, float(300000.0), float(7600000.0), 3.0), dtype=torch.float32, device="cpu").unsqueeze(0)
        in_time = create_sent_time()
        #preamble, symbol = in_time[:, :256], in_time[:, 256:]
        py_freq.append(ed_gs._frame_to_freq(in_time, OFDM_CONFIG))


        outp = encoder(in_time)
        
        py_freq.append(ed_gs._frame_to_freq(outp, OFDM_CONFIG))
        recieved_time = send_through_channel(outp)
        py_freq.append(ed_gs._frame_to_freq(recieved_time, OFDM_CONFIG))

        decoder = TCN(**arch).to("cpu")
        decoder.load_state_dict(checkpoint["decoder"])
        decoder.eval()

        decoded_time = decoder(recieved_time)
        #decoded_time = _decode_freq(encoder.eval(), decoder.eval(), channel_model, eval_sent_time, ofdm_config)
        
        py_freq.append(ed_gs._frame_to_freq(decoded_time, OFDM_CONFIG))

        #print(f"input av power: {torch.square(torch.mean(in_time))}")
        #print(f"encoded av power: {torch.square(torch.mean(outp))}")
        #print(f"recieved av power: {torch.square(torch.mean(recieved_time))}")
        #print(f"decoded av power: {torch.square(torch.mean(decoded_time))}")

        #print(f"encoded average: {torch.mean(outp)}")
        #print(f"encoded max: {torch.max(outp)}")
        #print(f"encoded min: {torch.min(outp)}")
        #print(f"decoded average: {torch.mean(decoded_time)}")
        #print(f"decoded max: {torch.max(decoded_time)}")
        #print(f"decoded min: {torch.min(decoded_time)}\n")
    
    all_time_series = []
    for file in FILES:
          with open(os.path.join(base_pth, SV_PTH, file), "r") as file:
                time = []
                for line in file:
                    line = line.strip()
                    if line[0:2] != "//":
                        num = q88_hex_to_float(line)
                        time.append(num)
                all_time_series.append(time)

    time_tensors = [torch.tensor(all_time_series[0]).view(WAVEFORMS, 940),  torch.tensor(all_time_series[1]).view(WAVEFORMS, 940),  torch.tensor(all_time_series[2]).view(WAVEFORMS, 940), torch.tensor(all_time_series[3]).view(WAVEFORMS, 940)]

    freq_tensors = [ed_gs._frame_to_freq(time_tensors[0], OFDM_CONFIG), ed_gs._frame_to_freq(time_tensors[1], OFDM_CONFIG), ed_gs._frame_to_freq(time_tensors[2], OFDM_CONFIG), ed_gs._frame_to_freq(time_tensors[3], OFDM_CONFIG)]
    
    evm = evm_pct(freq_tensors[0], freq_tensors[3]).item()
    rrmse = calculate_per_burst_rrmse_pct_loss(freq_tensors[0], freq_tensors[3])
    print("SystemVerilog TCN Performance Metrics:")
    print(f"EVM: {evm}")
    print(f"RRMSE: {rrmse}\n")

    evm = evm_pct(py_freq[0], py_freq[3]).item()
    rrmse = calculate_per_burst_rrmse_pct_loss(py_freq[0], py_freq[3])
    print("Pytorch TCN Performance Metrics:")
    print(f"EVM: {evm}")
    print(f"RRMSE: {rrmse}")

    ed_gs._plot_constellation_enhanced_for_sv(run_dir=Path(os.path.join(base_pth, SV_PTH)), sv=freq_tensors, py=py_freq, freqs=OFDM_CONFIG.subcarrier_freqs_hz)

    plt.plot(all_time_series[1][0:3760], label="SystemVerilog", marker='o')
    plt.plot(list(outp.detach().numpy().flatten())[0:3760], label="Pytorch", marker='s')
    plt.plot(list(in_time.detach().numpy().flatten())[0:3760], label="input", marker='s')

    plt.xlabel("Time Series")
    plt.ylabel("Output")
    plt.title("Plotting Input and encoded")
    plt.legend()
    plt.show()

    plt.plot(all_time_series[3][0:3760], label="SystemVerilog", marker='o')
    plt.plot(list(decoded_time.detach().numpy().flatten())[0:3760], label="Pytorch", marker='s')
    plt.plot(list(recieved_time.detach().numpy().flatten())[0:3760], label="input", marker='s')

    plt.xlabel("Time Series")
    plt.ylabel("Output")
    plt.title("Plotting Recived and Decoded")
    plt.legend()
    plt.show()
