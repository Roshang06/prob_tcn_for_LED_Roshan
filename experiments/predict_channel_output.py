"""
predict_channel_output.py — run a trained channel model over a dataset and write
its predicted output as a new zarr, laid out identically to the input dataset so
it drops straight into channel_response.py.

The channel model is trained on the OFDM symbol only (CP + payload, preamble
dropped), so the prediction fills the [preamble:] region of received_burst; the
preamble region is copied from sent_burst and is never read by channel_response.

Edit CONFIG, then:
    cd <repo_root>/experiments
    python predict_channel_output.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import zarr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from modules.grid_search.adapters import MODEL_REGISTRY
from modules.grid_search.orchestrator import select_channel_models
from modules.utils import load_ofdm_dataset

# ─── CONFIG ─────────────────────────────────────────────────────────────────────
CHANNEL_EXP_DIR = "data/experiments/train_and_validate/new_heath_channel_models_20260706_1730"
DATASET_PATH    = "data/sweeps/dc0.05A_fmin300000_fmax7.6e+06_20260703_1631.zarr"
RUN_ID          = None            # None -> best run by SELECTION_METRIC; or "tcn_xxxxxxxx"
SELECTION_METRIC = "val_rrmse_pct"
OUTPUT_PATH     = None            # None -> alongside the dataset as <stem>_pred_<run_id>.zarr
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
CHUNK_SIZE      = 512
# ────────────────────────────────────────────────────────────────────────────────


def choose_run_id(channel_exp_dir, run_id, metric):
    if run_id is not None:
        return run_id

    rows = [json.loads(line) for line
            in (channel_exp_dir / "runs.jsonl").read_text().splitlines() if line.strip()]

    def score(row):
        return row.get(metric, row.get("rrmse_pct", float("inf")))

    best_row = min(rows, key=score)
    print(f"best run by {metric}: {best_row['run_id']}  "
          f"{metric}={score(best_row):.4f}  model={best_row['model']}  "
          f"dist={best_row.get('distribution', 'none')}")
    return best_row["run_id"]


def predict_received_symbols(adapter, sent_symbols, chunk_size):
    predictions = []
    for start in range(0, sent_symbols.shape[0], chunk_size):
        output = adapter.predict(sent_symbols[start:start + chunk_size])
        if isinstance(output, tuple):
            output = output[1]
        predictions.append(output.detach().cpu())
    return torch.cat(predictions, dim=0).numpy()


def main():
    channel_exp_dir = HERE.parent / CHANNEL_EXP_DIR
    dataset_path = HERE.parent / DATASET_PATH

    run_id = choose_run_id(channel_exp_dir, RUN_ID, SELECTION_METRIC)
    selected = select_channel_models(channel_exp_dir, run_ids=[run_id])[0]
    adapter = MODEL_REGISTRY[selected["model"]].load(
        selected["params"], selected["checkpoint"], DEVICE)

    sent, _, config = load_ofdm_dataset(str(dataset_path), DEVICE)
    preamble_length = sent.shape[1] - config.baseband_fft_length - config.cyclic_prefix_length

    predicted_symbols = predict_received_symbols(
        adapter, sent[:, preamble_length:], CHUNK_SIZE)

    sent_burst = sent.cpu().numpy().astype("float32")
    predicted_received_burst = sent_burst.copy()
    predicted_received_burst[:, preamble_length:] = predicted_symbols.astype("float32")

    output_path = (Path(OUTPUT_PATH) if OUTPUT_PATH
                   else dataset_path.with_name(f"{dataset_path.stem}_pred_{run_id}.zarr"))

    source_root = zarr.open_group(str(dataset_path), mode="r")
    output_root = zarr.open_group(str(output_path), mode="w")
    output_root.attrs.update(dict(source_root.attrs))

    for name, data in [("sent_burst", sent_burst),
                       ("received_burst", predicted_received_burst)]:
        array = output_root.create_array(
            name, shape=data.shape, chunks=(min(64, data.shape[0]), data.shape[1]),
            dtype=data.dtype)
        array[:] = data

    print(f"wrote predicted dataset ({predicted_received_burst.shape}) to {output_path}")
    print("point channel_response.py DATASET_PATH at it to compare the phase response")


if __name__ == "__main__":
    main()
