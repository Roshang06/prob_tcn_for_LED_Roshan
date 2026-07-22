'''
Learning curve for a single channel architecture: validation per-burst rRMSE as a
function of how many training bursts it is fit on.

Answers "would more data help the channel model?" without collecting any. It holds a
fixed validation set aside, then fits the same architecture on nested, growing subsets
of the remaining bursts and plots val per-burst rRMSE vs training size. A curve that has
flattened means more data buys little; one still dropping means it would.

Offline (no hardware): reads an existing dataset and the DATA_SIZE_SWEEP config block.
'''
import sys
import yaml
from pathlib import Path

import torch
from matplotlib.figure import Figure

from modules.grid_search.adapters import MODEL_REGISTRY
from modules.grid_search.grid import expand_grid, resolve_runtime
from modules.utils import load_ofdm_dataset, calculate_per_burst_rrmse_pct_loss

HERE = Path(__file__).resolve().parent
CONFIG_FILE = HERE / "train_and_validate.yml"
PLOT_PATH = HERE.parent / "data/plots"


def held_out_split(sent, received, val_fraction, seed):
    '''Fixed validation set (seeded) plus the remaining training pool, so every training
    size is evaluated against the same held-out bursts.'''
    num_bursts = sent.shape[0]
    num_val = int(round(num_bursts * val_fraction))
    perm = torch.randperm(num_bursts, generator=torch.Generator().manual_seed(seed))
    val_indices, pool_indices = perm[:num_val], perm[num_val:]
    return (sent[pool_indices], received[pool_indices],
            sent[val_indices], received[val_indices])


def val_per_burst_rrmse(adapter, sent_val, received_val):
    predicted = adapter.predict(sent_val)
    if isinstance(predicted, tuple):        # probabilistic adapters return (noisy, mean, ...)
        predicted = predicted[1]
    received_val = received_val.to(predicted.device)
    return calculate_per_burst_rrmse_pct_loss(received_val, predicted)


if __name__ == "__main__":
    with open(CONFIG_FILE, encoding="utf-8") as f:
        full = yaml.safe_load(f)

    sweep = full["DATA_SIZE_SWEEP"]
    device, seed = resolve_runtime(full, None, None)

    dataset_path = sys.argv[1] if len(sys.argv) > 1 else (
        sweep.get("DATASET_PATH") or full["DATA_COLLECTION"]["DATASET_PATH"])
    val_fraction = float(sweep.get("VAL_FRACTION", 0.1))
    train_sizes = list(sweep["TRAIN_SIZES"])

    # one fixed architecture; if the config leaves list values, use the first combination
    point = expand_grid([{"model": sweep["model"], "params": sweep["params"]}])[0]
    model_type, params = point["model"], point["params"]

    sent, received, ofdm_config = load_ofdm_dataset(str(dataset_path), device)
    # channel model sees the OFDM symbol only (CP + payload); drop the preamble
    preamble_length = sent.shape[1] - ofdm_config.baseband_fft_length - ofdm_config.cyclic_prefix_length
    sent, received = sent[:, preamble_length:], received[:, preamble_length:]

    sent_pool, received_pool, sent_val, received_val = held_out_split(
        sent, received, val_fraction, seed)
    pool_size = sent_pool.shape[0]

    print(f"dataset {Path(dataset_path).name}")
    print(f"  {model_type} {params}")
    print(f"  {pool_size} training bursts available, {sent_val.shape[0]} held out for validation\n")

    measured_sizes, val_scores = [], []
    for size in train_sizes:
        if size > pool_size:
            print(f"  skipping size {size} (only {pool_size} training bursts available)")
            continue

        # reseed so every fit starts from the same initialization: only the data differs
        torch.manual_seed(seed)
        adapter = MODEL_REGISTRY[model_type].from_config(params, device)
        adapter.fit(sent_pool[:size], received_pool[:size], sent_val, received_val)
        score = val_per_burst_rrmse(adapter, sent_val, received_val)

        measured_sizes.append(size)
        val_scores.append(score)
        print(f"  train size {size:6d}  val_per_burst_rrmse_pct {score:.3f}")

    hidden = params.get("hidden_channels", params.get("hidden_dim", "?"))
    fig = Figure(figsize=(7, 5))
    ax = fig.subplots()
    ax.plot(measured_sizes, val_scores, marker="o", color="#0072B2", lw=1.5)
    ax.set_xlabel("Training bursts")
    ax.set_ylabel("Validation per-burst rRMSE (%)")
    ax.set_title(f"Channel learning curve: {model_type} "
                 f"(hidden={hidden}, kernel={params.get('kernel_size', '?')})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    PLOT_PATH.mkdir(parents=True, exist_ok=True)
    out_path = PLOT_PATH / "dataset_size_sweep.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"\nsaved {out_path}")
