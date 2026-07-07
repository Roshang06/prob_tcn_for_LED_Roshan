"""
ed_validation_phase_gap.py — diagnose the systematic phase error of hardware-validated
encoder/decoder pairs by separating what was already wrong in simulation from the
sim-to-real gap.

For every run in an E/D validation experiment it computes the end-to-end per-carrier
response H_ed = mean(FFT(received)/FFT(sent)) from validation.zarr and regresses its
phase onto the raw channel phase (coupling coefficient a: a=0 means the E/D fully
equalized the channel curvature, a=1 means it ignored it). For the best run per
channel family it then:
  * replays the exact validation sent bursts through encoder -> frozen channel model
    -> decoder (the training-time environment) to get the SIM phase error,
  * evaluates the channel model at the encoded operating point vs the sweep's
    training-distribution inputs,
  * decomposes hardware EVM into systematic (mean per-carrier gain error) and random
    (scatter) parts, and estimates per-trial residual delay jitter from the linear
    phase of each trial (0.29 samples std = integer-sample sync quantization limit).

Edit CONFIG, then:
    cd <repo_root>/experiments
    python ed_validation_phase_gap.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import zarr
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from modules.grid_search.adapters import MODEL_REGISTRY
from modules.grid_search.orchestrator import select_channel_models
from modules.models import TCN

# ─── CONFIG ─────────────────────────────────────────────────────────────────────
SWEEP_PATH  = "data/sweeps/dc0.05A_fmin300000_fmax7.6e+06_20260703_1631.zarr"
CHANNEL_EXP = "data/experiments/train_and_validate/new_heath_channel_models_20260706_1730"
ED_EXP      = "data/experiments/train_and_validate/new_heath_encoder_decoder_20260707_0250"
VAL_EXP     = "data/experiments/train_and_validate/new_heath_ed_validation_20260707_0630"
CLIP_THRESHOLD = 3.0
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
PLOT_PATH   = HERE.parent / "data/plots"
# ────────────────────────────────────────────────────────────────────────────────

ARCH_KEYS = ("nlayers", "dilation_base", "kernel_size", "hidden_channels")


def read_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def mean_response(sent, received, ks):
    S = np.fft.fft(sent.astype(np.float64), norm="ortho", axis=1)[:, ks]
    R = np.fft.fft(received.astype(np.float64), norm="ortho", axis=1)[:, ks]
    return np.mean(R / S, axis=0), S, R


def load_channel_model(channel_exp_dir, run_id):
    sel = select_channel_models(channel_exp_dir, run_ids=[run_id])[0]
    adapter = MODEL_REGISTRY[sel["model"]].load(sel["params"], sel["checkpoint"], DEVICE)
    model = adapter.model if hasattr(adapter, "model") else adapter
    if hasattr(model, "parameters"):
        for p in model.parameters():
            p.requires_grad_(False)
    return model


def run_model(model, x):
    with torch.no_grad():
        out = model(x)
    return out[0] if isinstance(out, tuple) else out


def main():
    PLOT_PATH.mkdir(parents=True, exist_ok=True)
    sweep = zarr.open_group(HERE.parent / SWEEP_PATH, mode="r")
    attrs = dict(sweep.attrs)
    ks = np.array(attrs["active_carrier_indices"])
    cp = int(attrs["cyclic_prefix_length"])
    offset = int(attrs["preamble_length"]) + cp
    x_sweep = sweep["sent_burst"][:, offset:].astype(np.float64)
    y_sweep = sweep["received_burst"][:, offset:].astype(np.float64)
    N = x_sweep.shape[1]
    spacing = float(attrs["f_min_hz"]) / ks[0]
    f_mhz = ks * spacing / 1e6
    fs = N * spacing

    H_raw, _, _ = mean_response(x_sweep, y_sweep, ks)
    phi_raw = np.unwrap(np.angle(H_raw))

    val = zarr.open_group(HERE.parent / VAL_EXP / "validation.zarr", mode="r")
    val_rows = {r["run_id"]: r for r in read_jsonl(HERE.parent / VAL_EXP / "runs.jsonl")}
    ed_rows = {r["run_id"]: r for r in read_jsonl(HERE.parent / ED_EXP / "runs.jsonl")}

    # hardware end-to-end phase error for every validated run
    phi_hw = {}
    for rid in val.group_keys():
        H, _, _ = mean_response(val[rid]["sent_time"][:], val[rid]["received_time"][:], ks)
        phi_hw[rid] = np.unwrap(np.angle(H))

    # phi_err ~ a*phi_raw + linear(f): a measures how much channel curvature survives
    A = np.column_stack([np.ones_like(f_mhz), f_mhz, phi_raw])
    a_vals = np.array([np.linalg.lstsq(A, pe, rcond=None)[0][2] for pe in phi_hw.values()])
    print(f"{len(phi_hw)} validated runs; coupling a to raw channel phase: "
          f"mean={a_vals.mean():.3f} std={a_vals.std():.3f} "
          f"range=({a_vals.min():.3f}, {a_vals.max():.3f})")

    # best run per channel family -> sim replay + operating-point checks
    picks, seen = [], set()
    for r in sorted(val_rows.values(), key=lambda r: r["evm_pct"]):
        if r["channel_form"] not in seen:
            picks.append(r)
            seen.add(r["channel_form"])

    results = {}
    for r in picks:
        rid, ed_rid = r["run_id"], r["model"]  # validation "model" = E/D grid run_id
        arch = {k: ed_rows[ed_rid][k] for k in ARCH_KEYS}
        encoder, decoder = TCN(**arch).to(DEVICE), TCN(**arch).to(DEVICE)
        ckpt = torch.load(HERE.parent / ED_EXP / "runs" / ed_rid / "model.pt",
                          map_location=DEVICE, weights_only=True)
        encoder.load_state_dict(ckpt["encoder"])
        decoder.load_state_dict(ckpt["decoder"])
        encoder.eval(); decoder.eval()
        channel = load_channel_model(HERE.parent / CHANNEL_EXP, r["channel_run_id"])

        sent = val[rid]["sent_time"][:].astype(np.float32)          # symbol only
        sym = torch.tensor(sent, device=DEVICE)
        burst = torch.hstack([sym[:, -cp:], sym])                    # [CP | symbol]
        with torch.no_grad():
            encoded = encoder(burst).clamp(-CLIP_THRESHOLD, CLIP_THRESHOLD)
            decoded = decoder(run_model(channel, encoded))

        H_sim, _, _ = mean_response(sent, decoded[:, -N:].cpu().numpy(), ks)
        phi_sim = np.unwrap(np.angle(H_sim))

        # channel-model response at the encoded operating point vs training inputs
        enc_np = encoded[:, -N:].cpu().numpy()
        H_enc, _, _ = mean_response(enc_np, run_model(channel, encoded)[:, -N:].cpu().numpy(), ks)
        x_train = torch.tensor(x_sweep[:256, -(N + cp):].astype(np.float32), device=DEVICE)
        H_train, _, _ = mean_response(x_sweep[:256], run_model(channel, x_train)[:, -N:].cpu().numpy(), ks)

        # hardware EVM split + per-trial residual delay jitter
        G, S, R = mean_response(val[rid]["sent_time"][:], val[rid]["received_time"][:], ks)
        sig = np.mean(np.abs(S) ** 2)
        evm_sys = np.sqrt(np.mean(np.abs((G[None, :] - 1) * S) ** 2) / sig) * 100
        evm_rand = np.sqrt(np.mean(np.abs(R - G[None, :] * S) ** 2) / sig) * 100
        taus_ns = np.array([
            -np.polyfit(f_mhz * 1e6, np.unwrap(np.angle(R[t] / (S[t] * G))), 1)[0]
            / (2 * np.pi) * 1e9 for t in range(R.shape[0])])
        phase_scatter = np.std(np.angle(R / (S * G[None, :])), axis=0)
        enc_power = (encoded[:, -N:] ** 2).mean(dim=1).cpu().numpy()

        results[rid] = dict(form=r["channel_form"], phi_sim=phi_sim,
                            dphi_op=np.unwrap(np.angle(H_enc)) - np.unwrap(np.angle(H_train)),
                            phase_scatter=phase_scatter)
        print(f"\n{rid} ({r['channel_form']}, channel={r['channel_run_id']})")
        print(f"  EVM: sim={ed_rows[ed_rid]['evm_pct']:.2f}%  hw={r['evm_pct']:.2f}% "
              f"= systematic {evm_sys:.2f}% ⊕ random {evm_rand:.2f}%")
        print(f"  phase-err span: sim={np.ptp(phi_sim):.3f} rad  hw={np.ptp(phi_hw[rid]):.3f} rad")
        sweep_power = (x_sweep ** 2).mean(axis=1)
        print(f"  encoder-out symbol power: {enc_power.mean():.2f} "
              f"(sweep bursts span {sweep_power.min():.2f}-{sweep_power.max():.2f})")
        print(f"  per-trial delay jitter: std={taus_ns.std():.2f} ns "
              f"({taus_ns.std() * fs / 1e9:.3f} samples @ {fs / 1e6:.1f} MS/s)")

    fig, axs = plt.subplots(2, 2, figsize=(13, 9))

    ax = axs[0, 0]
    for pe in phi_hw.values():
        ax.plot(f_mhz, pe - pe.mean(), color="C0", alpha=0.15, lw=0.8)
    mean_pe = np.mean(list(phi_hw.values()), axis=0)
    ax.plot(f_mhz, mean_pe - mean_pe.mean(), "C0", lw=2, label="hw E/D phase err (mean)")
    ax.plot(f_mhz, phi_raw - phi_raw.mean(), "k--", lw=1.5, label="raw channel phase")
    ax.set_title("Hardware E/D phase error vs channel phase (demeaned)")
    ax.set_ylabel("phase (rad)")

    ax = axs[0, 1]
    for rid, res in results.items():
        ax.plot(f_mhz, phi_hw[rid] - phi_hw[rid].mean(), lw=1.5, label=f"hw {res['form']}")
        ax.plot(f_mhz, res["phi_sim"] - res["phi_sim"].mean(), lw=1.5, ls="--",
                label=f"sim {res['form']}")
    ax.set_title("Best run per family: hardware vs sim-replay phase error")
    ax.set_ylabel("phase (rad)")

    ax = axs[1, 0]
    for res in results.values():
        d = res["dphi_op"] - res["dphi_op"].mean()
        ax.plot(f_mhz, d, lw=1.5, label=res["form"])
    ax.set_title("Channel-model phase: encoded operating point − training inputs")
    ax.set_ylabel("Δphase (rad)")

    ax = axs[1, 1]
    for res in results.values():
        ax.plot(f_mhz, res["phase_scatter"], lw=1.5, label=res["form"])
    ax.set_title("Per-carrier phase scatter across trials (sync jitter + noise)")
    ax.set_ylabel("std (rad)")

    for ax in axs.flat:
        ax.set_xlabel("Frequency (MHz)")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(PLOT_PATH / "ed_validation_phase_gap.png", dpi=130, bbox_inches="tight")
    fig.savefig(PLOT_PATH / "ed_validation_phase_gap.svg", format="svg", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
