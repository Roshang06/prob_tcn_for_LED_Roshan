'''
EncoderDecoderValidation: drives the real OFDM hardware chain (ModulateDataOFDM ->
ApplyEncoder -> Send -> Measure -> Resample -> ApplyDecoder -> DemodulateDataOFDM)
with frozen encoder/decoder pairs and writes a grid-search-style run folder.

Each validated E/D model gets runs/<run_id>/ with metrics.json (BER, rRMSE%, EVM%),
constellation.png and waveform.png. A single validation.zarr at the experiment root
stores the sent/received time-domain frames grouped by E/D model id.

Goal: run at the end of the test grid search on the live channel.
'''
import json
from pathlib import Path

import numpy as np
import torch
import zarr
from matplotlib.figure import Figure

from pyflux.core.block import Signal
from pyflux.core.chain import Chain
from modules.experimental_blocks import ApplyEncoder, ApplyDecoder
from modules.grid_search.base import GridSearchBase
from modules.models import TCN
from modules.utils import calculate_BER, evm_pct

ARCH_KEYS = ("nlayers", "dilation_base", "kernel_size", "hidden_channels")


class EncoderDecoderValidation(GridSearchBase):
    def __init__(self, ed_models, ofdm_blocks, num_trials, constellation, clip_value,
                 device="cpu", seed=0, experiments_dir=None, experiment_name="ed_validation",
                 run_prefix=None, debug=False):
        self.ofdm_blocks = ofdm_blocks  # (modulate, send, measure, resample, demod)
        self.num_trials = num_trials
        self.constellation = constellation
        self.clip_value = clip_value

        points = [{"model": m["run_id"],
                   "params": {k: m["arch"][k] for k in ARCH_KEYS},
                   "channel_form": m["channel_form"],
                   "channel_run_id": m.get("channel_run_id"),
                   "channel_receptive_field": m.get("channel_receptive_field"),
                   "channel_distribution": m.get("channel_distribution", "none"),
                   "checkpoint": str(m["checkpoint"])}
                  for m in ed_models]
        super().__init__(points, {"ed_models": [m["run_id"] for m in ed_models]},
                         {"num_trials": num_trials}, experiments_dir, device, seed,
                         experiment_name, run_prefix=run_prefix,
                         extra_manifest={"num_trials": num_trials})
        self.rank_by = "evm_pct"
        self.debug = debug

    def _prepare(self):
        self.val_group = zarr.open_group(self.exp_dir / "validation.zarr", mode="a")
        return None

    @staticmethod
    def _bits(source_bits):
        return np.array([int(b) for b in source_bits], dtype="uint8")

    def _load_pair(self, arch, checkpoint):
        encoder, decoder = TCN(**arch).to(self.device), TCN(**arch).to(self.device)
        ckpt = torch.load(checkpoint, map_location=self.device, weights_only=True)
        encoder.load_state_dict(ckpt["encoder"])
        decoder.load_state_dict(ckpt["decoder"])
        for model in (encoder, decoder):
            model.eval()
            for p in model.parameters():
                p.requires_grad_(False)
        return encoder, decoder

    def _run_point(self, point, run_dir, context) -> dict:
        modulate, send, measure, resample, demod = self.ofdm_blocks
        encoder, decoder = self._load_pair(point["params"], point["checkpoint"])
        chain = Chain([modulate,
                       ApplyEncoder(encoder, modulate, self.clip_value, self.device),
                       send, measure, resample,
                       ApplyDecoder(decoder, demod, self.device, debug=self.debug),
                       demod])

        sent_syms, recv_syms, true_bits, sent_time, recv_time = [], [], [], [], []
        freqs, example = None, None
        for i in range(self.num_trials):
            x = chain.run(Signal(data=np.zeros(1), sampling_rate=modulate.fs_out))
            art = x.artifact_container

            if self.debug:
                print(f"trial {i + 1}/{self.num_trials}  sent={art['sent_symbols'].shape} "
                      f"recv={art['received_symbols'].shape} "
                      f"sent_time={art['sent_baseband'].shape} "
                      f"recv_time={x.data.shape}")
                
            sent_syms.append(np.asarray(art["sent_symbols"]))
            recv_syms.append(np.asarray(art["received_symbols"]))
            true_bits.append(self._bits(art["source_bits"]))
            sent_time.append(np.asarray(art["sent_baseband"], dtype="float32"))
            recv_time.append(np.asarray(x.data, dtype="float32"))
            freqs = np.asarray(art["subcarrier_freqs_hz"])
            if i == 0:
                example = {k: np.asarray(art[k]) for k in
                           ("encoder_input", "encoder_output", "decoder_input", "decoder_output")}
                example["decoder_window"] = art.get("decoder_window")

        example["residual"] = sent_time[0] - recv_time[0]  # in-band sent vs recovered symbol

        sent_syms, recv_syms = np.stack(sent_syms), np.stack(recv_syms)
        sent_t = torch.tensor(sent_syms)
        recv_t = torch.tensor(recv_syms)
        metrics = {
            "evm_pct": evm_pct(sent_t, recv_t).item(),
            "ber": calculate_BER(recv_t.flatten(), np.concatenate(true_bits), self.constellation),
            "num_params": encoder.get_num_params() + decoder.get_num_params(),
            "channel_form": point["channel_form"],
            "channel_run_id": point.get("channel_run_id"),
            "channel_receptive_field": point.get("channel_receptive_field"),
            "channel_distribution": point.get("channel_distribution", "none"),
        }

        self._store_waveforms(run_dir.name, np.stack(sent_time), np.stack(recv_time), point["channel_form"])
        label = f"{run_dir.name} | channel: {point['channel_form']}"
        self._plot_constellation(run_dir, sent_syms, recv_syms, freqs,
                                 ed_run_id=run_dir.name,
                                 channel_form=point["channel_form"],
                                 channel_run_id=point.get("channel_run_id"),
                                 evm=metrics["evm_pct"])
        self._plot_waveform(run_dir, example, label)
        return metrics

    # ------------------------------------------------------------- artifacts
    def _store_waveforms(self, model_id, sent, received, channel_form):
        g = self.val_group[model_id] if model_id in self.val_group else self.val_group.create_group(model_id)
        for name, arr in (("sent_time", sent), ("received_time", received)):
            za = g.create_array(name, shape=arr.shape, chunks=arr.shape, dtype=arr.dtype, overwrite=True)
            za[:] = arr
        g.attrs["channel_form"] = channel_form

    def _plot_constellation(self, run_dir, sent, received, freqs, ed_run_id=None,
                            channel_form=None, channel_run_id=None, evm=None):
        sent = np.asarray(sent)
        received = np.asarray(received)
        freqs = np.asarray(freqs)
        n_rows = sent.shape[0] if sent.ndim > 1 else 1
        c = np.tile(freqs, n_rows)
        sent_flat, recv_flat = sent.ravel(), received.ravel()

        fig = Figure(figsize=(11, 5))
        ax_sent, ax_recv = fig.subplots(1, 2)
        ax_sent.scatter(sent_flat.real, sent_flat.imag, s=10, c=c, cmap="viridis")
        ax_sent.set_title("Sent")
        sc = ax_recv.scatter(recv_flat.real, recv_flat.imag, s=10, c=c, cmap="viridis")
        # overlay reference constellation symbols as red X's
        ax_recv.scatter(sent_flat.real, sent_flat.imag, s=30, marker="x", c="red", linewidth=1.5, alpha=0.7, label="Reference")
        ax_recv.set_title("Received" if evm is None else f"Received (EVM={evm:.2f}%)")
        ax_recv.legend(fontsize=8, loc="upper right")

        for ax in (ax_sent, ax_recv):
            ax.set_xlabel("In-Phase")
            ax.set_ylabel("Quadrature")
            ax.grid(True)
            ax.set_aspect("equal", "box")
        fig.colorbar(sc, ax=[ax_sent, ax_recv], label="Carrier Frequency (Hz)")
        channel = " ".join(p for p in (channel_form, channel_run_id) if p)
        suptitle = " | ".join(p for p in (ed_run_id, f"channel: {channel}" if channel else "") if p)
        if suptitle:
            fig.suptitle(suptitle)
        fig.savefig(run_dir / "plots" / "constellation.png", dpi=120)

    def _plot_waveform(self, run_dir, example, label):
        # mark=True rows are the full received capture; the decoder only touches the
        # synced symbol window, so draw red lines at its bounds on those panels.
        rows = [
            ("encoder input", example["encoder_input"], False),
            ("encoder output", example["encoder_output"], False),
            ("decoder input", example["decoder_input"], True),
            ("decoder output", example["decoder_output"], True),
            ("residual (sent - recovered symbol)", example["residual"], False),
        ]
        window = example.get("decoder_window")
        fig = Figure(figsize=(9, 11))
        axes = fig.subplots(len(rows), 1)
        for ax, (title, signal, mark) in zip(axes, rows):
            ax.plot(signal, lw=1)
            if mark and window is not None:
                for bound in window:
                    ax.axvline(bound, color="red", lw=1.0)
            ax.set_ylabel(title, fontsize=8)
            ax.grid(True, alpha=0.3)
        axes[-1].set_xlabel("Sample index")
        fig.suptitle(label)
        fig.savefig(run_dir / "plots" / "waveform.png", dpi=120)


def _read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def derive_channel_form(record) -> str:
    '''Human-readable channel model family for plot labels, from a channel-grid
    runs.jsonl record: "gmp", "prob TCN" (learns noise) or "nonprob TCN".'''
    rec = record or {}
    model = rec.get("model")
    if model == "gmp":
        return "gmp"
    if model == "tcn":
        dist = rec.get("distribution", "none")
        return "prob TCN" if dist not in ("none", None) else "nonprob TCN"
    return model or "unknown"


def select_encoder_decoders(ed_exp_dir, channel_exp_dir=None, run_ids=None):
    '''Build the EncoderDecoderValidation input from a finished E/D grid search.

    run_ids: manual list of E/D run_ids to validate; None selects every run.
    channel_exp_dir: the channel grid search dir, used to label each E/D model with
        its originating channel form (skipped -> "unknown").'''
    ed_exp_dir = Path(ed_exp_dir)
    rows = _read_jsonl(ed_exp_dir / "runs.jsonl")
    if run_ids:
        by_id = {r["run_id"]: r for r in rows}
        missing = [rid for rid in run_ids if rid not in by_id]
        if missing:
            raise ValueError(f"E/D run_ids not found in {ed_exp_dir}: {missing}")
        rows = [by_id[rid] for rid in run_ids]

    channel_map = {}
    if channel_exp_dir is not None:
        channel_map = {r["run_id"]: r for r in _read_jsonl(Path(channel_exp_dir) / "runs.jsonl")}

    selected = []
    for row in rows:
        rid = row["run_id"]
        ch_meta = channel_map.get(row.get("channel_run_id"), {})
        selected.append({
            "run_id": rid,
            "arch": {k: row[k] for k in ARCH_KEYS},
            "checkpoint": ed_exp_dir / "runs" / rid / "model.pt",
            "channel_run_id": row.get("channel_run_id"),
            "channel_form": derive_channel_form(ch_meta),
            "channel_receptive_field": ch_meta.get("receptive_field"),
            "channel_distribution": ch_meta.get("distribution", "none"),
        })
    return selected
