'''
Shared run-folder bookkeeping for grid-search orchestrators: experiment_config.yaml
manifest, resumable runs/<run_id>/ folders, runs.jsonl run table, and leaderboard.csv.

Subclasses provide `self.points` (list of {"model", "params", ...}), and override
`_prepare()` (load data once) and `_run_point(point, run_dir, context)` (train + eval
+ save one grid point, returning a metrics dict).
'''
import csv
import hashlib
import json
import random
import socket
import subprocess
import time
from datetime import datetime
from pathlib import Path
from matplotlib.figure import Figure

import numpy as np
import torch
import yaml

_REPO_DIR = Path(__file__).resolve().parents[2]


def _git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_REPO_DIR, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


def run_id(point: dict) -> str:
    digest = hashlib.md5(json.dumps(point, sort_keys=True).encode()).hexdigest()[:8]
    return f"{point['model']}_{digest}"


class GridSearchBase:
    def __init__(self, points, grid_manifest, shared_params, experiments_dir,
                 device, seed, experiment_name, extra_manifest=None):
        self.points = points
        self.grid_manifest = grid_manifest
        self.shared_params = shared_params
        self.device = device
        self.seed = seed
        self.extra_manifest = extra_manifest or {}

        base = Path(experiments_dir) if experiments_dir else _REPO_DIR / "data" / "experiments"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        self.exp_dir = base / f"{experiment_name}_{timestamp}"
        self.runs_dir = self.exp_dir / "runs"
        self.summary_dir = self.exp_dir / "summary"
        self.runs_jsonl = self.exp_dir / "runs.jsonl"

    # ------------------------------------------------------------------ setup
    def setup(self):
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.summary_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "experiment_dir": str(self.exp_dir),
            "git_hash": _git_hash(),
            "seed": self.seed,
            "device": self.device,
            "host": socket.gethostname(),
            "created": datetime.now().isoformat(timespec="seconds"),
            "n_points": len(self.points),
            "grid": self.grid_manifest,
            **self.extra_manifest,
        }
        with open(self.exp_dir / "experiment_config.yaml", "w") as f:
            yaml.safe_dump(manifest, f, sort_keys=False)
        print(f"[grid] experiment dir: {self.exp_dir}  ({len(self.points)} points)")
        return self

    # ----------------------------------------------------------------- helpers
    def _seed_everything(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

    def _write_history(self, run_dir: Path, history: dict):
        if not history:
            return
        n = len(next(iter(history.values())))
        with open(run_dir / "history.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["epoch", *history.keys()])
            for i in range(n):
                w.writerow([i, *(history[k][i] for k in history)])
        self._plot_history(run_dir, history)

    def _plot_history(self, run_dir: Path, history: dict):
        '''Save a train/validation loss-vs-epoch curve under plots/ for trainable
        models. Closed-form models pass an empty history and are skipped upstream.
        Uses the OO Figure API so it never switches the global matplotlib backend.'''
        labels = {"loss": "train", "val_loss": "validation"}
        series = {k: v for k, v in history.items() if k.endswith("loss")}
        if not series:
            return
        epochs = range(len(next(iter(series.values()))))
        fig = Figure(figsize=(7, 4))
        ax = fig.subplots()
        for key, values in series.items():
            ax.plot(epochs, values, label=labels.get(key, key), marker=".", ms=3)
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.set_title(run_dir.name)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        plots_dir = run_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(plots_dir / "loss.png", dpi=120)

    def _append_run_record(self, rid, point, metrics):
        record = {"run_id": rid, "model": point["model"], **point["params"], **metrics}
        with open(self.runs_jsonl, "a") as f:
            f.write(json.dumps(record) + "\n")

    def _write_leaderboard(self):
        if not self.runs_jsonl.exists():
            return
        rows = [json.loads(line) for line in self.runs_jsonl.read_text().splitlines() if line.strip()]
        if not rows:
            return
        cols = list(dict.fromkeys(k for r in rows for k in r))  # union, order-preserving
        with open(self.summary_dir / "leaderboard.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)

    # ------------------------------------------------------------- subclass hooks
    def _prepare(self, **kwargs):
        '''Load whatever data/models every grid point needs, once.'''
        return None

    def _run_point(self, point, run_dir, context):
        '''Train + evaluate + save one grid point. Must return a metrics dict.'''
        raise NotImplementedError

    # --------------------------------------------------------------------- run
    def run(self, **prepare_kwargs):
        self.setup()
        context = self._prepare(**prepare_kwargs)

        for i, point in enumerate(self.points):
            rid = run_id(point)
            run_dir = self.runs_dir / rid
            metrics_path = run_dir / "metrics.json"
            if metrics_path.exists():
                print(f"[grid] ({i + 1}/{len(self.points)}) skip {rid} (done)")
                continue

            (run_dir / "plots").mkdir(parents=True, exist_ok=True)
            with open(run_dir / "config.yaml", "w") as f:
                yaml.safe_dump({**point, "shared": self.shared_params}, f, sort_keys=False)

            self._seed_everything()
            t0 = time.time()
            metrics = self._run_point(point, run_dir, context)
            metrics["train_seconds"] = round(time.time() - t0, 3)

            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)
            self._append_run_record(rid, point, metrics)
            print(f"[grid] ({i + 1}/{len(self.points)}) done {rid}")

        self._write_leaderboard()
        return self.exp_dir
