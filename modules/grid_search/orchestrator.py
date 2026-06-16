'''
ChannelModelGridSearch: runs a config-driven grid over channel-model
architectures and writes a self-describing, resumable run folder.

Layout produced under data/experiments/<name>_<timestamp>/:

    experiment_config.yaml   manifest: grid + dataset path + git hash + seed
    runs.jsonl               one line per finished run (the "run table")
    runs/<run_id>/
        config.yaml          this run's resolved params
        model.pt             checkpoint (adapter.save)
        metrics.json         eval metrics + num_params + train_seconds
        history.csv          per-epoch loss (omitted for closed-form models)
        plots/
    summary/leaderboard.csv  aggregated view of runs.jsonl

run_id is a deterministic hash of the point, so a finished run (metrics.json
present) is skipped on re-run -> crash-resumable for free.

Training/eval internals (_load_data, _evaluate) are TODO stubs.
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

import numpy as np
import torch
import yaml

from modules.grid_search.adapters import MODEL_REGISTRY
from modules.grid_search.grid import expand_grid

_REPO_DIR = Path(__file__).resolve().parents[2]


def _git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_REPO_DIR, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


def _run_id(point: dict) -> str:
    digest = hashlib.md5(json.dumps(point, sort_keys=True).encode()).hexdigest()[:8]
    return f"{point['model']}_{digest}"


class ChannelModelGridSearch:
    def __init__(self, grid_config: dict, dataset_path, experiments_dir=None,
                 device="cpu", seed=0, experiment_name="channel_models"):
        self.grid_config = dict(grid_config)
        self.dataset_path = Path(dataset_path)
        self.device = device
        self.seed = seed
        self.points = expand_grid(self.grid_config["models"])
        # every CHANNEL_GRID_SEARCH key except `models` is a grid-wide setting
        # (e.g. RECEPTIVE_FIELD) handed to every adapter via from_config(shared=...)
        self.shared_params = {k: v for k, v in self.grid_config.items() if k != "models"}

        base = Path(experiments_dir) if experiments_dir else _REPO_DIR / "data" / "experiments"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        self.exp_dir = base / f"{experiment_name}_{timestamp}"
        self.runs_dir = self.exp_dir / "runs"
        self.summary_dir = self.exp_dir / "summary"
        self.runs_jsonl = self.exp_dir / "runs.jsonl"

    @classmethod
    def from_experiment_config(cls, config_path, device="cpu", seed=0,
                               experiment_name="channel_models", experiments_dir=None):
        '''Build the grid search from the unified end-to-end config: the grid spec
        comes from the CHANNEL_GRID_SEARCH section and the dataset from
        DATA_COLLECTION.DATASET_PATH.'''
        with open(Path(config_path)) as f:
            full = yaml.safe_load(f)
        grid_config = full["CHANNEL_GRID_SEARCH"]
        dataset_path = full["DATA_COLLECTION"]["DATASET_PATH"]
        return cls(grid_config, dataset_path=dataset_path, device=device, seed=seed,
                   experiment_name=experiment_name, experiments_dir=experiments_dir)

    # ------------------------------------------------------------------ setup
    def setup(self):
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.summary_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "experiment_dir": str(self.exp_dir),
            "dataset_path": str(self.dataset_path),
            "git_hash": _git_hash(),
            "seed": self.seed,
            "device": self.device,
            "host": socket.gethostname(),
            "created": datetime.now().isoformat(timespec="seconds"),
            "n_points": len(self.points),
            "grid": self.grid_config,
        }
        with open(self.exp_dir / "experiment_config.yaml", "w") as f:
            yaml.safe_dump(manifest, f, sort_keys=False)
        print(f"[grid] experiment dir: {self.exp_dir}  ({len(self.points)} points)")
        return self

    # ------------------------------------------------------------- TODO hooks
    def _load_data(self):
        # TODO: load X (sent_baseband) and Y (received_baseband) tensors from the
        # zarr dataset at self.dataset_path. See modules.utils.extract_zarr_data.
        # Return (X, Y) on self.device (later: split into train/val).
        raise NotImplementedError("dataset loading not wired yet")

    def _evaluate(self, adapter, X, Y) -> dict:
        # TODO: compute validation metrics, e.g.
        #   rrmse = modules.utils.calculate_rrmse_pct_loss(Y, adapter.predict(X))
        return {"rrmse_pct": None, "nmse_db": None}

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

    def _append_run_record(self, run_id, point, metrics):
        record = {"run_id": run_id, "model": point["model"], **point["params"], **metrics}
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

    # --------------------------------------------------------------------- run
    def run(self, data=None):
        '''data: optional (X, Y) tuple; if None, self._load_data() is called.'''
        self.setup()
        if data is None:
            data = self._load_data()
        X, Y = data

        for i, point in enumerate(self.points):
            run_id = _run_id(point)
            run_dir = self.runs_dir / run_id
            metrics_path = run_dir / "metrics.json"
            if metrics_path.exists():
                print(f"[grid] ({i + 1}/{len(self.points)}) skip {run_id} (done)")
                continue

            (run_dir / "plots").mkdir(parents=True, exist_ok=True)
            with open(run_dir / "config.yaml", "w") as f:
                yaml.safe_dump({**point, "shared": self.shared_params}, f, sort_keys=False)

            self._seed_everything()
            adapter = MODEL_REGISTRY[point["model"]].from_config(
                point["params"], self.device, shared=self.shared_params)

            t0 = time.time()
            history = adapter.fit(X, Y)
            train_seconds = round(time.time() - t0, 3)

            metrics = self._evaluate(adapter, X, Y)
            metrics.update(num_params=int(adapter.num_params()), train_seconds=train_seconds)

            adapter.save(run_dir / "model.pt")
            self._write_history(run_dir, history)
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)
            self._append_run_record(run_id, point, metrics)
            print(f"[grid] ({i + 1}/{len(self.points)}) done {run_id}  params={metrics['num_params']}")

        self._write_leaderboard()
        return self.exp_dir
