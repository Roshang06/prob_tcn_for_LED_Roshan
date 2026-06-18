'''
ChannelModelGridSearch: runs a config-driven grid over channel-model
architectures and writes a self-describing, resumable run folder.

Layout produced under data/experiments/<name>_<timestamp>/:

    experiment_config.yaml   manifest: grid + dataset path + git hash + seed
    runs.jsonl               one line per finished run (the "run table")
    runs/<run_id>/
        config.yaml          this run's resolved params
        model.pt              checkpoint (adapter.save)
        metrics.json          eval metrics (rrmse_pct + val_rrmse_pct) + num_params + train_seconds
        history.csv          per-epoch train/val loss (omitted for closed-form models)
        plots/loss.png       train vs validation loss curve (trainable models only)
    summary/leaderboard.csv  aggregated view of runs.jsonl

run_id is a deterministic hash of the point, so a finished run (metrics.json
present) is skipped on re-run -> crash-resumable for free. See base.py for the
shared run-folder bookkeeping reused by EncoderDecoderGridSearch.

VAL_FRACTION (grid-wide) holds out a random slice of frames for validation;
channel models are then selected on val_rrmse_pct (see select_channel_models).
SELECTED_RUN_IDS can pin exactly which runs advance to the encoder/decoder stage.
'''
import json
from pathlib import Path

import torch
import yaml

from modules.grid_search.adapters import MODEL_REGISTRY
from modules.grid_search.base import GridSearchBase
from modules.grid_search.grid import expand_grid, resolve_runtime
from modules.utils import calculate_rrmse_pct_loss, load_ofdm_dataset


class ChannelModelGridSearch(GridSearchBase):
    # keys consumed by the orchestrator itself, not forwarded to adapters as shared
    _ORCHESTRATOR_KEYS = ("models", "VAL_FRACTION", "SELECTED_RUN_IDS")

    def __init__(self,
                 grid_config: dict,
                 dataset_path,
                 experiments_dir=None,
                 device="cpu",
                 seed=0,
                 experiment_name="channel_models",
                 val_fraction=None
                 ):

        self.dataset_path = Path(dataset_path)
        self.val_fraction = float(grid_config.get("VAL_FRACTION", val_fraction or 0.0))

        points = expand_grid(grid_config["models"])
        # every CHANNEL_GRID_SEARCH key except orchestrator settings is a grid-wide
        # setting (e.g. RECEPTIVE_FIELD) handed to every adapter via from_config(shared=...)
        shared_params = {k: v for k, v in grid_config.items() if k not in self._ORCHESTRATOR_KEYS}
        super().__init__(points, grid_config, shared_params, experiments_dir, device, seed,
                          experiment_name, extra_manifest={
                              "dataset_path": str(self.dataset_path),
                              "val_fraction": self.val_fraction,
                          })

    @classmethod
    def from_experiment_config(cls, 
                               config_path,
                               device=None,
                               seed=None,
                               experiment_name="channel_models",
                               experiments_dir=None
                               ):
        '''Build the grid search from the unified end-to-end config'''
    
        with open(Path(config_path)) as f:
            full = yaml.safe_load(f)
    
        grid_config = full["CHANNEL_GRID_SEARCH"]
        dataset_path = full["DATA_COLLECTION"]["DATASET_PATH"]
        device, seed = resolve_runtime(full, device, seed)
        
        return cls(grid_config, dataset_path=dataset_path, device=device, seed=seed,
                   experiment_name=experiment_name, experiments_dir=experiments_dir)

    # ------------------------------------------------------------- data/eval
    def _prepare(self, data=None):
        '''data: optional (X, Y) tuple; if None, loaded from self.dataset_path.
        Returns (X_train, Y_train, X_val, Y_val); the val tensors are None when
        VAL_FRACTION is 0.'''
        if data is not None:
            sent, received = data
        else:
            sent, received, _ = load_ofdm_dataset(str(self.dataset_path), self.device)
        return self._split_train_val(sent, received)

    def _split_train_val(self, X, Y):
        n = X.shape[0]
        n_val = int(round(n * self.val_fraction))
        if n_val == 0:
            return X, Y, None, None
        # deterministic shuffle so the same seed reproduces the same split
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(self.seed))
        val_idx, train_idx = perm[:n_val], perm[n_val:]
        return X[train_idx], Y[train_idx], X[val_idx], Y[val_idx]

    def _evaluate(self, adapter, X, Y) -> dict:
        y_pred = adapter.predict(X)
        if isinstance(y_pred, tuple):  # TCN learn_noise: (noisy, mean, std, nu)
            y_pred = y_pred[1]
        return {"rrmse_pct": calculate_rrmse_pct_loss(Y.to(y_pred.device), y_pred)}

    # ----------------------------------------------------------------- run
    def _run_point(self, point, run_dir, context) -> dict:
        X, Y, X_val, Y_val = context
        adapter = MODEL_REGISTRY[point["model"]].from_config(
            point["params"], self.device, shared=self.shared_params)

        history = adapter.fit(X, Y, X_val, Y_val)
        metrics = self._evaluate(adapter, X, Y)
        if X_val is not None:
            # selection metric: held-out validation rRMSE (see select_channel_models)
            metrics["val_rrmse_pct"] = self._evaluate(adapter, X_val, Y_val)["rrmse_pct"]
            # probabilistic (learn_noise) models also report validation NLL
            val_nll = adapter.val_nll(X_val, Y_val)
            if val_nll is not None:
                metrics["val_nll"] = val_nll
        metrics["num_params"] = int(adapter.num_params())

        adapter.save(run_dir / "model.pt")
        self._write_history(run_dir, history)
        return metrics


def select_channel_models(exp_dir, mode="best", metric="val_rrmse_pct", run_ids=None):
    '''Pick channel models out of a finished ChannelModelGridSearch run for the
    encoder/decoder stage.

    run_ids: optional explicit list of run_ids to forward (config-level override);
        when given, exactly those runs are returned and `mode` is ignored.
    mode="best": the lowest-`metric` run per model family (default metric is the
        held-out validation rRMSE, falling back to train rRMSE when no validation
        split was used). mode="all": every finished run.'''
    exp_dir = Path(exp_dir)
    rows = [json.loads(line) for line in (exp_dir / "runs.jsonl").read_text().splitlines() if line.strip()]

    def score(row):
        return row.get(metric, row.get("rrmse_pct"))

    if run_ids:
        by_id = {row["run_id"]: row for row in rows}
        missing = [rid for rid in run_ids if rid not in by_id]
        if missing:
            raise ValueError(f"run_ids not found in {exp_dir}: {missing}")
        rows = [by_id[rid] for rid in run_ids]
    elif mode == "best":
        best = {}
        for row in rows:
            cur = best.get(row["model"])
            if cur is None or score(row) < score(cur):
                best[row["model"]] = row
        rows = list(best.values())
    elif mode != "all":
        raise ValueError(f"unknown mode {mode!r}, expected 'all' or 'best'")

    selected = []
    for row in rows:
        run_dir = exp_dir / "runs" / row["run_id"]
        with open(run_dir / "config.yaml") as f:
            params = yaml.safe_load(f)["params"]
        selected.append({
            "run_id": row["run_id"],
            "model": row["model"],
            "params": params,
            "checkpoint": run_dir / "model.pt",
        })
    return selected
