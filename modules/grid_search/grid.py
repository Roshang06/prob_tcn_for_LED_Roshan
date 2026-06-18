'''
Grid spec loading + expansion.

A grid config is a YAML file like:

    models:
      - model: gmp
        params:
          memory_linear: [4, 8]          # list -> swept
          memory_nonlinear: 4            # scalar -> fixed
          nonlinearity_order: 5
          cross_term_depth: [0, 2]
          ridge: 1.0e-3
      - model: tcn
        params:
          nlayers: [2, 3]
          hidden_channels: [16, 32]
          num_taps: 10
          epochs: 200
          lr: 1.0e-3

expand_grid() turns each model spec into the cartesian product of its
list-valued params, yielding one point dict per model instance:

    {"model": "gmp", "params": {"memory_linear": 4, ...}}
'''
import itertools
from pathlib import Path

import yaml


def load_grid(config_path) -> dict:
    with open(Path(config_path)) as f:
        return yaml.safe_load(f)


def resolve_runtime(full_config: dict, device=None, seed=None):
    runtime = full_config.get("RUNTIME") or {}
    device = device if device is not None else runtime.get("DEVICE", "cpu")
    seed = seed if seed is not None else runtime.get("SEED", 0)
    return device, int(seed)


def expand_grid(model_specs: list) -> list:
    points = []
    for spec in model_specs:
        model_name = spec["model"]
        grid = spec.get("params", {})
        keys = list(grid)
        value_lists = [v if isinstance(v, list) else [v] for v in grid.values()]
        for combo in itertools.product(*value_lists):
            points.append({"model": model_name, "params": dict(zip(keys, combo))})
    return points