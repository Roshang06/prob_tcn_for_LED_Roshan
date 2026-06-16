'''
Model adapters + registry for the channel-model grid search.

Each adapter wraps one model family from modules.models behind a single
interface so the orchestrator never needs to know how a given model trains
(TCN = iterative SGD, GMP = closed-form least squares).

Training internals are intentionally left as TODOs; only the structure is here.
'''
from pathlib import Path
from typing import Protocol, runtime_checkable

import torch

from modules.models import TCN_channel, GeneralizedMemoryPolynomial


@runtime_checkable
class ChannelModel(Protocol):
    name: str

    @classmethod
    def from_config(cls, params: dict, device: str, shared: dict = None) -> "ChannelModel": ...

    def fit(self, X, Y) -> dict:
        '''Train and return a history dict (e.g. {"loss": [...]}); {} if closed-form.'''
        ...

    def predict(self, X): ...

    def num_params(self) -> int: ...

    def save(self, path: Path) -> None: ...


class TCNAdapter:
    name = "tcn"
    ARCH_KEYS = ("nlayers", "dilation_base", "num_taps", "hidden_channels", "learn_noise", "gaussian")
    TRAIN_KEYS = ("epochs", "lr", "batch_size")

    def __init__(self, model: TCN_channel, train_params: dict, device: str, shared: dict = None):
        self.model = model
        self.train_params = train_params
        self.device = device
        self.shared = shared or {}

    @classmethod
    def from_config(cls, params: dict, device: str, shared: dict = None) -> "TCNAdapter":
        arch = {k: params[k] for k in cls.ARCH_KEYS if k in params}
        model = TCN_channel(**arch).to(device)
        train_params = {k: params[k] for k in cls.TRAIN_KEYS if k in params}
        return cls(model, train_params, device, shared=shared)

    def fit(self, X, Y) -> dict:
        # TODO: SGD loop. Build optimizer (see modules.utils.make_optimizer), iterate
        # epochs over (X, Y), backprop the NLL / rrmse loss, collect per-epoch history.
        # Grid-level settings are in self.shared, e.g. self.shared["RECEPTIVE_FIELD"].
        raise NotImplementedError("TCN SGD training loop not implemented yet")

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            return self.model(X.to(self.device))

    def num_params(self) -> int:
        return self.model.get_num_params()

    def save(self, path: Path) -> None:
        torch.save(self.model.state_dict(), path)


class GMPAdapter:
    name = "gmp"
    ARCH_KEYS = ("memory_linear", "memory_nonlinear", "nonlinearity_order", "cross_term_depth")
    FIT_KEYS = ("ridge", "batch_chunk")

    def __init__(self, model: GeneralizedMemoryPolynomial, fit_params: dict, device: str, shared: dict = None):
        self.model = model
        self.fit_params = fit_params
        self.device = device
        self.shared = shared or {}

    @classmethod
    def from_config(cls, params: dict, device: str, shared: dict = None) -> "GMPAdapter":
        model = GeneralizedMemoryPolynomial(
            weights=None,
            memory_linear=params["memory_linear"],
            memory_nonlinear=params["memory_nonlinear"],
            nonlinearity_order=params["nonlinearity_order"],
            cross_term_depth=params["cross_term_depth"],
            device=torch.device(device),
        )
        fit_params = {k: params[k] for k in cls.FIT_KEYS if k in params}
        return cls(model, fit_params, device, shared=shared)

    def fit(self, X, Y) -> dict:
        self.model.fit(X, Y, **self.fit_params)
        return {}  # closed-form: no per-epoch history

    def predict(self, X):
        return self.model.predict(X)

    def num_params(self) -> int:
        return self.model.get_num_regressors()

    def save(self, path: Path) -> None:
        torch.save({"weights": self.model.weights, "fit_params": self.fit_params}, path)


class MockAdapter:
    '''Smoke-test adapter: no real model, lets you exercise the orchestrator
    (folders, runs.jsonl, resume) without touching torch or data. Remove later.'''
    name = "mock"

    def __init__(self, params: dict, shared: dict = None):
        self.params = params
        self.shared = shared or {}

    @classmethod
    def from_config(cls, params: dict, device: str, shared: dict = None) -> "MockAdapter":
        return cls(params, shared=shared)

    def fit(self, X, Y) -> dict:
        return {"loss": [1.0, 0.5, 0.25]}

    def predict(self, X):
        return X

    def num_params(self) -> int:
        return int(self.params.get("n", 100))

    def save(self, path: Path) -> None:
        Path(path).write_bytes(b"mock-checkpoint")


MODEL_REGISTRY = {
    TCNAdapter.name: TCNAdapter,
    GMPAdapter.name: GMPAdapter,
    MockAdapter.name: MockAdapter,
}
