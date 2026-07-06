'''
Model adapters + registry for the channel-model grid search.

Each adapter wraps one model family from modules.models behind a single
interface so the orchestrator never needs to know how a given model trains
(TCN = iterative SGD, GMP = closed-form least squares).
'''
from pathlib import Path
from typing import Protocol, runtime_checkable

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from modules.models import TCN_channel, GeneralizedMemoryPolynomial, LRU_channel


_DIST_TO_FLAGS = {
    "none":       {"learn_noise": False, "gaussian": True},
    "gaussian":   {"learn_noise": True,  "gaussian": True},
    "students_t": {"learn_noise": True,  "gaussian": False},
}


def _beta_nll_reduce(per_element_nll, std, beta):
    if not beta:
        return torch.mean(per_element_nll)
    return torch.mean((std.detach() ** (2.0 * beta)) * per_element_nll)


def gaussian_nll(residual, std, beta=0.0):
    per_element_nll = 0.5 * torch.log(2 * torch.pi * std ** 2) + 0.5 * (residual / std) ** 2
    return _beta_nll_reduce(per_element_nll, std, beta)


def students_t_loss(residual, std, nu, beta=0.0):
    z = residual / std
    term1 = (-torch.lgamma((nu + 1) / 2) + 0.5 * torch.log(torch.pi * nu)
             + torch.lgamma(nu / 2) + torch.log(std + 1e-8))
    term2 = ((nu + 1) / 2) * torch.log(1 + torch.square(z) / nu + 1e-8)
    return _beta_nll_reduce(term1 + term2, std, beta)


@runtime_checkable
class ChannelModel(Protocol):
    name: str

    @classmethod
    def from_config(cls, params: dict, device: str, shared: dict = None) -> "ChannelModel": ...

    @classmethod
    def load(cls, params: dict, checkpoint: Path, device: str, shared: dict = None) -> "ChannelModel":
        '''Rebuild a trained adapter from a checkpoint written by save().'''
        ...

    def fit(self, X, Y, X_val=None, Y_val=None) -> dict:
        '''Train and return a history dict (e.g. {"loss": [...], "val_loss": [...]});
        {} if closed-form. X_val/Y_val are optional held-out validation tensors used
        to record a per-epoch validation loss for trainable models.'''
        ...

    def predict(self, X): ...

    def val_nll(self, X, Y):
        '''Validation negative log-likelihood for probabilistic models; None for
        deterministic ones (used to add `val_nll` to the metrics).'''
        ...

    def num_params(self) -> int: ...

    def save(self, path: Path) -> None: ...


class TCNAdapter:
    name = "tcn"
    ARCH_KEYS = ("nlayers", "dilation_base", "kernel_size", "hidden_channels", "distribution", "learn_noise", "gaussian")
    TRAIN_KEYS = ("epochs", "lr", "batch_size", "factor", "patience", "min_lr", "beta_nll")

    def __init__(self, model: TCN_channel, train_params: dict, device: str, shared: dict = None):
        self.model = model
        self.train_params = train_params
        self.device = device
        self.shared = shared or {}
        self.exclude_warmup = bool(self.shared.get("EXCLUDE_WARMUP", False))
        self.beta_nll = float(self.train_params.get("beta_nll", self.shared.get("BETA_NLL", 0.0)))

    @classmethod
    def from_config(cls, params: dict, device: str, shared: dict = None) -> "TCNAdapter":
        arch = {k: params[k] for k in cls.ARCH_KEYS if k in params}
        if "distribution" in arch:
            arch.update(_DIST_TO_FLAGS[arch.pop("distribution")])
        model = TCN_channel(**arch).to(device)
        train_params = {k: params[k] for k in cls.TRAIN_KEYS if k in params}
        return cls(model, train_params, device, shared=shared)

    @classmethod
    def load(cls, params: dict, checkpoint: Path, device: str, shared: dict = None) -> "TCNAdapter":
        adapter = cls.from_config(params, device, shared=shared)
        adapter.model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        adapter.model.eval()
        return adapter

    def _warmup_slice(self, T):
        '''Time-index from which outputs are fully warmed up (causal RF filled). 0 when the
        toggle is off; clamped so at least one sample always survives.'''
        if not self.exclude_warmup:
            return 0
        return min(self.model.receptive_field - 1, T - 1)

    def _loss(self, xb, yb):
        '''The training objective on one batch: Gaussian/Student-t NLL when the
        model learns noise, plain MSE otherwise. With EXCLUDE_WARMUP the leading
        receptive-field samples are dropped before the loss is taken.'''
        s = self._warmup_slice(xb.shape[-1])
        if self.model.learn_noise:
            _, y_pred, y_pred_std, y_pred_nu = self.model(xb)
            residual = (yb - y_pred)[..., s:]
            std = y_pred_std[..., s:]
            if self.model.gaussian:
                return gaussian_nll(residual, std, beta=self.beta_nll)
            return students_t_loss(residual, std, y_pred_nu[..., s:], beta=self.beta_nll)
        return F.mse_loss(self.model(xb)[..., s:], yb[..., s:])

    def _val_loss(self, X_val, Y_val) -> float:
        self.model.eval()
        with torch.no_grad():
            loss = self._loss(X_val, Y_val).item()
        self.model.train()
        return loss

    def val_nll(self, X, Y):
        '''Validation negative log-likelihood for probabilistic (learn_noise)
        models; None otherwise (plain-MSE TCNs have no likelihood).'''
        if not self.model.learn_noise:
            return None
        return self._val_loss(X.to(self.device), Y.to(self.device))

    def fit(self, X, Y, X_val=None, Y_val=None) -> dict:
        X, Y = X.to(self.device), Y.to(self.device)
        has_val = X_val is not None
        if has_val:
            X_val, Y_val = X_val.to(self.device), Y_val.to(self.device)
        epochs = self.train_params["epochs"]
        batch_size = self.train_params["batch_size"]
        loader = DataLoader(TensorDataset(X, Y), batch_size=batch_size, shuffle=True)
        optimizer = optim.AdamW(self.model.parameters(), lr=float(self.train_params["lr"]))

        scheduler = None
        if "patience" in self.train_params:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min",
                factor=float(self.train_params.get("factor", 0.5)),
                patience=int(self.train_params["patience"]),
                min_lr=float(self.train_params.get("min_lr", 1e-6)),
            )

        self.model.train()
        history = {"loss": [], "lr": []}
        if has_val:
            history["val_loss"] = []
        for _ in range(epochs):
            epoch_loss, n_batches = 0.0, 0
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = self._loss(xb, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            train_loss = epoch_loss / n_batches
            history["loss"].append(train_loss)
            history["lr"].append(optimizer.param_groups[0]["lr"])
            if has_val:
                val_loss = self._val_loss(X_val, Y_val)
                history["val_loss"].append(val_loss)
            if scheduler is not None:
                scheduler.step(val_loss if has_val else train_loss)
        return history

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            return self.model(X.to(self.device))

    def num_params(self) -> int:
        return self.model.get_num_params()

    def save(self, path: Path) -> None:
        torch.save(self.model.state_dict(), path)


class LRUAdapter(TCNAdapter):
    '''Linear Recurrent Unit (deep linear-RNN / state-space) baseline.

    Trains exactly like the TCN — same AdamW loop, Gaussian/Student-t NLL or MSE
    objective, and probabilistic `distribution` switch — so it reuses everything
    on TCNAdapter and only swaps in the LRU_channel construction. LRU_channel
    shares TCN_channel's forward contract (mean, or (noisy, mean, std, nu)) and
    exposes `receptive_field` (=1: an RNN has no fixed warm-up window).
    '''
    name = "lru"
    ARCH_KEYS = ("state_dim", "hidden_dim", "n_layers", "dropout",
                 "r_min", "r_max", "max_phase",
                 "distribution", "learn_noise", "gaussian")
    # TRAIN_KEYS inherited from TCNAdapter (epochs, lr, batch_size, factor, patience, min_lr)

    @classmethod
    def from_config(cls, params: dict, device: str, shared: dict = None) -> "LRUAdapter":
        arch = {k: params[k] for k in cls.ARCH_KEYS if k in params}
        if "distribution" in arch:
            arch.update(_DIST_TO_FLAGS[arch.pop("distribution")])
        model = LRU_channel(**arch).to(device)
        train_params = {k: params[k] for k in cls.TRAIN_KEYS if k in params}
        return cls(model, train_params, device, shared=shared)


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

    @classmethod
    def load(cls, params: dict, checkpoint: Path, device: str, shared: dict = None) -> "GMPAdapter":
        adapter = cls.from_config(params, device, shared=shared)
        ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
        adapter.model.weights = ckpt["weights"]
        return adapter

    def fit(self, X, Y, X_val=None, Y_val=None) -> dict:
        self.model.fit(X, Y, **self.fit_params)
        return {}  # closed-form: no per-epoch history (val loss measured by orchestrator)

    def val_nll(self, X, Y):
        return None  # deterministic least-squares fit, no likelihood

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

    @classmethod
    def load(cls, params: dict, checkpoint: Path, device: str, shared: dict = None) -> "MockAdapter":
        return cls(params, shared=shared)

    def fit(self, X, Y, X_val=None, Y_val=None) -> dict:
        history = {"loss": [1.0, 0.5, 0.25]}
        if X_val is not None:
            history["val_loss"] = [1.1, 0.6, 0.35]
        return history

    def val_nll(self, X, Y):
        return None

    def predict(self, X):
        return X

    def num_params(self) -> int:
        return int(self.params["n"])

    def save(self, path: Path) -> None:
        Path(path).write_bytes(b"mock-checkpoint")


MODEL_REGISTRY = {
    TCNAdapter.name: TCNAdapter,
    LRUAdapter.name: LRUAdapter,
    GMPAdapter.name: GMPAdapter,
    MockAdapter.name: MockAdapter,
}
