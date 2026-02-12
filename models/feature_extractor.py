from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, Type

import numpy as np


class FeatureExtractor(Protocol):
    def clone(self) -> "FeatureExtractor": ...
    def fit(self, X: np.ndarray, y: np.ndarray) -> "FeatureExtractor": ...
    def transform(self, X: np.ndarray) -> np.ndarray: ...


@dataclass
class ChannelLogVar(FeatureExtractor):
    eps: float = 1e-12
    normalize_var: bool = True
    log_var: bool = True

    def __post_init__(self):
        if not isinstance(self.eps, float) or not np.isfinite(self.eps) or self.eps <= 0:
            raise ValueError(f"eps must be finite and > 0; got {self.eps!r}")
        if not isinstance(self.normalize_var, bool):
            raise ValueError(f"normalize_var must be a bool, got {self.normalize_var!r}")
        if not isinstance(self.log_var, bool):
            raise ValueError(f"log_var must be a bool, got {self.log_var!r}")

    def clone(self) -> "ChannelLogVar":
        return ChannelLogVar(normalize_var=self.normalize_var, log_var=self.log_var, eps=self.eps)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ChannelLogVar":
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if X.ndim != 3:
            raise ValueError(f"Expected X to have shape (N, Ch, Time); got {X.shape}")
        var = np.var(X, axis=-1, ddof=0)
        if self.normalize_var:
            var = var / (np.sum(var, axis=1, keepdims=True) + self.eps)
        return np.log(var + self.eps) if self.log_var else var


@dataclass
class CSPLogVar(FeatureExtractor):
    n_components: int = 4
    reg: float = 1e-10
    eps: float = 1e-12
    normalize_var: bool = True
    log_var: bool = True
    csp_filters_: list[np.ndarray] | None = None

    def __post_init__(self):
        if not isinstance(self.n_components, int) or self.n_components <= 0:
            raise ValueError(f"n_components must be a positive int; got {self.n_components!r}")
        if self.n_components % 2 != 0:
            raise ValueError(f"n_components must be even for CSP extremes selection; got {self.n_components}")
        if not isinstance(self.reg, float) or not np.isfinite(self.reg) or self.reg < 0:
            raise ValueError(f"reg must be finite and >= 0; got {self.reg!r}")
        if not isinstance(self.eps, float) or not np.isfinite(self.eps) or self.eps <= 0:
            raise ValueError(f"eps must be finite and > 0; got {self.eps!r}")
        if not isinstance(self.normalize_var, bool):
            raise ValueError(f"normalize_var must be a bool, got {self.normalize_var!r}")
        if not isinstance(self.log_var, bool):
            raise ValueError(f"log_var must be a bool, got {self.log_var!r}")

    def clone(self) -> "CSPLogVar":
        return CSPLogVar(n_components=self.n_components, reg=self.reg, normalize_var=self.normalize_var, log_var=self.log_var, eps=self.eps)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CSPLogVar":
        if X.ndim != 3:
            raise ValueError(f"Expected X to have shape (N, Ch, Time); got {X.shape}")
        y = np.asarray(y)
        classes = np.unique(y)
        if len(classes) < 2:
            raise ValueError("CSPLogVar requires at least 2 classes.")

        self.csp_filters_ = []
        if len(classes) == 2:
            self.csp_filters_.append(self._fit_csp_binary(X, y))
            return self
        
        for c in classes:
            y_bin = (y == c).astype(np.int8)
            if np.all(y_bin == 0) or np.all(y_bin == 1):
                raise ValueError(f"Class {c!r} has no samples; cannot fit CSP.")
            self.csp_filters_.append(self._fit_csp_binary(X, y_bin))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if X.ndim != 3:
            raise ValueError(f"Expected X to have shape (N, Ch, Time); got {X.shape}")
        if self.csp_filters_ is None or len(self.csp_filters_) == 0:
            raise RuntimeError("CSPLogVar not fitted")
        feats = []
        for csp_filter in self.csp_filters_:
            Z = np.einsum("nct,ck->nkt", X, csp_filter)  # (N, K, T)
            var = np.var(Z, axis=-1, ddof=0)
            if self.normalize_var:
                var = var / (np.sum(var, axis=1, keepdims=True) + self.eps)
            feats.append(np.log(var + self.eps) if self.log_var else var)
        return np.concatenate(feats, axis=1)

    @staticmethod
    def _compute_cov(x: np.ndarray) -> np.ndarray:
        C = x @ x.T
        tr = np.trace(C)
        return C / tr if tr > 0 else C

    def _fit_csp_binary(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError("CSP requires exactly 2 classes.")
        c0, c1 = classes[0], classes[1]
        X0 = X[y == c0]; X1 = X[y == c1]
        if len(X0) == 0 or len(X1) == 0:
            raise ValueError("Both classes must have at least one trial for CSP.")
        C0 = np.mean([self._compute_cov(x) for x in X0], axis=0)
        C1 = np.mean([self._compute_cov(x) for x in X1], axis=0)
        n_ch = C0.shape[0]
        if self.reg and self.reg > 0:
            C0 = C0 + self.reg * np.eye(n_ch)
            C1 = C1 + self.reg * np.eye(n_ch)
        Cc = C0 + C1
        evals, evecs = np.linalg.eig(np.linalg.solve(Cc, C0))
        evals, evecs = np.real(evals), np.real(evecs)
        evecs = evecs[:, np.argsort(evals)]
        k2 = self.n_components // 2
        return np.concatenate([evecs[:, :k2], evecs[:, -k2:]], axis=1)


class FeatureExtractorType(Enum):
    CSP_LOGVAR = CSPLogVar
    CHANNEL_LOGVAR = ChannelLogVar

    @property
    def cls(self) -> Type[FeatureExtractor]:
        return self.value
